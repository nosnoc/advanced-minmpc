#!/usr/bin/env python3
# Adapted from the doubletank benchmark in the CAMINO toolbox.
# Copyright (C) 2024 Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later
# Modified for advanced-minmpc; distribution changes documented 2026-10-31.
# Model source: CAMINO, create_double_tank_problem (doubletank).
# https://github.com/minlp-toolbox/CAMINO/blob/22ab5c75949fcb1bc99d1eb886fc2c456872c583/camino/problems/problem_collection.py
# Original model: R. Abbasi-Esfeden, W. Van Roy, and J. Swevers,
# Iterative Switching Time Optimization for Mixed-integer Optimal Control
# Problems, European Control Conference, 2023.
# https://doi.org/10.23919/ECC57647.2023.10178419

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import ThreePhaseOptions, ThreePhaseSolver


N = 50
DT = 1.0 / 30.0
MIQP_SOLVER = "daqp"
MILP_SOLVER = "highs"


@dataclass(slots=True)
class DoubleTankProblem:
    nlp: dict[str, Any]
    discrete: list[bool]
    x0: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    p: ca.DM
    integrator: ca.Function
    state_indices: list[list[int]]
    continuous_control_indices: list[list[int]]
    binary_indices: list[int]
    binary_intervals: list[int]
    initial_state: ca.DM
    demand: np.ndarray
    N: int
    dt: float

    def trajectories(
        self, decision: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(decision, dtype=float).reshape(-1)
        states = values[np.asarray(self.state_indices, dtype=int)]
        continuous = values[
            np.asarray(self.continuous_control_indices, dtype=int).reshape(-1)
        ]
        binary = values[np.asarray(self.binary_indices, dtype=int)]
        return states, continuous, binary


def demand_trajectory(length: int, dt: float = DT) -> np.ndarray:
    return 2.0 + 0.5 * np.sin(np.arange(length, dtype=float) * dt)


def build_double_tank_problem(
    initial_state: tuple[float, float] = (2.0, 2.5),
    N: int = N,
    dt: float = DT,
) -> DoubleTankProblem:
    """Build the CAMINO double-tank multiple-shooting problem."""

    epsilon = 1e-3
    alpha = 100.0
    beta = (1.0, 1.2)
    gamma = 10.0
    demand = demand_trajectory(N + 1, dt)

    x = ca.MX.sym("double_tank_x", 2)
    control = ca.MX.sym("double_tank_u", 2)
    rhs = ca.Function(
        "double_tank_rhs",
        [x, control],
        [
            ca.vertcat(
                gamma * control[0] + control[1] - ca.sqrt(x[0] + epsilon),
                ca.sqrt(x[0] + epsilon) - ca.sqrt(x[1] + epsilon),
            )
        ],
    )
    integrator_state = ca.MX.sym("double_tank_integrator_x", 2)
    integrator_control = ca.MX.sym("double_tank_integrator_u", 2)
    k1 = rhs(integrator_state, integrator_control)
    k2 = rhs(integrator_state + dt * k1 / 2.0, integrator_control)
    k3 = rhs(integrator_state + dt * k2 / 2.0, integrator_control)
    k4 = rhs(integrator_state + dt * k3, integrator_control)
    integrated_state = integrator_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    integrator = ca.Function(
        "double_tank_integrator",
        [integrator_state, integrator_control],
        [integrated_state],
        ["x0", "u"],
        ["xf"],
    )

    demand_parameter = ca.MX.sym("double_tank_demand", N)
    variables: list[ca.MX] = []
    guesses: list[Any] = []
    lower_variables: list[float] = []
    upper_variables: list[float] = []
    discrete: list[bool] = []
    constraints: list[ca.MX] = []
    state_indices: list[list[int]] = []
    continuous_control_indices: list[list[int]] = []
    binary_indices: list[int] = []
    binary_intervals: list[int] = []
    objective = 0
    offset = 0

    state = ca.MX.sym("X_0", 2)
    variables.append(state)
    guesses.append(ca.DM(initial_state))
    lower_variables.extend(initial_state)
    upper_variables.extend(initial_state)
    discrete.extend([False, False])
    state_indices.append([offset, offset + 1])
    offset += 2

    for k in range(N):
        stage_control = ca.MX.sym(f"U_{k}", 2)
        variables.append(stage_control)
        guesses.append(ca.DM([0.5, 0.5]))
        lower_variables.extend([0.0, 0.0])
        upper_variables.extend([1.0, gamma])
        discrete.extend([True, False])
        binary_indices.append(offset)
        continuous_control_indices.append([offset + 1])
        binary_intervals.append(k)
        offset += 2

        objective += dt * alpha * (state[1] - demand_parameter[k]) ** 2
        objective += dt * (beta[0] * gamma * stage_control[0] + beta[1] * stage_control[1])
        integrated = integrator(x0=state, u=stage_control)["xf"]

        next_state = ca.MX.sym(f"X_{k + 1}", 2)
        variables.append(next_state)
        guesses.append(ca.DM([0.5, 0.5]))
        lower_variables.extend([0.0, 0.0])
        upper_variables.extend([1e3, 1e3])
        discrete.extend([False, False])
        state_indices.append([offset, offset + 1])
        offset += 2
        constraints.append(integrated - next_state)
        state = next_state

    decision = ca.vertcat(*variables)
    constraint = ca.vertcat(*constraints)
    return DoubleTankProblem(
        nlp={"x": decision, "f": objective, "g": constraint, "p": demand_parameter},
        discrete=discrete,
        x0=ca.vertcat(*guesses),
        lbx=ca.DM(lower_variables),
        ubx=ca.DM(upper_variables),
        lbg=ca.DM.zeros(constraint.numel(), 1),
        ubg=ca.DM.zeros(constraint.numel(), 1),
        p=ca.DM(demand[:-1]),
        integrator=integrator,
        state_indices=state_indices,
        continuous_control_indices=continuous_control_indices,
        binary_indices=binary_indices,
        binary_intervals=binary_intervals,
        initial_state=ca.DM(initial_state),
        demand=demand,
        N=N,
        dt=dt,
    )


def three_phase_options(
    problem: DoubleTankProblem, reconstruction: str
) -> ThreePhaseOptions:
    if reconstruction not in {"miqp", "milp"}:
        raise ValueError("reconstruction must be 'miqp' or 'milp'.")
    miqp = reconstruction == "miqp"
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER if miqp else MILP_SOLVER,
        enable_miqp=miqp,
        enable_milp=not miqp,
        enable_pycombina=False,
        n_x=2,
        n_u=1,
        n_v=1,
        N=problem.N,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
        highs_options={"highs": {"output_flag": False}},
    )


def solve_bonmin(problem: DoubleTankProblem) -> dict[str, ca.DM]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "double_tank_bonmin",
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
            "print_time": False,
        },
    )
    return solver(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
        p=problem.p,
    )


def solve_three_phase(
    problem: DoubleTankProblem, reconstruction: str
) -> dict[str, ca.DM]:
    solver = ThreePhaseSolver(
        f"double_tank_{reconstruction}",
        problem.nlp,
        problem.discrete,
        three_phase_options(problem, reconstruction),
    )
    return solver.solve(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
        p=problem.p,
    ).solution


def solve_open_loop(problem: DoubleTankProblem) -> dict[str, dict[str, ca.DM]]:
    return {
        "Bonmin": solve_bonmin(problem),
        "Three-phase MIQP": solve_three_phase(problem, "miqp"),
        "Three-phase MILP": solve_three_phase(problem, "milp"),
    }


def plot_open_loop(
    problem: DoubleTankProblem, solutions: dict[str, dict[str, ca.DM]]
) -> None:
    state_time = np.arange(problem.N + 1) * problem.dt
    control_time = np.arange(problem.N) * problem.dt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flat
    styles = ("-", "--", "-.")
    for (label, solution), style in zip(solutions.items(), styles):
        states, continuous, binary = problem.trajectories(solution["x"])
        axes[0].plot(state_time, states[:, 0], style, label=label)
        axes[1].plot(state_time, states[:, 1], style, label=label)
        axes[2].step(control_time, binary, where="post", linestyle=style, label=label)
        axes[3].step(
            control_time, continuous, where="post", linestyle=style, label=label
        )

    for axis, title, ylabel in zip(
        axes,
        ("Tank 1 state", "Tank 2 state", "Binary pump control", "Continuous inflow"),
        ("$x_1$", "$x_2$", "$v$", "$u$"),
    ):
        axis.set_title(title)
        axis.set_xlabel("time")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_double_tank_problem()
    plot_open_loop(problem, solve_open_loop(problem))


if __name__ == "__main__":
    main()
