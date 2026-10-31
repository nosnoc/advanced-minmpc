#!/usr/bin/env python3
# Adapted from the unstable_ocp problem in the CAMINO toolbox.
# Copyright (C) 2024 Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later
# Modified for advanced-minmpc; distribution changes documented 2026-10-31.
# Model source: CAMINO, create_ocp_unstable_system (unstable_ocp).
# https://github.com/minlp-toolbox/CAMINO/blob/22ab5c75949fcb1bc99d1eb886fc2c456872c583/camino/problems/problem_collection.py
# Reference: A. Ghezzi, W. Van Roy, S. Sager, and M. Diehl,
# A Sequential Benders-based Mixed-Integer Quadratic Programming Algorithm
# and Its Implementation in the CAMINO Toolbox, 2026, Section 4.2.
# https://doi.org/10.1007/s12532-026-00331-4

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


N = 40
DT = 0.05
INITIAL_STATE = 0.9
REFERENCE = 0.7
MIN_UPTIME = 2
MIQP_SOLVER = "gurobi"  #
MILP_SOLVER = "highs"


@dataclass(slots=True)
class UnstableOcpProblem:
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
    initial_state: float
    reference: float
    previous_control: float
    second_previous_control: float
    N: int
    dt: float
    min_uptime: int

    def parameters(
        self,
        reference: float | None = None,
        previous_control: float | None = None,
        second_previous_control: float | None = None,
    ) -> ca.DM:
        return ca.DM(
            [
                self.reference if reference is None else reference,
                self.previous_control
                if previous_control is None
                else previous_control,
                self.second_previous_control
                if second_previous_control is None
                else second_previous_control,
            ]
        )

    def trajectories(self, decision: Any) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(decision, dtype=float).reshape(-1)
        states = values[np.asarray(self.state_indices, dtype=int).reshape(-1)]
        controls = values[np.asarray(self.binary_indices, dtype=int)]
        return states, controls


def build_unstable_ocp_problem(
    initial_state: float = INITIAL_STATE,
    reference: float = REFERENCE,
    N: int = N,
    dt: float = DT,
    previous_control: float = 0.0,
    second_previous_control: float = 0.0,
) -> UnstableOcpProblem:
    state_symbol = ca.MX.sym("unstable_rhs_x")
    control_symbol = ca.MX.sym("unstable_rhs_u")
    rhs = ca.Function(
        "unstable_rhs",
        [state_symbol, control_symbol],
        [state_symbol**3 - control_symbol],
    )
    integrator_state = ca.MX.sym("unstable_integrator_x")
    integrator_control = ca.MX.sym("unstable_integrator_u")
    k1 = rhs(integrator_state, integrator_control)
    k2 = rhs(integrator_state + dt * k1 / 2.0, integrator_control)
    k3 = rhs(integrator_state + dt * k2 / 2.0, integrator_control)
    k4 = rhs(integrator_state + dt * k3, integrator_control)
    integrated_state = integrator_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    integrator = ca.Function(
        "unstable_integrator",
        [integrator_state, integrator_control],
        [integrated_state],
        ["x0", "u"],
        ["xf"],
    )

    parameter = ca.MX.sym("unstable_parameter", 3)
    reference_parameter = parameter[0]
    previous_parameter = parameter[1]
    second_previous_parameter = parameter[2]
    variables: list[ca.MX] = []
    guesses: list[ca.DM] = []
    lower_variables: list[float] = []
    upper_variables: list[float] = []
    discrete: list[bool] = []
    constraints: list[ca.MX] = []
    lower_constraints: list[float] = []
    upper_constraints: list[float] = []
    state_indices: list[list[int]] = []
    binary_indices: list[int] = []
    binary_intervals: list[int] = []
    objective = 0
    offset = 0

    state = ca.MX.sym("X_0")
    variables.append(state)
    guesses.append(ca.DM(initial_state))
    lower_variables.append(initial_state)
    upper_variables.append(initial_state)
    discrete.append(False)
    state_indices.append([offset])
    offset += 1
    prior_control = previous_parameter
    prior_prior_control = second_previous_parameter

    for k in range(N):
        control = ca.MX.sym(f"U_{k}")
        variables.append(control)
        guesses.append(ca.DM(1.0))
        lower_variables.append(0.0)
        upper_variables.append(1.0)
        discrete.append(True)
        binary_indices.append(offset)
        binary_intervals.append(k)
        offset += 1

        constraints.append(-control + prior_control - prior_prior_control)
        lower_constraints.append(-ca.inf)
        upper_constraints.append(0.0)
        objective += (state - reference_parameter) ** 2
        integrated = integrator(x0=state, u=control)["xf"]

        next_state = ca.MX.sym(f"X_{k + 1}")
        variables.append(next_state)
        guesses.append(ca.DM(initial_state))
        lower_variables.append(-100.0)
        upper_variables.append(100.0)
        discrete.append(False)
        state_indices.append([offset])
        offset += 1
        constraints.append(integrated - next_state)
        lower_constraints.append(0.0)
        upper_constraints.append(0.0)
        state = next_state
        prior_prior_control = prior_control
        prior_control = control

    objective += (state - reference_parameter) ** 2
    decision = ca.vertcat(*variables)
    constraint = ca.vertcat(*constraints)
    problem = UnstableOcpProblem(
        nlp={"x": decision, "f": objective, "g": constraint, "p": parameter},
        discrete=discrete,
        x0=ca.vertcat(*guesses),
        lbx=ca.DM(lower_variables),
        ubx=ca.DM(upper_variables),
        lbg=ca.DM(lower_constraints),
        ubg=ca.DM(upper_constraints),
        p=ca.DM([reference, previous_control, second_previous_control]),
        integrator=integrator,
        state_indices=state_indices,
        continuous_control_indices=[],
        binary_indices=binary_indices,
        binary_intervals=binary_intervals,
        initial_state=initial_state,
        reference=reference,
        previous_control=previous_control,
        second_previous_control=second_previous_control,
        N=N,
        dt=dt,
        min_uptime=MIN_UPTIME,
    )
    validate_problem_dimensions(problem)
    return problem


def three_phase_options(
    problem: UnstableOcpProblem, reconstruction: str
) -> ThreePhaseOptions:
    if reconstruction not in {"miqp", "milp", "pycombina"}:
        raise ValueError("reconstruction must be 'miqp', 'milp', or 'pycombina'.")
    miqp = reconstruction == "miqp"
    milp = reconstruction == "milp"
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER if miqp else MILP_SOLVER,
        enable_miqp=miqp,
        enable_milp=milp,
        enable_pycombina=reconstruction == "pycombina",
        n_x=1,
        n_u=0,
        n_v=1,
        N=problem.N,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        ipopt_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1000,
            "print_time": False,
        },
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
        highs_options={"highs": {"output_flag": False}},
        pycombina_options={
            "time_grid": np.arange(problem.N + 1) * problem.dt,
            "min_up_times": problem.min_uptime * problem.dt,
            "b_bin_pre": problem.previous_control,
        },
    )


def validate_problem_dimensions(problem: UnstableOcpProblem) -> None:
    if len(problem.state_indices) != problem.N + 1:
        raise RuntimeError("The state index map has an inconsistent horizon length.")
    if len(problem.binary_indices) != problem.N:
        raise RuntimeError("The binary index map has an inconsistent horizon length.")
    if problem.p.numel() != 3:
        raise RuntimeError("The runtime parameter vector must contain three values.")


def validate_binary_trajectory(
    controls: Any,
    previous_control: float = 0.0,
    second_previous_control: float = 0.0,
    tolerance: float = 1e-5,
) -> None:
    values = np.asarray(controls, dtype=float).reshape(-1)
    if not np.all(np.abs(values - np.rint(values)) <= tolerance):
        raise RuntimeError("The reconstructed control trajectory is not integral.")
    prior = float(previous_control)
    prior_prior = float(second_previous_control)
    for value in np.rint(values):
        if -value + prior - prior_prior > tolerance:
            raise RuntimeError("The two-step minimum up-time constraint is violated.")
        prior_prior, prior = prior, float(value)


def solve_bonmin(problem: UnstableOcpProblem) -> dict[str, ca.DM]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "unstable_ocp_bonmin",
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
            "bonmin.linear_solver": "mumps",
            "print_time": False,
        },
    )
    solution = solver(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
        p=problem.p,
    )
    if not solver.stats().get("success", False):
        raise RuntimeError(
            f"Bonmin failed: {solver.stats().get('return_status', 'unknown status')}"
        )
    return solution


def solve_three_phase(
    problem: UnstableOcpProblem, reconstruction: str
) -> dict[str, ca.DM]:
    solver = ThreePhaseSolver(
        f"unstable_ocp_{reconstruction}",
        problem.nlp,
        problem.discrete,
        three_phase_options(problem, reconstruction),
    )
    result = solver.solve(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
        p=problem.p,
    )
    return result.solution


def solve_open_loop(
    problem: UnstableOcpProblem,
) -> dict[str, dict[str, ca.DM]]:
    solutions = {
        "Bonmin": solve_bonmin(problem),
        "Three-phase MIQP": solve_three_phase(problem, "miqp"),
        "Three-phase MILP": solve_three_phase(problem, "milp"),
    }
    for solution in solutions.values():
        states, controls = problem.trajectories(solution["x"])
        if states.size != problem.N + 1 or controls.size != problem.N:
            raise RuntimeError("A returned trajectory has inconsistent dimensions.")
        validate_binary_trajectory(
            controls,
            problem.previous_control,
            problem.second_previous_control,
        )
    return solutions


def plot_open_loop(
    problem: UnstableOcpProblem,
    solutions: dict[str, dict[str, ca.DM]],
) -> None:
    state_time = np.arange(problem.N + 1) * problem.dt
    control_time = np.arange(problem.N) * problem.dt
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    styles = ("-", "--", "-.", ":")
    for (label, solution), style in zip(solutions.items(), styles):
        states, controls = problem.trajectories(solution["x"])
        axes[0].plot(state_time, states, style, label=label)
        axes[1].step(
            control_time,
            controls,
            where="post",
            linestyle=style,
            label=label,
        )
    axes[0].plot(
        state_time,
        np.full(problem.N + 1, problem.reference),
        color="black",
        linestyle=(0, (1, 1)),
        label="reference",
    )
    axes[0].set_title("State trajectory")
    axes[0].set_ylabel("$x$")
    axes[1].set_title("Binary control")
    axes[1].set_ylabel("$u$")
    axes[1].set_yticks((0, 1))
    for axis in axes:
        axis.set_xlabel("time [s]")
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_unstable_ocp_problem()
    plot_open_loop(problem, solve_open_loop(problem))


if __name__ == "__main__":
    main()
