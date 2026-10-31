#!/usr/bin/env python3
# Adapted from the mintOC Three-Tank multimode benchmark.
# https://mintoc.de/index.php?title=Three_Tank_multimode_problem

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


T = 12.0
N = 150
DT = T / N
#INITIAL_STATE = (1.0, 1.5, 2.0)
INITIAL_STATE = (2.0, 2.0, 2.0)
REFERENCE = (3.0, 3.0)
MIQP_SOLVER = "gurobi"
MILP_SOLVER = "gurobi"
RUN_BONMIN = False


@dataclass(slots=True)
class ThreeTankProblem:
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
    binary_indices: list[list[int]]
    binary_intervals: list[int]
    initial_state: ca.DM
    reference: ca.DM
    N: int
    dt: float

    def parameters(self, reference: Any | None = None) -> ca.DM:
        return ca.DM(self.reference if reference is None else reference)

    def trajectories(self, decision: Any) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(decision, dtype=float).reshape(-1)
        states = values[np.asarray(self.state_indices, dtype=int)]
        controls = values[np.asarray(self.binary_indices, dtype=int)]
        return states, controls


def build_three_tank_problem(
    initial_state: tuple[float, float, float] = INITIAL_STATE,
    reference: tuple[float, float] = REFERENCE,
    N: int = N,
    dt: float = DT,
) -> ThreeTankProblem:
    state_symbol = ca.MX.sym("three_tank_rhs_x", 3)
    control_symbol = ca.MX.sym("three_tank_rhs_w", 3)
    reference_symbol = ca.MX.sym("three_tank_rhs_reference", 2)
    c1, c2, c3 = 1.0, 2.0, 0.8
    k1, k3 = 2.0, 1.0
    root_x1 = ca.sqrt(state_symbol[0])
    transfer = control_symbol[2] * ca.sqrt(c3 * state_symbol[0])
    rhs_expression = ca.vertcat(
        -root_x1 + c1 * control_symbol[0] + c2 * control_symbol[1] - transfer,
        root_x1 - ca.sqrt(state_symbol[1]),
        ca.sqrt(state_symbol[1]) - ca.sqrt(state_symbol[2]) + transfer,
    )
    running_cost = (
        k1 * (state_symbol[1] - reference_symbol[0]) ** 2
        + k3 * (state_symbol[2] - reference_symbol[1]) ** 2
    )
    rhs = ca.Function(
        "three_tank_rhs",
        [state_symbol, control_symbol, reference_symbol],
        [rhs_expression, running_cost],
        ["x", "w", "reference"],
        ["xdot", "cost"],
    )

    integrator_state = ca.MX.sym("three_tank_integrator_x", 3)
    integrator_control = ca.MX.sym("three_tank_integrator_w", 3)
    integrator_reference = ca.MX.sym("three_tank_integrator_reference", 2)
    stage_1 = rhs(integrator_state, integrator_control, integrator_reference)
    stage_2 = rhs(
        integrator_state + dt * stage_1[0] / 2.0,
        integrator_control,
        integrator_reference,
    )
    stage_3 = rhs(
        integrator_state + dt * stage_2[0] / 2.0,
        integrator_control,
        integrator_reference,
    )
    stage_4 = rhs(
        integrator_state + dt * stage_3[0],
        integrator_control,
        integrator_reference,
    )
    integrated_state = integrator_state + dt * (
        stage_1[0] + 2.0 * stage_2[0] + 2.0 * stage_3[0] + stage_4[0]
    ) / 6.0
    integrated_cost = dt * (
        stage_1[1] + 2.0 * stage_2[1] + 2.0 * stage_3[1] + stage_4[1]
    ) / 6.0
    integrator = ca.Function(
        "three_tank_integrator",
        [integrator_state, integrator_control, integrator_reference],
        [integrated_state, integrated_cost],
        ["x0", "w", "reference"],
        ["xf", "qf"],
    )

    parameter = ca.MX.sym("three_tank_parameter", 2)
    variables: list[ca.MX] = []
    guesses: list[ca.DM] = []
    lower_variables: list[float] = []
    upper_variables: list[float] = []
    discrete: list[bool] = []
    constraints: list[ca.MX] = []
    state_indices: list[list[int]] = []
    binary_indices: list[list[int]] = []
    binary_intervals: list[int] = []
    objective: Any = 0
    offset = 0

    state = ca.MX.sym("X_0", 3)
    state_guess = ca.DM(initial_state)
    variables.append(state)
    guesses.append(state_guess)
    lower_variables.extend(initial_state)
    upper_variables.extend(initial_state)
    discrete.extend([False] * 3)
    state_indices.append(list(range(offset, offset + 3)))
    offset += 3

    relaxed_control = ca.DM.ones(3, 1) / 3.0
    for interval in range(N):
        control = ca.MX.sym(f"W_{interval}", 3)
        variables.append(control)
        guesses.append(relaxed_control)
        lower_variables.extend([0.0] * 3)
        upper_variables.extend([1.0] * 3)
        discrete.extend([True] * 3)
        binary_indices.append(list(range(offset, offset + 3)))
        binary_intervals.extend([interval] * 3)
        offset += 3

        integrated = integrator(x0=state, w=control, reference=parameter)
        objective += integrated["qf"]
        constraints.append(ca.sum1(control) - 1.0)

        next_state = ca.MX.sym(f"X_{interval + 1}", 3)
        state_guess = ca.DM(
            integrator(
                x0=state_guess,
                w=relaxed_control,
                reference=ca.DM(reference),
            )["xf"]
        )
        variables.append(next_state)
        guesses.append(state_guess)
        lower_variables.extend([0.0] * 3)
        upper_variables.extend([ca.inf] * 3)
        discrete.extend([False] * 3)
        state_indices.append(list(range(offset, offset + 3)))
        offset += 3
        constraints.append(integrated["xf"] - next_state)
        state = next_state

    decision = ca.vertcat(*variables)
    constraint = ca.vertcat(*constraints)
    problem = ThreeTankProblem(
        nlp={"x": decision, "f": objective, "g": constraint, "p": parameter},
        discrete=discrete,
        x0=ca.vertcat(*guesses),
        lbx=ca.DM(lower_variables),
        ubx=ca.DM(upper_variables),
        lbg=ca.DM.zeros(constraint.numel(), 1),
        ubg=ca.DM.zeros(constraint.numel(), 1),
        p=ca.DM(reference),
        integrator=integrator,
        state_indices=state_indices,
        continuous_control_indices=[],
        binary_indices=binary_indices,
        binary_intervals=binary_intervals,
        initial_state=ca.DM(initial_state),
        reference=ca.DM(reference),
        N=N,
        dt=dt,
    )
    validate_problem_dimensions(problem)
    return problem


def three_phase_options(
    problem: ThreeTankProblem,
    reconstruction: str,
) -> ThreePhaseOptions:
    if reconstruction not in {"miqp", "milp"}:
        raise ValueError("reconstruction must be 'miqp', 'milp', or 'pycombina'.")
    miqp = reconstruction == "miqp"
    milp = reconstruction == "milp"
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER if miqp else MILP_SOLVER,
        enable_miqp=miqp,
        enable_milp=milp,
        binary_controls_one_hot=True,
        n_x=3,
        n_u=0,
        n_v=3,
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
        gurobi_options={
            "gurobi": {
                "OutputFlag": 0,
                "LogToConsole": 0,
                "IntFeasTol": 1e-9,
            }
        },
        highs_options={"highs": {"output_flag": False}},
        pycombina_options={"time_grid": np.arange(problem.N + 1) * problem.dt},
    )


def validate_problem_dimensions(problem: ThreeTankProblem) -> None:
    if np.asarray(problem.state_indices).shape != (problem.N + 1, 3):
        raise RuntimeError("The state index map has inconsistent dimensions.")
    if np.asarray(problem.binary_indices).shape != (problem.N, 3):
        raise RuntimeError("The binary index map has inconsistent dimensions.")
    if problem.p.numel() != 2:
        raise RuntimeError("The parameter vector must contain two references.")


def validate_one_hot(controls: Any, tolerance: float = 1e-5) -> None:
    values = np.asarray(controls, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("The mode trajectory must have three columns.")
    if not np.all(np.abs(values - np.rint(values)) <= tolerance):
        raise RuntimeError("The reconstructed mode trajectory is not integral.")
    if not np.allclose(np.sum(values, axis=1), 1.0, atol=tolerance):
        raise RuntimeError("The reconstructed mode trajectory is not one-hot.")


def solve_bonmin(problem: ThreeTankProblem) -> dict[str, ca.DM]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "three_tank_multimode_bonmin",
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
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
    stats = solver.stats()
    if not stats.get("success", False):
        raise RuntimeError(
            f"Bonmin failed: {stats.get('return_status', 'unknown status')}"
        )
    return solution


def solve_three_phase(
    problem: ThreeTankProblem,
    reconstruction: str,
) -> dict[str, ca.DM]:
    solver = ThreePhaseSolver(
        f"three_tank_multimode_{reconstruction}",
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
    problem: ThreeTankProblem,
) -> dict[str, dict[str, ca.DM]]:
    solutions: dict[str, dict[str, ca.DM]] = {}
    if RUN_BONMIN:
        solutions["Bonmin"] = solve_bonmin(problem)
    solutions["Three-phase MIQP"] = solve_three_phase(problem, "miqp")
    solutions["Three-phase MILP"] = solve_three_phase(problem, "milp")
    for label, solution in solutions.items():
        states, controls = problem.trajectories(solution["x"])
        if states.shape != (problem.N + 1, 3):
            raise RuntimeError(f"{label} returned inconsistent state dimensions.")
        validate_one_hot(controls)
        print(f"{label} objective: {float(solution['f']):.6f}")
    return solutions


def plot_open_loop(
    problem: ThreeTankProblem,
    solutions: dict[str, dict[str, ca.DM]],
) -> None:
    state_time = np.arange(problem.N + 1) * problem.dt
    control_time = np.arange(problem.N) * problem.dt
    figure, axes = plt.subplots(4, 2, figsize=(12, 13))
    axes = axes.flat
    styles = ("-", "--", "-.", ":")
    for (label, solution), style in zip(solutions.items(), styles):
        states, controls = problem.trajectories(solution["x"])
        for index in range(3):
            axes[index].plot(
                state_time,
                states[:, index],
                linestyle=style,
                label=label,
            )
            axes[index + 3].step(
                control_time,
                controls[:, index],
                where="post",
                linestyle=style,
                label=label,
            )
        axes[6].step(
            control_time,
            np.argmax(controls, axis=1) + 1,
            where="post",
            linestyle=style,
            label=label,
        )
    axes[1].axhline(
        float(problem.reference[0]),
        color="black",
        linestyle=":",
        label="reference",
    )
    axes[2].axhline(
        float(problem.reference[1]),
        color="black",
        linestyle=":",
        label="reference",
    )
    axes[7].axis("off")
    titles = (
        "Tank 1 level",
        "Tank 2 level",
        "Tank 3 level",
        "Mode $w_1$",
        "Mode $w_2$",
        "Mode $w_3$",
        "Selected mode",
    )
    ylabels = ("$x_1$", "$x_2$", "$x_3$", "$w_1$", "$w_2$", "$w_3$", "mode")
    for axis, title, ylabel in zip(axes[:7], titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("time")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()
    for axis in axes[3:6]:
        axis.set_yticks((0, 1))
    axes[6].set_yticks((1, 2, 3))
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_three_tank_problem()
    plot_open_loop(problem, solve_open_loop(problem))


if __name__ == "__main__":
    main()
