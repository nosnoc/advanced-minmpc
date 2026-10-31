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

import logging
from pathlib import Path
import sys
from time import perf_counter

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import (
    AdvancedStepMPC,
    ThreePhaseOptions,
    ThreePhaseSolver,
    prepare_mpc_initial_guess,
    shift_mpc_trajectory,
)
from unstable_ocp_ocp import (
    MIQP_SOLVER,
    N,
    UnstableOcpProblem,
    build_unstable_ocp_problem,
    validate_binary_trajectory,
)


PREDICTION_HORIZON = N
N_MPC = 40
RUN_BONMIN = False
WARM_START_BONMIN = True


def mpc_options(
    problem: UnstableOcpProblem,
    M_f: int | None,
) -> ThreePhaseOptions:
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER,
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=False,
        n_x=1,
        n_u=0,
        n_v=1,
        N=problem.N,
        M_f=M_f,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=True,
        solve_pnlp_in_feedback=False,
        ipopt_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1000,
            "print_time": False,
        },
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
    )


def parameter_vector(
    problem: UnstableOcpProblem,
    previous_control: float,
    second_previous_control: float,
) -> ca.DM:
    return problem.parameters(
        problem.reference,
        previous_control,
        second_previous_control,
    )


def first_control(problem: UnstableOcpProblem, decision: ca.DM) -> float:
    value = float(decision[problem.binary_indices[0]])
    if abs(value - round(value)) > 1e-5:
        raise RuntimeError("The first reconstructed MPC control is not integral.")
    return float(round(value))


def apply_control(
    problem: UnstableOcpProblem,
    state: ca.DM,
    control: float,
) -> ca.DM:
    return ca.DM(problem.integrator(x0=state, u=control)["xf"])


def empty_record(problem: UnstableOcpProblem) -> dict[str, list[float]]:
    return {
        "states": [problem.initial_state],
        "binary": [],
        "total_preparation": [],
        "total_feedback": [],
    }


def append_step(
    problem: UnstableOcpProblem,
    record: dict[str, list[float]],
    state: ca.DM,
    control: float,
    preparation_time: float,
    feedback_time: float,
) -> ca.DM:
    next_state = apply_control(problem, state, control)
    record["states"].append(float(next_state))
    record["binary"].append(control)
    record["total_preparation"].append(preparation_time)
    record["total_feedback"].append(feedback_time)
    return next_state


def validate_applied_control(
    control: float,
    previous_control: float,
    second_previous_control: float,
) -> None:
    validate_binary_trajectory(
        [control],
        previous_control,
        second_previous_control,
    )


def run_bonmin(problem: UnstableOcpProblem) -> dict[str, list[float]]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "unstable_ocp_mpc_bonmin",
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
    state = ca.DM(problem.initial_state)
    previous_solution = None
    previous_control = problem.previous_control
    second_previous_control = problem.second_previous_control
    initial_index = problem.state_indices[0][0]
    record = empty_record(problem)

    for step in range(N_MPC):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            problem.continuous_control_indices,
            problem.binary_indices,
            state,
            WARM_START_BONMIN,
        )
        lbx, ubx = ca.DM(problem.lbx), ca.DM(problem.ubx)
        lbx[initial_index] = state
        ubx[initial_index] = state
        start = perf_counter()
        solution = solver(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=parameter_vector(problem, previous_control, second_previous_control),
        )
        elapsed = perf_counter() - start
        stats = solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(
                f"Bonmin failed at MPC step {step}: "
                f"{stats.get('return_status', 'unknown status')}"
            )
        guess = ca.DM(solution["x"])
        previous_solution = guess
        control = first_control(problem, guess)
        validate_applied_control(
            control,
            previous_control,
            second_previous_control,
        )
        state = append_step(problem, record, state, control, 0.0, elapsed)
        second_previous_control, previous_control = previous_control, control
    validate_record(problem, record)
    return record


def run_three_phase(problem: UnstableOcpProblem) -> dict[str, list[float]]:
    solver = ThreePhaseSolver(
        "unstable_ocp_three_phase_mpc",
        problem.nlp,
        problem.discrete,
        mpc_options(problem, None),
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_control = problem.previous_control
    second_previous_control = problem.second_previous_control
    initial_index = problem.state_indices[0][0]
    record = empty_record(problem)

    for step in range(N_MPC):
        if step:
            guess = shift_mpc_trajectory(
                guess,
                problem.state_indices,
                problem.continuous_control_indices,
                problem.binary_indices,
                state,
            )
        lbx, ubx = ca.DM(problem.lbx), ca.DM(problem.ubx)
        lbx[initial_index] = state
        ubx[initial_index] = state
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=parameter_vector(problem, previous_control, second_previous_control),
        )
        guess = ca.DM(result.solution["x"])
        control = first_control(problem, guess)
        validate_applied_control(
            control,
            previous_control,
            second_previous_control,
        )
        state = append_step(
            problem,
            record,
            state,
            control,
            0.0,
            result.timings["total"],
        )
        second_previous_control, previous_control = previous_control, control
    validate_record(problem, record)
    return record


def run_advanced(
    problem: UnstableOcpProblem,
    M_f: int | None,
    name: str,
) -> dict[str, list[float]]:
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(x0=x, u=v[0])["xf"]

    controller = AdvancedStepMPC(
        name,
        problem.nlp,
        problem.discrete,
        mpc_options(problem, M_f),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    previous_control = problem.previous_control
    second_previous_control = problem.second_previous_control
    record = empty_record(problem)

    for _step in range(N_MPC):
        parameter = parameter_vector(
            problem,
            previous_control,
            second_previous_control,
        )
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
            p=parameter,
        )
        feedback = controller.feedback_phase(state, preparation, p=parameter)
        guess = ca.DM(feedback.solution["x"])
        previous_solution = guess
        control = first_control(problem, guess)
        validate_applied_control(
            control,
            previous_control,
            second_previous_control,
        )
        state = append_step(
            problem,
            record,
            state,
            control,
            preparation.timings["total_preparation"],
            feedback.timings["total_feedback"],
        )
        second_previous_control, previous_control = previous_control, control
    validate_record(problem, record)
    return record


def validate_record(
    problem: UnstableOcpProblem,
    record: dict[str, list[float]],
) -> None:
    expected = {
        "states": N_MPC + 1,
        "binary": N_MPC,
        "total_preparation": N_MPC,
        "total_feedback": N_MPC,
    }
    for key, length in expected.items():
        if len(record[key]) != length:
            raise RuntimeError(f"The {key} record has an inconsistent length.")
    validate_binary_trajectory(
        record["binary"],
        problem.previous_control,
        problem.second_previous_control,
    )


def as_array(record: dict[str, list[float]], key: str) -> np.ndarray:
    return np.asarray(record[key], dtype=float)


def plot_closed_loop(
    problem: UnstableOcpProblem,
    results: dict[str, dict[str, list[float]]],
) -> None:
    state_time = np.arange(N_MPC + 1) * problem.dt
    control_time = np.arange(N_MPC) * problem.dt
    mpc_steps = np.arange(N_MPC)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flat
    styles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    for (label, record), style in zip(results.items(), styles):
        axes[0].plot(
            state_time,
            as_array(record, "states"),
            linestyle=style,
            label=label,
        )
        axes[1].step(
            control_time,
            as_array(record, "binary"),
            where="post",
            linestyle=style,
            label=label,
        )
        axes[2].plot(
            mpc_steps,
            as_array(record, "total_preparation"),
            linestyle=style,
            label=label,
        )
        axes[3].plot(
            mpc_steps,
            as_array(record, "total_feedback"),
            linestyle=style,
            label=label,
        )
    axes[0].plot(
        state_time,
        np.full(N_MPC + 1, problem.reference),
        color="black",
        linestyle=(0, (1, 1)),
        label="reference",
    )
    for axis, title, xlabel, ylabel in zip(
        axes,
        (
            "Closed-loop state",
            "Applied binary control",
            "Preparation time",
            "Feedback time",
        ),
        ("time [s]", "time [s]", "MPC step", "MPC step"),
        ("$x$", "$u$", "time [s]", "time [s]"),
    ):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()
    axes[1].set_yticks((0, 1))
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_unstable_ocp_problem(N=PREDICTION_HORIZON)
    results = {
        "Classic three-phase MIQP": run_three_phase(problem),
        "Advanced MIQP": run_advanced(
            problem,
            None,
            "unstable_ocp_advanced_full",
        ),
        "Advanced MIQP ($M_f=2$)": run_advanced(
            problem,
            2,
            "unstable_ocp_advanced_mf2",
        ),
        "Advanced MIQP ($M_f=10$)": run_advanced(
            problem,
            10,
            "unstable_ocp_advanced_mf10",
        ),
    }
    if RUN_BONMIN:
        results["Bonmin"] = run_bonmin(problem)
    plot_closed_loop(problem, results)


if __name__ == "__main__":
    main()
