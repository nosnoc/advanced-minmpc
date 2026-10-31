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
from double_tank_ocp import (
    DoubleTankProblem,
    build_double_tank_problem,
    demand_trajectory,
)

MIQP_SOLVER = "gurobi"  #   
N_MPC = 80
RUN_BONMIN = False
WARM_START_BONMIN = True


def advanced_options(
    problem: DoubleTankProblem, M_f: int | None
) -> ThreePhaseOptions:
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER,
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=False,
        n_x=2,
        n_u=1,
        n_v=1,
        N=problem.N,
        M_f=M_f,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=False,
        solve_pnlp_in_feedback=False,
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
    )


def first_control(
    problem: DoubleTankProblem, decision: ca.DM
) -> tuple[float, float]:
    binary = float(decision[problem.binary_indices[0]])
    continuous = float(decision[problem.continuous_control_indices[0][0]])
    return binary, continuous


def apply_control(
    problem: DoubleTankProblem,
    state: ca.DM,
    binary: float,
    continuous: float,
) -> ca.DM:
    return problem.integrator(x0=state, u=ca.DM([binary, continuous]))["xf"]


def empty_record(problem: DoubleTankProblem) -> dict[str, list]:
    return {
        "states": [np.asarray(problem.initial_state, dtype=float).reshape(-1)],
        "continuous": [],
        "binary": [],
        "total_preparation": [],
        "total_feedback": [],
    }


def run_bonmin(
    problem: DoubleTankProblem, demand: np.ndarray
) -> dict[str, list]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "double_tank_mpc_bonmin",
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
            "print_time": False,
        },
    )
    logger = logging.getLogger(__name__)
    state = ca.DM(problem.initial_state)
    previous_solution = None
    record = empty_record(problem)
    initial_indices = np.asarray(problem.state_indices[0], dtype=int)

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
        lbx[initial_indices] = state
        ubx[initial_indices] = state
        start = perf_counter()
        solution = solver(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=ca.DM(demand[step : step + problem.N]),
        )
        elapsed = perf_counter() - start
        stats = solver.stats()
        logger.info(
            "Bonmin success=%s time=%.4fs status=%s",
            stats.get("success"),
            elapsed,
            stats.get("return_status", "unknown"),
        )
        guess = ca.DM(solution["x"])
        previous_solution = guess
        binary, continuous = first_control(problem, guess)
        state = apply_control(problem, state, binary, continuous)
        record["states"].append(np.asarray(state, dtype=float).reshape(-1))
        record["binary"].append(binary)
        record["continuous"].append(continuous)
        record["total_preparation"].append(0.0)
        record["total_feedback"].append(elapsed)
    return record


def run_three_phase(
    problem: DoubleTankProblem, demand: np.ndarray
) -> dict[str, list]:
    solver = ThreePhaseSolver(
        "double_tank_three_phase",
        problem.nlp,
        problem.discrete,
        advanced_options(problem, None),
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    record = empty_record(problem)
    initial_indices = np.asarray(problem.state_indices[0], dtype=int)

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
        lbx[initial_indices] = state
        ubx[initial_indices] = state
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=ca.DM(demand[step : step + problem.N]),
        )
        guess = ca.DM(result.solution["x"])
        binary, continuous = first_control(problem, guess)
        state = apply_control(problem, state, binary, continuous)
        record["states"].append(np.asarray(state, dtype=float).reshape(-1))
        record["binary"].append(binary)
        record["continuous"].append(continuous)
        record["total_preparation"].append(0.0)
        record["total_feedback"].append(result.timings["total"])
    return record


def run_advanced(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    M_f: int | None,
    name: str,
) -> dict[str, list]:
    def integrate(x: ca.DM, u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(x0=x, u=ca.vertcat(v[0], u[0]))["xf"]

    controller = AdvancedStepMPC(
        name,
        problem.nlp,
        problem.discrete,
        advanced_options(problem, M_f),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous = None
    record = empty_record(problem)

    for step in range(N_MPC):
        parameter = ca.DM(demand[step : step + problem.N])
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous is None else None,
            previous_solution=previous,
            p=parameter,
        )
        feedback = controller.feedback_phase(state, preparation, p=parameter)
        guess = ca.DM(feedback.solution["x"])
        previous = guess
        binary, continuous = first_control(problem, guess)
        state = apply_control(problem, state, binary, continuous)
        record["states"].append(np.asarray(state, dtype=float).reshape(-1))
        record["binary"].append(binary)
        record["continuous"].append(continuous)
        record["total_preparation"].append(
            preparation.timings["total_preparation"]
        )
        record["total_feedback"].append(feedback.timings["total_feedback"])
    return record


def as_array(record: dict[str, list], key: str) -> np.ndarray:
    return np.asarray(record[key], dtype=float)


def plot_closed_loop(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    results: dict[str, dict[str, list]],
) -> None:
    state_steps = np.arange(N_MPC + 1)
    control_steps = np.arange(N_MPC)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flat
    styles = ("-", "--", "-.", ":", "-")

    for (label, record), style in zip(results.items(), styles):
        states = as_array(record, "states")
        axes[0].plot(state_steps, states[:, 0], style, label=label)
        axes[1].plot(state_steps, states[:, 1], style, label=label)
        axes[2].step(
            control_steps,
            as_array(record, "binary"),
            where="post",
            linestyle=style,
            label=label,
        )
        axes[3].step(
            control_steps,
            as_array(record, "continuous"),
            where="post",
            linestyle=style,
            label=label,
        )
        axes[4].plot(
            control_steps, as_array(record, "total_preparation"), style, label=label
        )
        axes[5].plot(
            control_steps, as_array(record, "total_feedback"), style, label=label
        )

    axes[1].plot(
        state_steps,
        demand[: N_MPC + 1],
        ":",
        color="black",
        label="demand",
    )
    for axis, title, ylabel in zip(
        axes,
        (
            "Tank 1 closed-loop state",
            "Tank 2 closed-loop state",
            "Binary pump control",
            "Continuous inflow",
            "Preparation time",
            "Feedback time",
        ),
        ("$x_1$", "$x_2$", "$v$", "$u$", "time [s]", "time [s]"),
    ):
        axis.set_title(title)
        axis.set_xlabel("MPC step")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_double_tank_problem()
    demand = demand_trajectory(problem.N + N_MPC, problem.dt)
    results = {
        "Three-phase MIQP": run_three_phase(problem, demand),
        "Advanced MIQP": run_advanced(problem, demand, None, "double_tank_advanced_full"),
        "Advanced MIQP ($M_f=2$)": run_advanced(problem, demand, 2, "double_tank_advanced_mf2"),
        "Advanced MIQP ($M_f=4$)": run_advanced(problem, demand, 4, "double_tank_advanced_mf4"),
        "Advanced MIQP ($M_f=10$)": run_advanced(problem, demand, 10,"double_tank_advanced_mf10"),
    }
    if RUN_BONMIN:
        results["Bonmin"] = run_bonmin(problem, demand)
    plot_closed_loop(problem, demand, results)


if __name__ == "__main__":
    main()
