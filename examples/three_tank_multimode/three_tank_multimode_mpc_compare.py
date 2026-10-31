#!/usr/bin/env python3
# Adapted from the mintOC Three-Tank multimode benchmark.
# https://mintoc.de/index.php?title=Three_Tank_multimode_problem

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import AdvancedStepMPC, ThreePhaseOptions
from three_tank_multimode_ocp import (
    T,
    ThreeTankProblem,
    build_three_tank_problem,
    validate_one_hot,
)


N = 20
N_MPC = 100
M_O = 2
PYCOMBINA_WARM_START = False
SOLVE_MIQP_IN_PREPARATION = False
GUROBI_SEED = 0
GUROBI_SINGLE_THREAD = False


@dataclass(frozen=True, slots=True)
class Configuration:
    label: str
    M_o: int | None
    color: str
    linestyle: str


@dataclass(slots=True)
class Record:
    states: list[np.ndarray]
    controls: list[np.ndarray]
    accumulated_cost: list[float]
    rnlp_objectives: list[float]
    miqp_objectives: list[float]
    preparation_times: list[float]
    feedback_times: list[float]


def options(
    problem: ThreeTankProblem,
    configuration: Configuration,
) -> ThreePhaseOptions:
    gurobi = {
        "OutputFlag": 0,
        "LogToConsole": 0,
        "IntFeasTol": 1e-9,
        "Seed": GUROBI_SEED,
    }
    if GUROBI_SINGLE_THREAD:
        gurobi["Threads"] = 1
    use_pycombina = configuration.M_o is not None or PYCOMBINA_WARM_START
    return ThreePhaseOptions(
        mip_solver="gurobi",
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=use_pycombina,
        binary_controls_one_hot=True,
        n_x=3,
        n_u=0,
        n_v=3,
        N=problem.N,
        M_o=configuration.M_o,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=SOLVE_MIQP_IN_PREPARATION,
        solve_pnlp_in_feedback=False,
        ipopt_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1000,
            "print_time": False,
        },
        gurobi_options={"gurobi": gurobi},
        pycombina_options={
            "time_grid": np.arange(problem.N + 1) * problem.dt,
        },
    )


def first_control(problem: ThreeTankProblem, decision: Any) -> np.ndarray:
    values = np.asarray(decision, dtype=float).reshape(-1)
    control = values[np.asarray(problem.binary_indices[0], dtype=int)]
    validate_one_hot(control.reshape(1, -1))
    return np.rint(control)


def simulate_interval(
    problem: ThreeTankProblem,
    state: ca.DM,
    control: np.ndarray,
) -> tuple[ca.DM, float]:
    result = problem.integrator(
        x0=state,
        w=ca.DM(control),
        reference=problem.reference,
    )
    return ca.DM(result["xf"]), float(result["qf"])


def run_controller(
    problem: ThreeTankProblem,
    configuration: Configuration,
    n_mpc: int = N_MPC,
) -> Record:
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(
            x0=x,
            w=v,
            reference=problem.reference,
        )["xf"]

    controller = AdvancedStepMPC(
        "three_tank_compare_" + ("full" if configuration.M_o is None else "mo"),
        problem.nlp,
        problem.discrete,
        options(problem, configuration),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    record = Record(
        states=[np.asarray(state, dtype=float).reshape(-1)],
        controls=[],
        accumulated_cost=[0.0],
        rnlp_objectives=[],
        miqp_objectives=[],
        preparation_times=[],
        feedback_times=[],
    )

    for step in range(n_mpc):
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
            p=problem.p,
        )
        if not preparation.stats["rnlp"].get("success", False):
            raise RuntimeError(
                f"{configuration.label} RNLP failed at step {step}: "
                f"{preparation.stats['rnlp'].get('return_status', 'unknown status')}"
            )
        feedback = controller.feedback_phase(state, preparation, p=problem.p)
        if not feedback.stats["miqp"].get("success", False):
            raise RuntimeError(
                f"{configuration.label} MIQP failed at step {step}: "
                f"{feedback.stats['miqp'].get('return_status', 'unknown status')}"
            )

        guess = ca.DM(feedback.solution["x"])
        previous_solution = guess
        control = first_control(problem, guess)
        state, stage_cost = simulate_interval(problem, state, control)
        levels = np.asarray(state, dtype=float).reshape(-1)
        if np.any(levels < -1e-8):
            raise RuntimeError("The simulated tank levels became negative.")
        record.states.append(levels)
        record.controls.append(control)
        record.accumulated_cost.append(record.accumulated_cost[-1] + stage_cost)
        record.rnlp_objectives.append(float(preparation.rnlp_solution["f"]))
        record.miqp_objectives.append(
            float("nan")
            if feedback.miqp_solution is None
            else controller._mip_objective(
                preparation.quadratic_model,
                feedback.miqp_solution,
            )
        )
        record.preparation_times.append(preparation.timings["total_preparation"])
        record.feedback_times.append(feedback.timings["total_feedback"])

    validate_record(record, configuration.label, n_mpc)
    return record


def validate_record(record: Record, label: str, n_mpc: int) -> None:
    if np.asarray(record.states).shape != (n_mpc + 1, 3):
        raise RuntimeError(f"{label} has inconsistent state dimensions.")
    if np.asarray(record.controls).shape != (n_mpc, 3):
        raise RuntimeError(f"{label} has inconsistent control dimensions.")
    if np.asarray(record.accumulated_cost).shape != (n_mpc + 1,):
        raise RuntimeError(f"{label} has inconsistent cost dimensions.")
    if np.asarray(record.rnlp_objectives).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent RNLP-objective dimensions.")
    if np.asarray(record.miqp_objectives).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent MIQP-objective dimensions.")
    if np.asarray(record.preparation_times).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent preparation-time dimensions.")
    if np.asarray(record.feedback_times).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent feedback-time dimensions.")
    validate_one_hot(record.controls)


def print_summary(results: dict[Configuration, Record]) -> None:
    headings = (
        "Algorithm",
        "Final cost",
        "Mean prep [s]",
        "Max prep [s]",
        "Mean feedback [s]",
        "Max feedback [s]",
    )
    rows = []
    for configuration, record in results.items():
        preparation = np.asarray(record.preparation_times)
        feedback = np.asarray(record.feedback_times)
        rows.append(
            (
                configuration.label,
                f"{record.accumulated_cost[-1]:.6f}",
                f"{np.mean(preparation):.4f}",
                f"{np.max(preparation):.4f}",
                f"{np.mean(feedback):.4f}",
                f"{np.max(feedback):.4f}",
            )
        )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(values, widths))
        ) + " |"

    print("\nThree-Tank advanced-step MPC comparison")
    print(separator)
    print(line(headings))
    print(separator)
    for row in rows:
        print(line(row))
    print(separator)


def plot_trajectories(
    problem: ThreeTankProblem,
    results: dict[Configuration, Record],
) -> None:
    state_time = np.arange(N_MPC + 1) * problem.dt
    control_time = np.arange(N_MPC) * problem.dt
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for configuration, record in results.items():
        states = np.asarray(record.states)
        controls = np.asarray(record.controls)
        for index in range(3):
            axes[0, index].plot(
                state_time,
                states[:, index],
                color=configuration.color,
                linestyle=configuration.linestyle,
                label=configuration.label,
            )
            axes[1, index].step(
                control_time,
                controls[:, index],
                where="post",
                color=configuration.color,
                linestyle=configuration.linestyle,
                label=configuration.label,
            )
    axes[0, 1].axhline(
        float(problem.reference[0]), color="black", linestyle=":", label="Reference"
    )
    axes[0, 2].axhline(
        float(problem.reference[1]), color="black", linestyle=":", label="Reference"
    )
    for index in range(3):
        axes[0, index].set_title(rf"Tank level $x_{index + 1}$")
        axes[0, index].set_xlabel("Time [s]")
        axes[0, index].set_ylabel(rf"$x_{index + 1}$")
        axes[1, index].set_title(rf"Applied binary control $w_{index + 1}$")
        axes[1, index].set_xlabel("Time [s]")
        axes[1, index].set_ylabel(rf"$w_{index + 1}$")
        axes[1, index].set_yticks((0, 1))
        for axis in axes[:, index]:
            axis.grid(True, alpha=0.35)
            axis.legend()
    figure.tight_layout()


def plot_performance(results: dict[Configuration, Record]) -> None:
    mpc_steps = np.arange(1, N_MPC + 1)
    cost_steps = np.arange(N_MPC + 1)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat
    for configuration, record in results.items():
        style = {
            "color": configuration.color,
            "linestyle": configuration.linestyle,
            "label": configuration.label,
        }
        axes[0].plot(cost_steps, record.accumulated_cost, **style)
        axes[1].plot(mpc_steps, record.rnlp_objectives, **style)
        axes[2].plot(mpc_steps, record.miqp_objectives, **style)
        axes[3].plot(mpc_steps, record.preparation_times, **style)
        axes[4].plot(mpc_steps, record.feedback_times, **style)
    titles = (
        "Accumulated applied cost",
        "RNLP objective",
        "MIQP objective",
        "Preparation time",
        "Feedback time",
    )
    ylabels = (
        "Closed-loop cost",
        "Objective",
        "Shifted quadratic objective",
        "Time [s]",
        "Time [s]",
    )
    for axis, title, ylabel in zip(axes[:5], titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("MPC step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.35)
        axis.legend()
    axes[5].axis("off")
    figure.tight_layout()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not 0 <= M_O <= N:
        raise ValueError("M_O must satisfy 0 <= M_O <= N.")
    if not ca.has_conic("gurobi"):
        raise RuntimeError("The CasADi Gurobi conic plugin is unavailable.")
    if importlib.util.find_spec("pycombina") is None:
        raise RuntimeError("PyCombina is required for the M_o comparison.")

    problem = build_three_tank_problem(N=N, dt=T / N)
    configurations = (
        Configuration("AS-MIQP full", None, "#1f77b4", "-"),
        Configuration(f"AS-MIQP $M_o={M_O}$", M_O, "#d62728", "--"),
    )
    results = {
        configuration: run_controller(problem, configuration)
        for configuration in configurations
    }
    print_summary(results)
    plot_trajectories(problem, results)
    plot_performance(results)
    plt.show()


if __name__ == "__main__":
    main()
