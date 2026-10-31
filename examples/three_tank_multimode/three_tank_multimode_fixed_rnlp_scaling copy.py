#!/usr/bin/env python3
# Scaling experiment for the mintOC Three-Tank multimode benchmark.
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

from mpc_implementation import AdvancedStepMPC, ThreePhaseOptions, ThreePhaseSolver
from mpc_implementation.results import PreparationResult, QuadraticModel
from three_tank_multimode_ocp import T, ThreeTankProblem, build_three_tank_problem

# TODO: with 60, 40 there is case where the relaxed value has a worse value than the fixed tail value. needs investigationN = 60
N = 40
HORIZON_VALUES = tuple(range(2, N, 2)) + (N,)
LOG_SCALE = False
OBJECTIVE_TOL = 1e-6
GUROBI_SEED = 0
GUROBI_SINGLE_THREAD = False


@dataclass(frozen=True, slots=True)
class ScalingResult:
    method: str
    horizon: int
    objective: float
    solve_time: float
    active_binaries: int


def gurobi_options() -> dict[str, Any]:
    settings: dict[str, Any] = {
        "OutputFlag": 0,
        "LogToConsole": 0,
        "IntFeasTol": 1e-9,
        "Seed": GUROBI_SEED,
    }
    if GUROBI_SINGLE_THREAD:
        settings["Threads"] = 1
    return {"gurobi": settings}


def solver_options(
    problem: ThreeTankProblem,
    *,
    M_f: int | None = None,
    M_o: int | None = None,
    enable_pycombina: bool = False,
) -> ThreePhaseOptions:
    return ThreePhaseOptions(
        mip_solver="gurobi",
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=enable_pycombina,
        binary_controls_one_hot=True,
        n_x=3,
        n_u=0,
        n_v=3,
        N=problem.N,
        M_f=M_f,
        M_o=M_o,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=False,
        solve_pnlp_in_feedback=False,
        ipopt_options={
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1000,
            "print_time": False,
        },
        gurobi_options=gurobi_options(),
        pycombina_options={
            "time_grid": np.arange(problem.N + 1) * problem.dt,
        },
    )


def frozen_preparation(problem: ThreeTankProblem) -> PreparationResult:
    controller = AdvancedStepMPC(
        "three_tank_fixed_rnlp",
        problem.nlp,
        problem.discrete,
        solver_options(problem, enable_pycombina=True),
    )
    preparation = controller.preparation_phase(
        problem.x0,
        problem.lbx,
        problem.ubx,
        problem.lbg,
        problem.ubg,
        predicted_state=problem.initial_state,
        p=problem.p,
    )
    stats = preparation.stats["rnlp"]
    if not stats.get("success", False):
        raise RuntimeError(
            f"RNLP failed: {stats.get('return_status', 'unknown status')}"
        )
    if preparation.pycombina_binary is None:
        binary, _ = controller.solver.solve_pycombina(
            preparation.rnlp_solution["x"], log_call=False
        )
        preparation.pycombina_binary = binary
    return preparation


def shifted_objective(model: QuadraticModel, solution: dict[str, ca.DM]) -> float:
    hessian = ca.DM(model.hessian)
    constant = model.objective_value - float(
        ca.dot(model.gradient, model.reference)
    )
    constant += 0.5 * float(
        ca.mtimes([model.reference.T, hessian, model.reference])
    )
    return float(solution["cost"]) + constant


def solve_variant(
    problem: ThreeTankProblem,
    preparation: PreparationResult,
    warm_start: ca.DM,
    *,
    method: str,
    horizon: int,
) -> ScalingResult:
    if method == "full":
        options = solver_options(problem)
        pycombina_binary = None
        name = "three_tank_frozen_full"
    elif method == "M_f":
        options = solver_options(problem, M_f=horizon)
        pycombina_binary = None
        name = f"three_tank_frozen_mf_{horizon}"
    elif method == "M_o":
        options = solver_options(
            problem,
            M_o=horizon,
            enable_pycombina=True,
        )
        pycombina_binary = preparation.pycombina_binary
        name = f"three_tank_frozen_mo_{horizon}"
    else:
        raise ValueError("method must be 'full', 'M_f', or 'M_o'.")

    solver = ThreePhaseSolver(name, problem.nlp, problem.discrete, options)
    solution, _, solve_time = solver.solve_mip(
        preparation.quadratic_model,
        preparation.lbx,
        preparation.ubx,
        preparation.lbg,
        preparation.ubg,
        pycombina_binary,
        warm_start,
        log_call=False,
    )
    stats = solver._mip_solver.stats()
    if not stats.get("success", False):
        raise RuntimeError(
            f"{method}={horizon} failed: "
            f"{stats.get('return_status', 'unknown status')}"
        )
    canonical = solver._canonical_mip(
        preparation.quadratic_model,
        preparation.lbx,
        preparation.ubx,
        preparation.lbg,
        preparation.ubg,
        pycombina_binary,
    )
    discrete = np.asarray(solver._mip_discrete_mask(), dtype=bool)
    lower = np.asarray(canonical["lbx"], dtype=float).reshape(-1)
    upper = np.asarray(canonical["ubx"], dtype=float).reshape(-1)
    active_binaries = int(
        np.count_nonzero(discrete & (upper - lower > 1e-12))
    )
    return ScalingResult(
        method=method,
        horizon=horizon,
        objective=shifted_objective(preparation.quadratic_model, solution),
        solve_time=solve_time,
        active_binaries=active_binaries,
    )


def validate_configuration() -> None:
    if N <= 1:
        raise ValueError("N must be greater than one.")
    if len(set(HORIZON_VALUES)) != len(HORIZON_VALUES):
        raise ValueError("HORIZON_VALUES must not contain duplicates.")
    if any(
        not isinstance(value, int) or not 1 <= value <= N
        for value in HORIZON_VALUES
    ):
        raise ValueError("Every horizon value must be an integer in [1, N].")
    if N not in HORIZON_VALUES:
        raise ValueError("HORIZON_VALUES must include N.")
    if OBJECTIVE_TOL <= 0.0:
        raise ValueError("OBJECTIVE_TOL must be positive.")
    if not ca.has_conic("gurobi"):
        raise RuntimeError("The CasADi Gurobi conic plugin is unavailable.")
    if importlib.util.find_spec("pycombina") is None:
        raise RuntimeError("PyCombina is required for the M_o variants.")


def verify_full_horizon(
    full: ScalingResult,
    M_f_results: list[ScalingResult],
    M_o_results: list[ScalingResult],
) -> None:
    for result in (M_f_results[-1], M_o_results[-1]):
        if not np.isclose(
            result.objective,
            full.objective,
            rtol=OBJECTIVE_TOL,
            atol=OBJECTIVE_TOL,
        ):
            raise RuntimeError(
                f"{result.method}=N does not reproduce the full MIQP objective: "
                f"{result.objective:.9f} versus {full.objective:.9f}."
            )


def report_behavior(
    full: ScalingResult,
    M_f_results: list[ScalingResult],
    M_o_results: list[ScalingResult],
) -> None:
    tolerance = OBJECTIVE_TOL * max(1.0, abs(full.objective))
    M_f_objectives = np.asarray([result.objective for result in M_f_results])
    M_o_objectives = np.asarray([result.objective for result in M_o_results])
    if np.any(M_f_objectives > full.objective + tolerance):
        print("Warning: an M_f relaxation objective exceeds the full MIQP objective.")
    if np.any(M_o_objectives < full.objective - tolerance):
        print("Warning: an M_o fixed-tail objective is below the full MIQP objective.")
    if np.any(np.diff(M_f_objectives) < -tolerance):
        print("Note: the observed M_f objectives are not monotone.")
    if np.any(np.diff(M_o_objectives) > tolerance):
        print("Note: the observed M_o objectives are not monotone.")


def print_table(full: ScalingResult, results: list[ScalingResult]) -> None:
    headings = (
        "Method",
        "Horizon",
        "Objective",
        "Gap to full",
        "Solve [s]",
        "Active binaries",
    )
    all_results = [full, *results]
    rows = [
        (
            result.method,
            str(result.horizon),
            f"{result.objective:.9f}",
            f"{result.objective - full.objective:+.3e}",
            f"{result.solve_time:.6f}",
            str(result.active_binaries),
        )
        for result in all_results
    ]
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

    print("\nFrozen-RNLP MIQP scaling")
    print(separator)
    print(line(headings))
    print(separator)
    for row in rows:
        print(line(row))
    print(separator)


def plot_results(
    full: ScalingResult,
    M_f_results: list[ScalingResult],
    M_o_results: list[ScalingResult],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(
        HORIZON_VALUES,
        [result.objective for result in M_f_results],
        color="#1f77b4",
        marker="o",
        label=r"$M_f$ (relaxed tail)",
    )
    axis.plot(
        HORIZON_VALUES,
        [result.objective for result in M_o_results],
        color="#d62728",
        marker="s",
        label=r"$M_o$ (PyCombina-fixed tail)",
    )
    axis.axhline(
        full.objective,
        color="black",
        linestyle="--",
        label="Full MIQP",
    )
    if LOG_SCALE:
        objectives = [
            full.objective,
            *(result.objective for result in M_f_results),
            *(result.objective for result in M_o_results),
        ]
        if min(objectives) <= 0.0:
            raise RuntimeError("Logarithmic objective scale requires positive values.")
        axis.set_yscale("log")
    axis.set_xlabel("Approximation horizon [control intervals]")
    axis.set_ylabel("Shifted quadratic objective")
    axis.set_xticks(HORIZON_VALUES)
    axis.grid(True, which="both", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_configuration()
    problem = build_three_tank_problem(N=N, dt=T / N)
    preparation = frozen_preparation(problem)
    warm_start = ca.DM(preparation.rnlp_solution["x"])
    warm_start[
        np.flatnonzero(np.asarray(problem.discrete, dtype=bool))
    ] = preparation.pycombina_binary

    full = solve_variant(
        problem,
        preparation,
        warm_start,
        method="full",
        horizon=N,
    )
    M_f_results = [
        solve_variant(
            problem,
            preparation,
            warm_start,
            method="M_f",
            horizon=horizon,
        )
        for horizon in HORIZON_VALUES
    ]
    M_o_results = [
        solve_variant(
            problem,
            preparation,
            warm_start,
            method="M_o",
            horizon=horizon,
        )
        for horizon in HORIZON_VALUES
    ]
    verify_full_horizon(full, M_f_results, M_o_results)
    report_behavior(full, M_f_results, M_o_results)
    print_table(full, [*M_f_results, *M_o_results])
    plot_results(full, M_f_results, M_o_results)


if __name__ == "__main__":
    main()
