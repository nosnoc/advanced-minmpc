#!/usr/bin/env python3
# Adapted from the mintOC Three-Tank multimode benchmark.
# https://mintoc.de/index.php?title=Three_Tank_multimode_problem

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import (
    AdvancedStepMPC,
    CaminoSolver,
    ThreePhaseOptions,
    ThreePhaseSolver,
    prepare_mpc_initial_guess,
    shift_mpc_trajectory,
)
from three_tank_multimode_ocp import (
    T,
    ThreeTankProblem,
    build_three_tank_problem,
    validate_one_hot,
)


N = 60
N_MPC = 100
REPETITIONS = 1
RUN_WARMUP = False

RUN_THREE_PHASE_MIQP = True
RUN_BONMIN = False
WARM_START_BONMIN = True
RUN_CAMINO = False
CAMINO_MIP_SOLVER = "gurobi"
CAMINO_MAX_TIME = 30.0
CAMINO_MAX_ITERATIONS = 50

RUN_AS_MIQP_GUROBI = True
RUN_AS_MIQP_GUROBI_WITH_PREP = False
RUN_AS_MIQP_DAQP = False
RUN_AS_MIQP_DAQP_WITH_PREP = False

RUN_AS_MILP_GUROBI = False
RUN_AS_MILP_HIGHS = False

RUN_MF_VARIANTS = True
MF_VALUES = (2, 4, 8, 10)

RUN_MO_VARIANTS = True
MO_VALUES = (0, 2, 4, 8, 10)

PYCOMBINA_WARM_START_MIQP = False

NORMALIZE_BY = "best"
PARETO_COST_LOG_SCALE = False
PARETO_TIME_LOG_SCALE = True


@dataclass(frozen=True, slots=True)
class Algorithm:
    label: str
    kind: str
    mip_solver: str | None
    solve_in_preparation: bool
    M_f: int | None
    M_o: int | None
    color: str
    marker: str


@dataclass(slots=True)
class BenchmarkRecord:
    states: list[np.ndarray]
    controls: list[np.ndarray]
    preparation_times: list[float]
    feedback_times: list[float]
    miqp_feedback_times: list[float]
    integer_found_in_rnlp: list[bool]
    closed_loop_cost: float = 0.0


class ExternalSolverError(RuntimeError):
    pass


def algorithms() -> list[Algorithm]:
    enabled: list[Algorithm] = []
    if RUN_THREE_PHASE_MIQP:
        enabled.append(
            Algorithm("ThreeP-MIQP", "three_phase", "gurobi", False, None, None, "#222222", "P")
        )
    if RUN_BONMIN:
        enabled.append(
            Algorithm("Bonmin", "bonmin", None, False, None, None, "#7f7f7f", "X")
        )
    if RUN_CAMINO:
        enabled.append(
            Algorithm(
                "CAMINO",
                "camino",
                CAMINO_MIP_SOLVER,
                False,
                None,
                None,
                "#17becf",
                "*",
            )
        )
    if RUN_AS_MIQP_GUROBI:
        enabled.append(
            Algorithm("AS-MIQP GRB", "miqp", "gurobi", False, None, None, "#1f77b4", "o")
        )
    if RUN_AS_MIQP_GUROBI_WITH_PREP:
        enabled.append(
            Algorithm("AS-MIQP GRB+prep", "miqp", "gurobi", True, None, None, "#1f77b4", "^")
        )
    if RUN_AS_MIQP_DAQP:
        enabled.append(
            Algorithm("AS-MIQP DAQP", "miqp", "daqp", False, None, None, "#ff7f0e", "o")
        )
    if RUN_AS_MIQP_DAQP_WITH_PREP:
        enabled.append(
            Algorithm("AS-MIQP DAQP+prep", "miqp", "daqp", True, None, None, "#ff7f0e", "^")
        )
    if RUN_AS_MILP_GUROBI:
        enabled.append(
            Algorithm("AS-MILP GRB", "milp", "gurobi", True, None, None, "#2ca02c", "s")
        )
    if RUN_AS_MILP_HIGHS:
        enabled.append(
            Algorithm("AS-MILP HiGHS", "milp", "highs", True, None, None, "#2ca02c", "D")
        )

    markers = ("o", "s", "^", "D", "v", "<", ">")
    if RUN_MF_VARIANTS:
        for index, value in enumerate(MF_VALUES):
            enabled.append(
                Algorithm(
                    f"AS-MIQP Mf={value}",
                    "miqp",
                    "gurobi",
                    False,
                    value,
                    None,
                    "#d62728",
                    markers[index % len(markers)],
                )
            )
    if RUN_MO_VARIANTS:
        for index, value in enumerate(MO_VALUES):
            enabled.append(
                Algorithm(
                    f"AS-MIQP Mo={value}",
                    "miqp",
                    "gurobi",
                    True,
                    None,
                    value,
                    "#9467bd",
                    markers[index % len(markers)],
                )
            )
    return enabled


def validate_configuration(enabled: list[Algorithm]) -> None:
    if not enabled:
        raise ValueError("No benchmark algorithm is enabled.")
    if N < 1 or N_MPC < 1:
        raise ValueError("N and N_MPC must be positive.")
    if REPETITIONS < 1:
        raise ValueError("REPETITIONS must be positive.")
    if RUN_CAMINO and CAMINO_MAX_TIME <= 0.0:
        raise ValueError("CAMINO_MAX_TIME must be positive.")
    if (
        RUN_CAMINO
        and CAMINO_MAX_ITERATIONS is not None
        and CAMINO_MAX_ITERATIONS < 1
    ):
        raise ValueError("CAMINO_MAX_ITERATIONS must be positive or None.")
    if NORMALIZE_BY not in {"best", "worst", "absolute"}:
        raise ValueError("NORMALIZE_BY must be 'best', 'worst', or 'absolute'.")
    variant_values = []
    if RUN_MF_VARIANTS:
        variant_values.append(("MF_VALUES", MF_VALUES, 1))
    if RUN_MO_VARIANTS:
        variant_values.append(("MO_VALUES", MO_VALUES, 0))
    for name, values, minimum in variant_values:
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicate values.")
        if any(not isinstance(value, int) or not minimum <= value <= N for value in values):
            raise ValueError(
                f"Every value in {name} must be an integer in [{minimum}, N]."
            )
    labels = [algorithm.label for algorithm in enabled]
    if len(set(labels)) != len(labels):
        raise ValueError("Enabled benchmark algorithms must have unique labels.")


def require_external_solver(algorithm: Algorithm) -> None:
    if algorithm.kind == "bonmin":
        if not ca.has_nlpsol("bonmin"):
            raise ExternalSolverError("the CasADi Bonmin plugin is unavailable")
        return
    if algorithm.kind == "camino":
        if importlib.util.find_spec("camino") is None:
            raise ExternalSolverError(
                "CAMINO is not installed; run "
                "'python -m pip install -r requirements-camino.txt'"
            )
        if not ca.has_nlpsol("ipopt"):
            raise ExternalSolverError("the CasADi IPOPT plugin is unavailable")
    if algorithm.mip_solver is not None and not ca.has_conic(algorithm.mip_solver):
        raise ExternalSolverError(
            f"the CasADi {algorithm.mip_solver} conic plugin is unavailable"
        )
    needs_pycombina = algorithm.M_o is not None or (
        PYCOMBINA_WARM_START_MIQP
        and algorithm.kind in {"miqp", "three_phase"}
    )
    if needs_pycombina and importlib.util.find_spec("pycombina") is None:
        raise ExternalSolverError("PyCombina is not installed")


def options(problem: ThreeTankProblem, algorithm: Algorithm) -> ThreePhaseOptions:
    miqp = algorithm.kind in {"miqp", "three_phase"}
    milp = algorithm.kind == "milp"
    return ThreePhaseOptions(
        mip_solver=algorithm.mip_solver,
        enable_miqp=miqp,
        enable_milp=milp,
        enable_pycombina=(
            algorithm.M_o is not None
            or (PYCOMBINA_WARM_START_MIQP and miqp)
        ),
        binary_controls_one_hot=True,
        n_x=3,
        n_u=0,
        n_v=3,
        N=problem.N,
        M_f=algorithm.M_f,
        M_o=algorithm.M_o,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=miqp and algorithm.solve_in_preparation,
        solve_milp_in_preparation=milp and algorithm.solve_in_preparation,
        solve_pnlp_in_feedback=False,
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
        pycombina_options={
            "time_grid": np.arange(problem.N + 1) * problem.dt
        },
    )


def solver_name(algorithm: Algorithm, run_tag: str = "") -> str:
    name = "three_tank_benchmark_" + re.sub(
        r"[^a-z0-9]+", "_", algorithm.label.lower()
    ).strip("_")
    return name if not run_tag else f"{name}_{run_tag}"


def empty_record(problem: ThreeTankProblem) -> BenchmarkRecord:
    return BenchmarkRecord(
        states=[np.asarray(problem.initial_state, dtype=float).reshape(-1)],
        controls=[],
        preparation_times=[],
        feedback_times=[],
        miqp_feedback_times=[],
        integer_found_in_rnlp=[],
    )


def first_control(problem: ThreeTankProblem, decision: Any) -> np.ndarray:
    values = np.asarray(decision, dtype=float).reshape(-1)
    control = values[np.asarray(problem.binary_indices[0], dtype=int)]
    validate_one_hot(control.reshape(1, -1))
    return np.rint(control)


def apply_control(
    problem: ThreeTankProblem,
    state: ca.DM,
    control: np.ndarray,
) -> tuple[ca.DM, float]:
    integration = problem.integrator(
        x0=state,
        w=ca.DM(control),
        reference=problem.reference,
    )
    return ca.DM(integration["xf"]), float(integration["qf"])


def append_step(
    problem: ThreeTankProblem,
    record: BenchmarkRecord,
    state: ca.DM,
    control: np.ndarray,
    preparation_time: float,
    feedback_time: float,
    miqp_feedback_time: float,
    integer_found_in_rnlp: bool,
) -> ca.DM:
    next_state, stage_cost = apply_control(problem, state, control)
    levels = np.asarray(next_state, dtype=float).reshape(-1)
    if np.any(levels < -1e-8):
        raise RuntimeError("The simulated tank levels became negative.")
    record.states.append(levels)
    record.controls.append(control)
    record.preparation_times.append(float(preparation_time))
    record.feedback_times.append(float(feedback_time))
    record.miqp_feedback_times.append(float(miqp_feedback_time))
    record.integer_found_in_rnlp.append(bool(integer_found_in_rnlp))
    record.closed_loop_cost += stage_cost
    return next_state


def fixed_initial_bounds(
    problem: ThreeTankProblem, state: ca.DM
) -> tuple[ca.DM, ca.DM]:
    lbx, ubx = ca.DM(problem.lbx), ca.DM(problem.ubx)
    indices = np.asarray(problem.state_indices[0], dtype=int)
    lbx[indices] = state
    ubx[indices] = state
    return lbx, ubx


def run_three_phase(
    problem: ThreeTankProblem,
    algorithm: Algorithm,
    run_tag: str = "",
    n_mpc: int = N_MPC,
) -> BenchmarkRecord:
    solver = ThreePhaseSolver(
        solver_name(algorithm, run_tag),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm),
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    record = empty_record(problem)
    for step in range(n_mpc):
        if step:
            guess = shift_mpc_trajectory(
                guess,
                problem.state_indices,
                problem.continuous_control_indices,
                problem.binary_indices,
                state,
            )
        lbx, ubx = fixed_initial_bounds(problem, state)
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=problem.p,
        )
        if not result.stats["pnlp"].get("success", False):
            raise RuntimeError(
                f"PNLP failed at MPC step {step}: "
                f"{result.stats['pnlp'].get('return_status', 'unknown status')}"
            )
        guess = ca.DM(result.solution["x"])
        control = first_control(problem, guess)
        state = append_step(
            problem,
            record,
            state,
            control,
            0.0,
            result.timings["total"],
            result.timings["miqp_solution"],
            bool(result.stats.get("integer_found_in_rnlp", False)),
        )
    return record


def run_bonmin(
    problem: ThreeTankProblem,
    algorithm: Algorithm,
    run_tag: str = "",
    n_mpc: int = N_MPC,
) -> BenchmarkRecord:
    solver = ca.nlpsol(
        solver_name(algorithm, run_tag),
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
            "print_time": False,
        },
    )
    state = ca.DM(problem.initial_state)
    previous_solution = None
    record = empty_record(problem)
    for step in range(n_mpc):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            problem.continuous_control_indices,
            problem.binary_indices,
            state,
            WARM_START_BONMIN,
        )
        lbx, ubx = fixed_initial_bounds(problem, state)
        start = perf_counter()
        solution = solver(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=problem.p,
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
        state = append_step(
            problem, record, state, control, 0.0, elapsed, 0.0, False
        )
    return record


def run_camino(
    problem: ThreeTankProblem,
    algorithm: Algorithm,
    run_tag: str = "",
    n_mpc: int = N_MPC,
) -> BenchmarkRecord:
    solver = CaminoSolver(
        solver_name(algorithm, run_tag),
        problem.nlp,
        problem.discrete,
        mip_solver=CAMINO_MIP_SOLVER,
        max_time=CAMINO_MAX_TIME,
        max_iterations=CAMINO_MAX_ITERATIONS,
        hessian_not_psd=True,
    )
    state = ca.DM(problem.initial_state)
    previous_solution = None
    record = empty_record(problem)
    for step in range(n_mpc):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            problem.continuous_control_indices,
            problem.binary_indices,
            state,
            True,
        )
        lbx, ubx = fixed_initial_bounds(problem, state)
        start = perf_counter()
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=problem.p,
        )
        elapsed = perf_counter() - start
        if not result.stats.get("success", False):
            raise RuntimeError(
                f"CAMINO failed at MPC step {step}: "
                f"{result.stats.get('return_status', 'unknown status')}"
            )
        previous_solution = ca.DM(result.solution["x"])
        control = first_control(problem, previous_solution)
        state = append_step(
            problem,
            record,
            state,
            control,
            0.0,
            elapsed,
            0.0,
            False,
        )
    return record


def run_advanced(
    problem: ThreeTankProblem,
    algorithm: Algorithm,
    run_tag: str = "",
    n_mpc: int = N_MPC,
) -> BenchmarkRecord:
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(
            x0=x,
            w=v,
            reference=problem.reference,
        )["xf"]

    controller = AdvancedStepMPC(
        solver_name(algorithm, run_tag),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    record = empty_record(problem)
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
                f"RNLP failed at MPC step {step}: "
                f"{preparation.stats['rnlp'].get('return_status', 'unknown status')}"
            )
        feedback = controller.feedback_phase(state, preparation, p=problem.p)
        method = "milp" if algorithm.kind == "milp" else "miqp"
        mip_stats = feedback.stats[method] or {}
        if not mip_stats.get("success", False):
            raise RuntimeError(
                f"{method.upper()} failed at MPC step {step}: "
                f"{mip_stats.get('return_status', 'unknown status')}"
            )
        guess = ca.DM(feedback.solution["x"])
        previous_solution = guess
        control = first_control(problem, guess)
        state = append_step(
            problem,
            record,
            state,
            control,
            preparation.timings["total_preparation"],
            feedback.timings["total_feedback"],
            feedback.timings["miqp_solution"],
            bool(feedback.stats.get("integer_found_in_rnlp", False)),
        )
    return record


def validate_record(
    record: BenchmarkRecord, label: str, n_mpc: int = N_MPC
) -> None:
    if np.asarray(record.states).shape != (n_mpc + 1, 3):
        raise RuntimeError(f"{label} has inconsistent state dimensions.")
    if np.asarray(record.controls).shape != (n_mpc, 3):
        raise RuntimeError(f"{label} has inconsistent control dimensions.")
    if len(record.preparation_times) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent preparation-time dimensions.")
    if len(record.feedback_times) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent feedback-time dimensions.")
    if len(record.miqp_feedback_times) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent MIQP-time dimensions.")
    if len(record.integer_found_in_rnlp) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent integrality-flag dimensions.")
    validate_one_hot(record.controls)
    if not np.isfinite(record.closed_loop_cost) or record.closed_loop_cost < 0.0:
        raise RuntimeError(f"{label} has an invalid realized closed-loop cost.")
    if np.any(np.asarray(record.preparation_times) < 0.0):
        raise RuntimeError(f"{label} has invalid preparation times.")
    if np.any(np.asarray(record.feedback_times) <= 0.0):
        raise RuntimeError(f"{label} has invalid feedback times.")


def classify_external_failure(error: RuntimeError) -> str | None:
    message = str(error).lower()
    if "licence" in message or "license" in message:
        return "licence"
    if any(
        term in message
        for term in (
            "not installed",
            "plugin is unavailable",
            "failed to load",
            "cannot load",
            "shared library",
            "dll",
        )
    ):
        return "plugin/dependency"
    return None


def execute(
    problem: ThreeTankProblem,
    algorithm: Algorithm,
    run_tag: str = "",
    announce: bool = True,
    n_mpc: int = N_MPC,
) -> BenchmarkRecord | None:
    if announce:
        print(f"Running {algorithm.label} ...")
    try:
        require_external_solver(algorithm)
        if algorithm.kind == "three_phase":
            record = run_three_phase(problem, algorithm, run_tag, n_mpc)
        elif algorithm.kind == "bonmin":
            record = run_bonmin(problem, algorithm, run_tag, n_mpc)
        elif algorithm.kind == "camino":
            record = run_camino(problem, algorithm, run_tag, n_mpc)
        else:
            record = run_advanced(problem, algorithm, run_tag, n_mpc)
    except ExternalSolverError as error:
        print(f"SKIPPED {algorithm.label} [plugin/dependency]: {error}")
        return None
    except RuntimeError as error:
        failure_type = classify_external_failure(error)
        if failure_type is None:
            raise RuntimeError(f"{algorithm.label} failed: {error}") from error
        print(f"SKIPPED {algorithm.label} [{failure_type}]: {error}")
        return None
    validate_record(record, algorithm.label, n_mpc)
    if announce:
        print(
            f"Finished {algorithm.label}: realized cost="
            f"{record.closed_loop_cost:.6f}"
        )
    return record


def normalized_costs(
    results: dict[Algorithm, list[BenchmarkRecord]],
) -> dict[Algorithm, float]:
    costs = {
        algorithm: float(np.median([record.closed_loop_cost for record in records]))
        for algorithm, records in results.items()
    }
    if NORMALIZE_BY == "absolute":
        return costs
    values = np.asarray(list(costs.values()), dtype=float)
    denominator = np.min(values) if NORMALIZE_BY == "best" else np.max(values)
    if denominator <= 0.0:
        raise RuntimeError("Closed-loop costs must be positive for normalization.")
    return {
        algorithm: cost / denominator for algorithm, cost in costs.items()
    }


def eligible_values(
    records: list[BenchmarkRecord], attribute: str
) -> np.ndarray:
    values = []
    for record in records:
        selected = np.asarray(getattr(record, attribute), dtype=float)
        integral = np.asarray(record.integer_found_in_rnlp, dtype=bool)
        values.extend(selected[~integral].tolist())
    return np.asarray(values, dtype=float)


def representative_record(records: list[BenchmarkRecord]) -> BenchmarkRecord:
    return min(
        records,
        key=lambda record: abs(
            np.mean(record.feedback_times)
            - np.median(
                [np.mean(candidate.feedback_times) for candidate in records]
            )
        ),
    )


def parameter_label(algorithm: Algorithm) -> str:
    if algorithm.M_f is not None:
        return f"Mf={algorithm.M_f}"
    if algorithm.M_o is not None:
        return f"Mo={algorithm.M_o}"
    return "-"


def print_summary(
    results: dict[Algorithm, list[BenchmarkRecord]],
    normalized: dict[Algorithm, float],
) -> None:
    headings = (
        "Algorithm",
        "Mf/Mo",
        "Max prep [s]",
        "Max feedback [s]",
        "Mean prep [s]",
        "Mean feedback [s]",
        "Normalized cost",
        "Absolute cost",
    )
    rows = []
    for algorithm, records in results.items():
        preparation = eligible_values(records, "preparation_times")
        feedback = eligible_values(records, "feedback_times")
        cost = float(np.median([record.closed_loop_cost for record in records]))
        rows.append(
            (
                algorithm.label,
                parameter_label(algorithm),
                f"{np.max(preparation):.4f}",
                f"{np.max(feedback):.4f}",
                f"{np.mean(preparation):.4f}",
                f"{np.mean(feedback):.4f}",
                "-" if NORMALIZE_BY == "absolute" else f"{normalized[algorithm]:.4f}",
                f"{cost:.6f}",
            )
        )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(width) if index < 2 else value.rjust(width)
            for index, (value, width) in enumerate(zip(row, widths))
        ) + " |"

    print("\nThree-Tank MPC benchmark summary")
    print(separator)
    print(format_row(headings))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def plot_performance(
    results: dict[Algorithm, list[BenchmarkRecord]],
    normalized: dict[Algorithm, float],
) -> None:
    if PARETO_COST_LOG_SCALE and any(value <= 0.0 for value in normalized.values()):
        raise RuntimeError("Pareto costs must be positive for a logarithmic y-axis.")
    ylabel = (
        "Absolute closed-loop cost"
        if NORMALIZE_BY == "absolute"
        else f"Normalized closed-loop cost ({NORMALIZE_BY})"
    )
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    ):
        selectors = (
            (np.mean, "Mean feedback time [s]", "Mean feedback-time performance"),
            (
                np.max,
                "Maximum feedback time [s]",
                "Worst-case feedback-time performance",
            ),
        )
        for reducer, xlabel, title in selectors:
            figure, axis = plt.subplots(figsize=(7, 6))
            for index, (algorithm, records) in enumerate(results.items()):
                feedback_time = float(
                    reducer(eligible_values(records, "feedback_times"))
                )
                axis.scatter(
                    feedback_time,
                    normalized[algorithm],
                    color=algorithm.color,
                    marker=algorithm.marker,
                    edgecolors="black",
                    linewidths=0.6,
                    s=75,
                    label=algorithm.label,
                    zorder=3,
                )
                axis.annotate(
                    algorithm.label,
                    (feedback_time, normalized[algorithm]),
                    xytext=(5, 6 + 5 * (index % 3)),
                    textcoords="offset points",
                    fontsize=8,
                )
            if PARETO_TIME_LOG_SCALE:
                axis.set_xscale("log")
            if PARETO_COST_LOG_SCALE:
                axis.set_yscale("log")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.35)
            axis.legend(fontsize=8)
            axis.set_title(title)
            figure.tight_layout()
        plt.show()


def plot_fastest_and_best(
    problem: ThreeTankProblem,
    results: dict[Algorithm, list[BenchmarkRecord]],
) -> None:
    fastest = min(
        results,
        key=lambda algorithm: np.mean(
            eligible_values(results[algorithm], "feedback_times")
        ),
    )
    best = min(
        results,
        key=lambda algorithm: np.median(
            [record.closed_loop_cost for record in results[algorithm]]
        ),
    )
    if fastest == best:
        selected = ((fastest, f"Fastest and best: {fastest.label}", "-"),)
    else:
        selected = (
            (fastest, f"Fastest: {fastest.label}", "-"),
            (best, f"Best cost: {best.label}", "--"),
        )

    state_time = np.arange(N_MPC + 1) * problem.dt
    control_time = np.arange(N_MPC) * problem.dt
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    ):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8))
        for algorithm, label, linestyle in selected:
            record = representative_record(results[algorithm])
            states = np.asarray(record.states, dtype=float)
            controls = np.asarray(record.controls, dtype=float)
            for index in range(3):
                axes[0, index].plot(
                    state_time,
                    states[:, index],
                    color=algorithm.color,
                    linestyle=linestyle,
                    label=label,
                )
                axes[1, index].step(
                    control_time,
                    controls[:, index],
                    where="post",
                    color=algorithm.color,
                    linestyle=linestyle,
                    label=label,
                )
        axes[0, 1].axhline(
            float(problem.reference[0]),
            color="black",
            linestyle=":",
            label="Reference",
        )
        axes[0, 2].axhline(
            float(problem.reference[1]),
            color="black",
            linestyle=":",
            label="Reference",
        )
        for index in range(3):
            axes[0, index].set_title(rf"Tank level $x_{index + 1}$")
            axes[0, index].set_xlabel("Time [s]")
            axes[0, index].set_ylabel(rf"$x_{index + 1}$")
            axes[1, index].set_title(rf"Applied mode $w_{index + 1}$")
            axes[1, index].set_xlabel("Time [s]")
            axes[1, index].set_ylabel(rf"$w_{index + 1}$")
            axes[1, index].set_yticks((0, 1))
            for axis in axes[:, index]:
                axis.grid(True, alpha=0.35)
                axis.legend(fontsize=8)
        figure.suptitle("Closed-loop behavior of the fastest and best algorithms")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    enabled = algorithms()
    validate_configuration(enabled)
    problem = build_three_tank_problem(N=N, dt=T / N)
    results: dict[Algorithm, list[BenchmarkRecord]] = {}
    for algorithm in enabled:
        if RUN_WARMUP and execute(
            problem, algorithm, "warmup", announce=False
        ) is None:
            continue
        records = []
        for repetition in range(REPETITIONS):
            print(
                f"Running {algorithm.label}, repetition "
                f"{repetition + 1}/{REPETITIONS} ..."
            )
            record = execute(
                problem,
                algorithm,
                f"repeat_{repetition}",
                announce=False,
            )
            if record is None:
                records = []
                break
            records.append(record)
        if records:
            results[algorithm] = records
    if not results:
        raise RuntimeError("No enabled benchmark algorithm completed successfully.")
    normalized = normalized_costs(results)
    print_summary(results, normalized)
    plot_performance(results, normalized)
    plot_fastest_and_best(problem, results)


if __name__ == "__main__":
    main()
