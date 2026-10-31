#!/usr/bin/env python3
# Benchmark for the CAMINO double-tank MPC example.
# Model source: CAMINO, create_double_tank_problem (doubletank).
# https://github.com/minlp-toolbox/CAMINO/blob/22ab5c75949fcb1bc99d1eb886fc2c456872c583/camino/problems/problem_collection.py
# Original model: R. Abbasi-Esfeden, W. Van Roy, and J. Swevers,
# Iterative Switching Time Optimization for Mixed-integer Optimal Control
# Problems, European Control Conference, 2023.
# https://doi.org/10.23919/ECC57647.2023.10178419

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import importlib.util
import logging
import multiprocessing
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
from double_tank_ocp import N, DoubleTankProblem, build_double_tank_problem, demand_trajectory


N_MPC = 80
WARMUP_RUNS = 1
REPETITIONS = 1
FRESH_PROCESS_MEASUREMENTS = True
GUROBI_SEED = 0
GUROBI_SINGLE_THREAD = False

RUN_THREE_PHASE_MIQP = True
RUN_BONMIN = False
WARM_START_BONMIN = True
RUN_CAMINO = True
CAMINO_MIP_SOLVER = "gurobi"
CAMINO_MAX_TIME = 5.0
CAMINO_MAX_ITERATIONS = 50

RUN_AS_MIQP_GUROBI = True
RUN_AS_MIQP_GUROBI_WITH_PREP = False
RUN_AS_MIQP_DAQP = False
RUN_AS_MIQP_DAQP_WITH_PREP = False

RUN_AS_MILP_GUROBI = False
RUN_AS_MILP_HIGHS = False

RUN_MF_VARIANTS = True
MF_VALUES = (1, 4, 8)

RUN_MO_VARIANTS = True
MO_VALUES = (1, 4, 8)

PYCOMBINA_WARM_START_MIQP = True

NORMALIZE_BY = "best"
PARETO_COST_LOG_SCALE = False
PARETO_TIME_LOG_SCALE = False


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


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    n_mpc: int
    gurobi_seed: int | None
    gurobi_single_thread: bool
    run_tag: str = ""


@dataclass(slots=True)
class BenchmarkRecord:
    states: list[np.ndarray]
    binary_controls: list[float]
    continuous_controls: list[float]
    preparation_times: list[float]
    feedback_times: list[float]
    component_timings: dict[str, list[float]]
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


def validate_configuration(
    problem: DoubleTankProblem,
    enabled: list[Algorithm],
    configuration: RunConfiguration,
) -> None:
    if not enabled:
        raise ValueError("No benchmark algorithm is enabled.")
    if configuration.n_mpc < 1:
        raise ValueError("N_MPC must be positive.")
    if WARMUP_RUNS < 0:
        raise ValueError("WARMUP_RUNS must be nonnegative.")
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
    if configuration.gurobi_seed is not None and configuration.gurobi_seed < 0:
        raise ValueError("GUROBI_SEED must be nonnegative or None.")
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
        if any(
            not isinstance(value, int) or not minimum <= value <= problem.N
            for value in values
        ):
            raise ValueError(
                f"Every value in {name} must be an integer in "
                f"[{minimum}, {problem.N}]."
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


def options(
    problem: DoubleTankProblem,
    algorithm: Algorithm,
    configuration: RunConfiguration,
) -> ThreePhaseOptions:
    miqp = algorithm.kind in {"miqp", "three_phase"}
    milp = algorithm.kind == "milp"
    gurobi = {
        "OutputFlag": 0,
        "LogToConsole": 0,
        "IntFeasTol": 1e-9,
    }
    if configuration.gurobi_seed is not None:
        gurobi["Seed"] = configuration.gurobi_seed
    if configuration.gurobi_single_thread:
        gurobi["Threads"] = 1
    return ThreePhaseOptions(
        mip_solver=algorithm.mip_solver,
        enable_miqp=miqp,
        enable_milp=milp,
        enable_pycombina=(
            algorithm.M_o is not None
            or (PYCOMBINA_WARM_START_MIQP and miqp)
        ),
        binary_controls_one_hot=False,
        n_x=2,
        n_u=1,
        n_v=1,
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
        gurobi_options={"gurobi": gurobi},
        highs_options={"highs": {"output_flag": False}},
        pycombina_options={"time_grid": np.arange(problem.N + 1) * problem.dt},
    )


def solver_name(algorithm: Algorithm, configuration: RunConfiguration) -> str:
    base = "two_tank_benchmark_" + re.sub(
        r"[^a-z0-9]+", "_", algorithm.label.lower()
    ).strip("_")
    if not configuration.run_tag:
        return base
    tag = re.sub(r"[^a-z0-9]+", "_", configuration.run_tag.lower()).strip("_")
    return f"{base}_{tag}"


def empty_record(problem: DoubleTankProblem) -> BenchmarkRecord:
    return BenchmarkRecord(
        states=[np.asarray(problem.initial_state, dtype=float).reshape(-1)],
        binary_controls=[],
        continuous_controls=[],
        preparation_times=[],
        feedback_times=[],
        component_timings={},
        integer_found_in_rnlp=[],
    )


def first_control(problem: DoubleTankProblem, decision: Any) -> tuple[float, float]:
    values = np.asarray(decision, dtype=float).reshape(-1)
    binary = float(values[int(np.asarray(problem.binary_indices[0]).reshape(-1)[0])])
    continuous = float(
        values[int(np.asarray(problem.continuous_control_indices[0]).reshape(-1)[0])]
    )
    if not np.isclose(binary, np.rint(binary), atol=1e-6):
        raise RuntimeError(f"Applied binary control is not integral: {binary}.")
    return float(np.rint(binary)), continuous


def apply_control(
    problem: DoubleTankProblem,
    state: ca.DM,
    binary: float,
    continuous: float,
) -> ca.DM:
    return ca.DM(
        problem.integrator(x0=state, u=ca.DM([binary, continuous]))["xf"]
    )


def realized_stage_cost(
    problem: DoubleTankProblem,
    state: ca.DM,
    binary: float,
    continuous: float,
    demand: float,
) -> float:
    x = np.asarray(state, dtype=float).reshape(-1)

    def rhs(levels: np.ndarray) -> np.ndarray:
        first_outflow = np.sqrt(levels[0] + 1e-3)
        second_outflow = np.sqrt(levels[1] + 1e-3)
        return np.asarray(
            [
                10.0 * binary + continuous - first_outflow,
                first_outflow - second_outflow,
            ]
        )

    def running_cost(levels: np.ndarray) -> float:
        return float(
            100.0 * (levels[1] - demand) ** 2
            + 10.0 * binary
            + 1.2 * continuous
        )

    k1 = rhs(x)
    k2_state = x + problem.dt * k1 / 2.0
    k2 = rhs(k2_state)
    k3_state = x + problem.dt * k2 / 2.0
    k3 = rhs(k3_state)
    k4_state = x + problem.dt * k3
    return problem.dt * (
        running_cost(x)
        + 2.0 * running_cost(k2_state)
        + 2.0 * running_cost(k3_state)
        + running_cost(k4_state)
    ) / 6.0


def append_component_timings(
    record: BenchmarkRecord,
    timings: dict[str, float],
) -> None:
    step = len(record.preparation_times)
    for name in set(record.component_timings) | set(timings):
        values = record.component_timings.setdefault(name, [0.0] * step)
        values.append(float(timings.get(name, 0.0)))


def append_step(
    problem: DoubleTankProblem,
    record: BenchmarkRecord,
    state: ca.DM,
    binary: float,
    continuous: float,
    preparation_time: float,
    feedback_time: float,
    stage_cost: float,
    component_timings: dict[str, float],
    integer_found_in_rnlp: bool,
) -> ca.DM:
    next_state = apply_control(problem, state, binary, continuous)
    levels = np.asarray(next_state, dtype=float).reshape(-1)
    if np.any(levels < -1e-8):
        raise RuntimeError("The simulated tank levels became negative.")
    record.states.append(levels)
    record.binary_controls.append(binary)
    record.continuous_controls.append(continuous)
    append_component_timings(record, component_timings)
    record.preparation_times.append(float(preparation_time))
    record.feedback_times.append(float(feedback_time))
    record.integer_found_in_rnlp.append(bool(integer_found_in_rnlp))
    record.closed_loop_cost += float(stage_cost)
    return next_state


def fixed_initial_bounds(
    problem: DoubleTankProblem, state: ca.DM
) -> tuple[ca.DM, ca.DM]:
    lbx, ubx = ca.DM(problem.lbx), ca.DM(problem.ubx)
    indices = np.asarray(problem.state_indices[0], dtype=int)
    lbx[indices] = state
    ubx[indices] = state
    return lbx, ubx


def parameter(problem: DoubleTankProblem, demand: np.ndarray, step: int) -> ca.DM:
    return ca.DM(demand[step : step + problem.N])


def run_three_phase(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    algorithm: Algorithm,
    configuration: RunConfiguration,
) -> BenchmarkRecord:
    solver = ThreePhaseSolver(
        solver_name(algorithm, configuration),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm, configuration),
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    record = empty_record(problem)

    for step in range(configuration.n_mpc):
        if step:
            guess = shift_mpc_trajectory(
                guess,
                problem.state_indices,
                problem.continuous_control_indices,
                problem.binary_indices,
                state,
            )
        p = parameter(problem, demand, step)
        lbx, ubx = fixed_initial_bounds(problem, state)
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=p,
        )
        if not result.stats["pnlp"].get("success", False):
            raise RuntimeError(
                f"PNLP failed at MPC step {step}: "
                f"{result.stats['pnlp'].get('return_status', 'unknown status')}"
            )
        guess = ca.DM(result.solution["x"])
        binary, continuous = first_control(problem, guess)
        stage_cost = realized_stage_cost(
            problem, state, binary, continuous, float(demand[step])
        )
        state = append_step(
            problem,
            record,
            state,
            binary,
            continuous,
            0.0,
            result.timings["total"],
            stage_cost,
            {f"feedback.{name}": value for name, value in result.timings.items()},
            bool(result.stats.get("integer_found_in_rnlp", False)),
        )
    return record


def run_bonmin(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    algorithm: Algorithm,
    configuration: RunConfiguration,
) -> BenchmarkRecord:
    solver = ca.nlpsol(
        solver_name(algorithm, configuration),
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

    for step in range(configuration.n_mpc):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            problem.continuous_control_indices,
            problem.binary_indices,
            state,
            WARM_START_BONMIN,
        )
        p = parameter(problem, demand, step)
        lbx, ubx = fixed_initial_bounds(problem, state)
        start = perf_counter()
        solution = solver(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=p,
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
        binary, continuous = first_control(problem, guess)
        stage_cost = realized_stage_cost(
            problem, state, binary, continuous, float(demand[step])
        )
        state = append_step(
            problem,
            record,
            state,
            binary,
            continuous,
            0.0,
            elapsed,
            stage_cost,
            {"feedback.bonmin_solution": elapsed},
            False,
        )
    return record


def run_camino(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    algorithm: Algorithm,
    configuration: RunConfiguration,
) -> BenchmarkRecord:
    solver = CaminoSolver(
        solver_name(algorithm, configuration),
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

    for step in range(configuration.n_mpc):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            problem.continuous_control_indices,
            problem.binary_indices,
            state,
            True,
        )
        p = parameter(problem, demand, step)
        lbx, ubx = fixed_initial_bounds(problem, state)
        start = perf_counter()
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=p,
        )
        elapsed = perf_counter() - start
        if not result.stats.get("success", False):
            raise RuntimeError(
                f"CAMINO failed at MPC step {step}: "
                f"{result.stats.get('return_status', 'unknown status')}"
            )
        previous_solution = ca.DM(result.solution["x"])
        binary, continuous = first_control(problem, previous_solution)
        stage_cost = realized_stage_cost(
            problem, state, binary, continuous, float(demand[step])
        )
        state = append_step(
            problem,
            record,
            state,
            binary,
            continuous,
            0.0,
            elapsed,
            stage_cost,
            {
                "feedback.camino_total": elapsed,
                "feedback.camino_solver": result.timings["solver"],
                "feedback.camino_python": result.timings["python"],
            },
            False,
        )
    return record


def run_advanced(
    problem: DoubleTankProblem,
    demand: np.ndarray,
    algorithm: Algorithm,
    configuration: RunConfiguration,
) -> BenchmarkRecord:
    def integrate(x: ca.DM, u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(x0=x, u=ca.vertcat(v[0], u[0]))["xf"]

    controller = AdvancedStepMPC(
        solver_name(algorithm, configuration),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm, configuration),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    record = empty_record(problem)

    for step in range(configuration.n_mpc):
        p = parameter(problem, demand, step)
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
            p=p,
        )
        if not preparation.stats["rnlp"].get("success", False):
            raise RuntimeError(
                f"RNLP failed at MPC step {step}: "
                f"{preparation.stats['rnlp'].get('return_status', 'unknown status')}"
            )

        feedback = controller.feedback_phase(state, preparation, p=p)
        method = "milp" if algorithm.kind == "milp" else "miqp"
        mip_stats = feedback.stats[method] or {}
        if not mip_stats.get("success", False):
            raise RuntimeError(
                f"{method.upper()} failed at MPC step {step}: "
                f"{mip_stats.get('return_status', 'unknown status')}"
            )

        guess = ca.DM(feedback.solution["x"])
        previous_solution = guess
        binary, continuous = first_control(problem, guess)
        integer_found = bool(feedback.stats.get("integer_found_in_rnlp", False))
        stage_cost = realized_stage_cost(
            problem, state, binary, continuous, float(demand[step])
        )
        component_timings = {
            **{
                f"preparation.{name}": value
                for name, value in preparation.timings.items()
            },
            **{
                f"feedback.{name}": value
                for name, value in feedback.timings.items()
            },
        }
        state = append_step(
            problem,
            record,
            state,
            binary,
            continuous,
            preparation.timings["total_preparation"],
            feedback.timings["total_feedback"],
            stage_cost,
            component_timings,
            integer_found,
        )
    return record


def validate_record(
    record: BenchmarkRecord,
    label: str,
    configuration: RunConfiguration,
) -> None:
    n_mpc = configuration.n_mpc
    if np.asarray(record.states).shape != (n_mpc + 1, 2):
        raise RuntimeError(f"{label} has inconsistent state dimensions.")
    if np.asarray(record.binary_controls).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent binary-control dimensions.")
    if np.asarray(record.continuous_controls).shape != (n_mpc,):
        raise RuntimeError(f"{label} has inconsistent continuous-control dimensions.")
    if len(record.preparation_times) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent preparation-time dimensions.")
    if len(record.feedback_times) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent feedback-time dimensions.")
    if len(record.integer_found_in_rnlp) != n_mpc:
        raise RuntimeError(f"{label} has inconsistent RNLP-integrality dimensions.")
    if any(len(values) != n_mpc for values in record.component_timings.values()):
        raise RuntimeError(f"{label} has inconsistent component-timing dimensions.")
    if not np.allclose(record.binary_controls, np.rint(record.binary_controls), atol=1e-6):
        raise RuntimeError(f"{label} has non-integral applied binary controls.")
    if not np.isfinite(record.closed_loop_cost):
        raise RuntimeError(f"{label} has an invalid accumulated MPC objective.")
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
    problem: DoubleTankProblem,
    demand: np.ndarray,
    algorithm: Algorithm,
    configuration: RunConfiguration,
    announce: bool = True,
) -> BenchmarkRecord | None:
    if announce:
        print(f"Running {algorithm.label} ...")
    try:
        require_external_solver(algorithm)
        if algorithm.kind == "three_phase":
            record = run_three_phase(problem, demand, algorithm, configuration)
        elif algorithm.kind == "bonmin":
            record = run_bonmin(problem, demand, algorithm, configuration)
        elif algorithm.kind == "camino":
            record = run_camino(problem, demand, algorithm, configuration)
        else:
            record = run_advanced(problem, demand, algorithm, configuration)
    except ExternalSolverError as error:
        print(f"SKIPPED {algorithm.label} [plugin/dependency]: {error}")
        return None
    except RuntimeError as error:
        failure_type = classify_external_failure(error)
        if failure_type is None:
            raise RuntimeError(f"{algorithm.label} failed: {error}") from error
        print(f"SKIPPED {algorithm.label} [{failure_type}]: {error}")
        return None

    validate_record(record, algorithm.label, configuration)
    if announce:
        print(
            f"Finished {algorithm.label}: realized closed-loop cost="
            f"{record.closed_loop_cost:.6f}"
        )
    return record


def measurement_run(
    algorithm: Algorithm,
    configuration: RunConfiguration,
    warmup_runs: int,
) -> BenchmarkRecord | None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_double_tank_problem()
    demand = demand_trajectory(problem.N + configuration.n_mpc, problem.dt)
    for warmup in range(warmup_runs):
        warmup_configuration = replace(
            configuration,
            run_tag=f"{configuration.run_tag}_warmup_{warmup}",
        )
        if execute(
            problem,
            demand,
            algorithm,
            warmup_configuration,
            announce=False,
        ) is None:
            return None
    return execute(problem, demand, algorithm, configuration, announce=False)


def repeated_measurements(
    enabled: list[Algorithm],
    configuration: RunConfiguration,
    repetitions: int,
    warmup_runs: int,
    fresh_processes: bool,
) -> dict[Algorithm, list[BenchmarkRecord]]:
    results: dict[Algorithm, list[BenchmarkRecord]] = {}
    for algorithm in enabled:
        records: list[BenchmarkRecord] = []
        for repetition in range(repetitions):
            print(
                f"Running {algorithm.label}, repetition "
                f"{repetition + 1}/{repetitions} ..."
            )
            run_configuration = replace(
                configuration,
                run_tag=f"repeat_{repetition}",
            )
            if fresh_processes:
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    record = executor.submit(
                        measurement_run,
                        algorithm,
                        run_configuration,
                        warmup_runs,
                    ).result()
            else:
                record = measurement_run(
                    algorithm,
                    run_configuration,
                    warmup_runs,
                )
            if record is None:
                records = []
                break
            records.append(record)
            print(
                f"Finished {algorithm.label}, repetition {repetition + 1}: "
                f"realized closed-loop cost={record.closed_loop_cost:.6f}"
            )
        if records:
            results[algorithm] = records
    return results


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
    if np.isclose(denominator, 0.0):
        raise RuntimeError("Cannot normalize costs by a zero denominator.")
    return {
        algorithm: cost / denominator
        for algorithm, cost in costs.items()
    }


def parameter_label(algorithm: Algorithm) -> str:
    if algorithm.M_f is not None:
        return f"Mf={algorithm.M_f}"
    if algorithm.M_o is not None:
        return f"Mo={algorithm.M_o}"
    return "-"


def eligible_values(
    records: list[BenchmarkRecord],
    attribute: str | None = None,
    component: str | None = None,
) -> np.ndarray:
    selected: list[float] = []
    for record in records:
        mask = ~np.asarray(record.integer_found_in_rnlp, dtype=bool)
        if component is not None:
            values = np.asarray(record.component_timings[component], dtype=float)
        elif attribute is not None:
            values = np.asarray(getattr(record, attribute), dtype=float)
        else:
            raise ValueError("An attribute or component name is required.")
        selected.extend(values[mask].tolist())
    return np.asarray(selected, dtype=float)


def timing_statistics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: float("nan") for name in ("mean", "max")}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def format_timing(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.4f}"


def representative_record(records: list[BenchmarkRecord]) -> BenchmarkRecord:
    per_run_means = []
    for record in records:
        values = eligible_values([record], attribute="feedback_times")
        per_run_means.append(float(np.mean(values)) if values.size else float("inf"))
    finite = np.asarray(per_run_means, dtype=float)
    if not np.any(np.isfinite(finite)):
        return records[0]
    target = float(np.median(finite[np.isfinite(finite)]))
    return records[int(np.argmin(np.abs(finite - target)))]


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
        "Realized MPC cost",
    )
    rows = []
    for algorithm, records in results.items():
        preparation = timing_statistics(
            eligible_values(records, attribute="preparation_times")
        )
        feedback = timing_statistics(
            eligible_values(records, attribute="feedback_times")
        )
        cost = float(np.median([record.closed_loop_cost for record in records]))
        rows.append(
            (
                algorithm.label,
                parameter_label(algorithm),
                format_timing(preparation["max"]),
                format_timing(feedback["max"]),
                format_timing(preparation["mean"]),
                format_timing(feedback["mean"]),
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

    print("\nDouble-Tank MPC benchmark summary")
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
        "Realized closed-loop cost"
        if NORMALIZE_BY == "absolute"
        else f"Normalized realized closed-loop cost ({NORMALIZE_BY})"
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
                eligible_feedback = eligible_values(
                    records, attribute="feedback_times"
                )
                if eligible_feedback.size == 0:
                    continue
                feedback_time = float(reducer(eligible_feedback))
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
    problem: DoubleTankProblem,
    demand: np.ndarray,
    results: dict[Algorithm, list[BenchmarkRecord]],
    configuration: RunConfiguration,
) -> None:
    fastest = min(
        results,
        key=lambda algorithm: np.mean(
            eligible_values(results[algorithm], attribute="feedback_times")
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

    state_time = np.arange(configuration.n_mpc + 1) * problem.dt
    control_time = np.arange(configuration.n_mpc) * problem.dt
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for algorithm, label, linestyle in selected:
            record = representative_record(results[algorithm])
            states = np.asarray(record.states, dtype=float)
            axes[0, 0].plot(
                state_time,
                states[:, 0],
                color=algorithm.color,
                linestyle=linestyle,
                label=label,
            )
            axes[0, 1].plot(
                state_time,
                states[:, 1],
                color=algorithm.color,
                linestyle=linestyle,
                label=label,
            )
            axes[1, 0].step(
                control_time,
                record.binary_controls,
                where="post",
                color=algorithm.color,
                linestyle=linestyle,
                label=label,
            )
            axes[1, 1].step(
                control_time,
                record.continuous_controls,
                where="post",
                color=algorithm.color,
                linestyle=linestyle,
                label=label,
            )

        axes[0, 1].plot(
            state_time,
            demand[: configuration.n_mpc + 1],
            color="black",
            linestyle=":",
            label="Demand",
        )
        axes[0, 0].set_title(r"Tank level $x_1$")
        axes[0, 1].set_title(r"Tank level $x_2$")
        axes[1, 0].set_title(r"Binary pump control $v$")
        axes[1, 1].set_title(r"Continuous inflow $u$")
        axes[0, 0].set_ylabel(r"$x_1$")
        axes[0, 1].set_ylabel(r"$x_2$")
        axes[1, 0].set_ylabel(r"$v$")
        axes[1, 1].set_ylabel(r"$u$")
        axes[1, 0].set_yticks((0, 1))

        for axis in axes.flat:
            axis.set_xlabel("Time [s]")
            axis.grid(True, alpha=0.35)
            axis.legend(fontsize=8)

        figure.suptitle("Closed-loop behavior of the fastest and best algorithms")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_double_tank_problem()
    enabled = algorithms()
    configuration = RunConfiguration(
        n_mpc=N_MPC,
        gurobi_seed=GUROBI_SEED,
        gurobi_single_thread=GUROBI_SINGLE_THREAD,
    )
    validate_configuration(problem, enabled, configuration)
    demand = demand_trajectory(problem.N + configuration.n_mpc, problem.dt)
    results = repeated_measurements(
        enabled,
        configuration,
        REPETITIONS,
        WARMUP_RUNS,
        FRESH_PROCESS_MEASUREMENTS,
    )

    if not results:
        raise RuntimeError("No enabled benchmark algorithm completed successfully.")

    normalized = normalized_costs(results)
    print_summary(results, normalized)
    plot_performance(results, normalized)
    plot_fastest_and_best(problem, demand, results, configuration)


if __name__ == "__main__":
    main()
