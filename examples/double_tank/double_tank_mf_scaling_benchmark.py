#!/usr/bin/env python3
# Scaling benchmark for the CAMINO double-tank MPC example.
# Model source: CAMINO, create_double_tank_problem (doubletank).
# https://github.com/minlp-toolbox/CAMINO/blob/22ab5c75949fcb1bc99d1eb886fc2c456872c583/camino/problems/problem_collection.py
# Original model: R. Abbasi-Esfeden, W. Van Roy, and J. Swevers,
# Iterative Switching Time Optimization for Mixed-integer Optimal Control
# Problems, European Control Conference, 2023.
# https://doi.org/10.23919/ECC57647.2023.10178419

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import logging
import multiprocessing

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from double_tank_ocp import N, build_double_tank_problem, demand_trajectory
from two_tank_benchmark import (
    Algorithm,
    RunConfiguration,
    apply_control,
    first_control,
    options,
    parameter,
    realized_stage_cost,
    require_external_solver,
    run_advanced,
    solver_name,
)
from mpc_implementation import AdvancedStepMPC


N_MPC = 80
MF_VALUES = (1, 2, 4, 8, 10, 15, 20, 25, 30, 35, 40, N)
WARMUP_RUNS = 1
REPETITIONS = 1
FRESH_PROCESS_MEASUREMENTS = True
GUROBI_SEED = 0
GUROBI_SINGLE_THREAD = False
FIX_RNLP = True


@dataclass(slots=True)
class ScalingRecord:
    feedback_miqp_times: dict[str, list[float]]
    integer_found_in_rnlp: dict[str, list[bool]]
    closed_loop_costs: dict[str, float]


def variant_key(M_f: int, solve_in_preparation: bool) -> str:
    return f"{M_f}:{int(solve_in_preparation)}"


def variant_algorithm(M_f: int, solve_in_preparation: bool) -> Algorithm:
    suffix = "+prep" if solve_in_preparation else "no-prep"
    return Algorithm(
        f"Mf={M_f} {suffix}",
        "miqp",
        "gurobi",
        solve_in_preparation,
        M_f,
        None,
        "#1f77b4" if solve_in_preparation else "#d62728",
        "^" if solve_in_preparation else "o",
    )


def validate_configuration() -> None:
    if N_MPC < 1:
        raise ValueError("N_MPC must be positive.")
    if WARMUP_RUNS < 0:
        raise ValueError("WARMUP_RUNS must be nonnegative.")
    if REPETITIONS < 1:
        raise ValueError("REPETITIONS must be positive.")
    if len(set(MF_VALUES)) != len(MF_VALUES):
        raise ValueError("MF_VALUES must not contain duplicates.")
    if any(not isinstance(value, int) or not 1 <= value <= N for value in MF_VALUES):
        raise ValueError(f"Every M_f must be an integer in [1, {N}].")
    if N not in MF_VALUES:
        raise ValueError(f"MF_VALUES must include the full horizon N={N}.")
    if GUROBI_SEED is not None and GUROBI_SEED < 0:
        raise ValueError("GUROBI_SEED must be nonnegative or None.")


def build_controller(problem, algorithm, configuration):
    def integrate(x: ca.DM, u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(x0=x, u=ca.vertcat(v[0], u[0]))["xf"]

    return AdvancedStepMPC(
        solver_name(algorithm, configuration),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm, configuration),
        integrator=integrate,
    )


def run_scaling_once(configuration: RunConfiguration) -> ScalingRecord:
    problem = build_double_tank_problem()
    demand = demand_trajectory(problem.N + configuration.n_mpc, problem.dt)
    variants = {
        (M_f, solve_in_preparation): variant_algorithm(M_f, solve_in_preparation)
        for M_f in MF_VALUES
        for solve_in_preparation in (False, True)
    }
    for algorithm in variants.values():
        require_external_solver(algorithm)
    controllers = {
        key: build_controller(
            problem,
            algorithm,
            replace(configuration, run_tag=f"{configuration.run_tag}_{key[0]}_{int(key[1])}"),
        )
        for key, algorithm in variants.items()
    }
    reference_key = (N, False)
    reference_controller = controllers[reference_key]
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    feedback_times = {
        variant_key(*key): [] for key in variants
    }
    integral_flags = {
        variant_key(*key): [] for key in variants
    }
    closed_loop_cost = 0.0

    for step in range(configuration.n_mpc):
        p = parameter(problem, demand, step)
        shared_preparation = reference_controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
            p=p,
        )
        if not shared_preparation.stats["rnlp"].get("success", False):
            raise RuntimeError(
                f"Shared RNLP failed at step {step}: "
                f"{shared_preparation.stats['rnlp'].get('return_status', 'unknown')}"
            )
        integer_found = bool(
            shared_preparation.stats.get("integer_found_in_rnlp", False)
        )
        feedback_results = {}
        for key, controller in controllers.items():
            solve_in_preparation = key[1]
            prepared = replace(
                shared_preparation,
                predicted_miqp_solution=None,
                predicted_milp_solution=None,
            )
            if solve_in_preparation and not integer_found:
                predicted_miqp, _, _ = controller.solver.solve_mip(
                    shared_preparation.quadratic_model,
                    shared_preparation.lbx,
                    shared_preparation.ubx,
                    shared_preparation.lbg,
                    shared_preparation.ubg,
                    None,
                    None,
                    log_call=False,
                )
                prepared = replace(
                    prepared,
                    predicted_miqp_solution=predicted_miqp,
                )
            feedback = controller.feedback_phase(state, prepared, p=p)
            if not feedback.stats["miqp"].get("success", False):
                raise RuntimeError(
                    f"MIQP failed for M_f={key[0]}, prep={key[1]}, step={step}: "
                    f"{feedback.stats['miqp'].get('return_status', 'unknown')}"
                )
            name = variant_key(*key)
            feedback_times[name].append(float(feedback.timings["miqp_solution"]))
            integral_flags[name].append(integer_found)
            feedback_results[key] = feedback

        reference_solution = ca.DM(feedback_results[reference_key].solution["x"])
        binary, continuous = first_control(problem, reference_solution)
        closed_loop_cost += realized_stage_cost(
            problem,
            state,
            binary,
            continuous,
            float(demand[step]),
        )
        state = apply_control(problem, state, binary, continuous)
        guess = reference_solution
        previous_solution = reference_solution

    return ScalingRecord(
        feedback_times,
        integral_flags,
        {variant_key(*key): closed_loop_cost for key in variants},
    )


def run_independent_scaling_once(
    configuration: RunConfiguration,
) -> ScalingRecord:
    problem = build_double_tank_problem()
    demand = demand_trajectory(problem.N + configuration.n_mpc, problem.dt)
    feedback_times = {}
    integral_flags = {}
    closed_loop_costs = {}
    for M_f in MF_VALUES:
        for solve_in_preparation in (False, True):
            algorithm = variant_algorithm(M_f, solve_in_preparation)
            require_external_solver(algorithm)
            run_configuration = replace(
                configuration,
                run_tag=(
                    f"{configuration.run_tag}_{M_f}_"
                    f"{int(solve_in_preparation)}"
                ),
            )
            record = run_advanced(
                problem,
                demand,
                algorithm,
                run_configuration,
            )
            key = variant_key(M_f, solve_in_preparation)
            feedback_times[key] = record.component_timings[
                "feedback.miqp_solution"
            ]
            integral_flags[key] = record.integer_found_in_rnlp
            closed_loop_costs[key] = record.closed_loop_cost
    return ScalingRecord(feedback_times, integral_flags, closed_loop_costs)


def scaling_measurement(
    configuration: RunConfiguration,
    warmup_runs: int,
    fix_rnlp: bool,
) -> ScalingRecord:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run = run_scaling_once if fix_rnlp else run_independent_scaling_once
    for warmup in range(warmup_runs):
        run(
            replace(configuration, run_tag=f"{configuration.run_tag}_warmup_{warmup}")
        )
    return run(configuration)


def repeated_scaling_measurements(
    configuration: RunConfiguration,
    fix_rnlp: bool,
) -> list[ScalingRecord]:
    records = []
    for repetition in range(REPETITIONS):
        print(f"Running M_f scaling repetition {repetition + 1}/{REPETITIONS} ...")
        run_configuration = replace(configuration, run_tag=f"repeat_{repetition}")
        if FRESH_PROCESS_MEASUREMENTS:
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                record = executor.submit(
                    scaling_measurement,
                    run_configuration,
                    WARMUP_RUNS,
                    fix_rnlp,
                ).result()
        else:
            record = scaling_measurement(
                run_configuration,
                WARMUP_RUNS,
                fix_rnlp,
            )
        records.append(record)
    return records


def eligible_times(
    records: list[ScalingRecord],
    M_f: int,
    solve_in_preparation: bool,
) -> np.ndarray:
    key = variant_key(M_f, solve_in_preparation)
    values = []
    for record in records:
        times = np.asarray(record.feedback_miqp_times[key], dtype=float)
        integral = np.asarray(record.integer_found_in_rnlp[key], dtype=bool)
        values.extend(times[~integral].tolist())
    return np.asarray(values, dtype=float)


def print_summary(records: list[ScalingRecord]) -> None:
    print("\nM_f scaling summary")
    print("+------+-------------+------------+")
    print("| M_f  | Prep MIQP   | Mean [s]   |")
    print("+------+-------------+------------+")
    for M_f in MF_VALUES:
        for solve_in_preparation in (False, True):
            values = eligible_times(records, M_f, solve_in_preparation)
            if values.size:
                mean = f"{np.mean(values):.6f}"
            else:
                mean = "-"
            print(
                f"| {M_f:>4} | {str(solve_in_preparation):<11} | "
                f"{mean:>10} |"
            )
    print("+------+-------------+------------+")


def plot_scaling(records: list[ScalingRecord]) -> None:
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    ):
        figure, axis = plt.subplots(figsize=(8, 5.5))
        for solve_in_preparation, label, color, marker in (
            (False, "MIQP in feedback only", "#d62728", "o"),
            (True, "MIQP warm-started from preparation", "#1f77b4", "^"),
        ):
            means = []
            for M_f in MF_VALUES:
                values = eligible_times(records, M_f, solve_in_preparation)
                if values.size == 0:
                    means.append(np.nan)
                    continue
                means.append(float(np.mean(values)))
            axis.plot(
                MF_VALUES,
                means,
                color=color,
                marker=marker,
                linewidth=1.4,
                label=label,
            )
        axis.set_xlabel(r"Integer horizon $M_f$")
        axis.set_ylabel("Mean MIQP feedback solve time [s]")
        axis.set_xticks(MF_VALUES)
        axis.grid(True, alpha=0.35)
        axis.legend()
        figure.tight_layout()
        plt.show()


def main() -> None:
    validate_configuration()
    configuration = RunConfiguration(
        n_mpc=N_MPC,
        gurobi_seed=GUROBI_SEED,
        gurobi_single_thread=GUROBI_SINGLE_THREAD,
    )
    records = repeated_scaling_measurements(configuration, FIX_RNLP)
    print_summary(records)
    plot_scaling(records)


if __name__ == "__main__":
    main()
