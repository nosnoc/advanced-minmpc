#!/usr/bin/env python3
# Scaling benchmark for the mintOC Three-Tank multimode problem.
# Model reference: mintOC, Three Tank multimode problem.
# https://mintoc.de/index.php?title=Three_Tank_multimode_problem

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
import sys

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import AdvancedStepMPC
from three_tank_multimode_benchmark import (
    Algorithm,
    apply_control,
    first_control,
    options,
    require_external_solver,
    run_advanced,
    solver_name,
)
from three_tank_multimode_ocp import T, build_three_tank_problem


N = 100
N_MPC = 80
MF_VALUES = (2, 4, 8, 10, 20, 40, 60, 80, N)
REPETITIONS = 1
RUN_WARMUP = False
FIX_RNLP = True


@dataclass(slots=True)
class ScalingRecord:
    feedback_miqp_times: dict[str, list[float]]
    integer_found_in_rnlp: dict[str, list[bool]]


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
    if N < 1 or N_MPC < 1:
        raise ValueError("N and N_MPC must be positive.")
    if REPETITIONS < 1:
        raise ValueError("REPETITIONS must be positive.")
    if len(set(MF_VALUES)) != len(MF_VALUES):
        raise ValueError("MF_VALUES must not contain duplicates.")
    if any(not isinstance(value, int) or not 1 <= value <= N for value in MF_VALUES):
        raise ValueError(f"Every M_f must be an integer in [1, {N}].")
    if N not in MF_VALUES:
        raise ValueError(f"MF_VALUES must include N={N}.")


def build_controller(problem, algorithm, run_tag):
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(
            x0=x,
            w=v,
            reference=problem.reference,
        )["xf"]

    return AdvancedStepMPC(
        solver_name(algorithm, run_tag),
        problem.nlp,
        problem.discrete,
        options(problem, algorithm),
        integrator=integrate,
    )


def variants() -> dict[tuple[int, bool], Algorithm]:
    return {
        (M_f, solve_in_preparation): variant_algorithm(M_f, solve_in_preparation)
        for M_f in MF_VALUES
        for solve_in_preparation in (False, True)
    }


def run_fixed_rnlp(run_tag: str) -> ScalingRecord:
    problem = build_three_tank_problem(N=N, dt=T / N)
    enabled = variants()
    for algorithm in enabled.values():
        require_external_solver(algorithm)
    controllers = {
        key: build_controller(problem, algorithm, f"{run_tag}_{key[0]}_{int(key[1])}")
        for key, algorithm in enabled.items()
    }
    reference_key = (N, False)
    reference_controller = controllers[reference_key]
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    times = {variant_key(*key): [] for key in enabled}
    integral = {variant_key(*key): [] for key in enabled}

    for _ in range(N_MPC):
        shared = reference_controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
            p=problem.p,
        )
        if not shared.stats["rnlp"].get("success", False):
            raise RuntimeError(
                f"RNLP failed: {shared.stats['rnlp'].get('return_status', 'unknown')}"
            )
        integer_found = bool(shared.stats.get("integer_found_in_rnlp", False))
        feedback_results = {}
        for key, controller in controllers.items():
            prepared = replace(
                shared,
                predicted_miqp_solution=None,
                predicted_milp_solution=None,
            )
            if key[1] and not integer_found:
                predicted, _, _ = controller.solver.solve_mip(
                    shared.quadratic_model,
                    shared.lbx,
                    shared.ubx,
                    shared.lbg,
                    shared.ubg,
                    None,
                    None,
                    log_call=False,
                )
                prepared = replace(prepared, predicted_miqp_solution=predicted)
            feedback = controller.feedback_phase(state, prepared, p=problem.p)
            if not feedback.stats["miqp"].get("success", False):
                raise RuntimeError(
                    f"MIQP failed for M_f={key[0]}, prep={key[1]}: "
                    f"{feedback.stats['miqp'].get('return_status', 'unknown')}"
                )
            name = variant_key(*key)
            times[name].append(float(feedback.timings["miqp_solution"]))
            integral[name].append(integer_found)
            feedback_results[key] = feedback

        reference_solution = ca.DM(feedback_results[reference_key].solution["x"])
        control = first_control(problem, reference_solution)
        state, _ = apply_control(problem, state, control)
        guess = reference_solution
        previous_solution = reference_solution
    return ScalingRecord(times, integral)


def run_independent(run_tag: str) -> ScalingRecord:
    problem = build_three_tank_problem(N=N, dt=T / N)
    times = {}
    integral = {}
    for key, algorithm in variants().items():
        require_external_solver(algorithm)
        record = run_advanced(
            problem,
            algorithm,
            f"{run_tag}_{key[0]}_{int(key[1])}",
            N_MPC,
        )
        name = variant_key(*key)
        times[name] = record.miqp_feedback_times
        integral[name] = record.integer_found_in_rnlp
    return ScalingRecord(times, integral)


def run_once(run_tag: str) -> ScalingRecord:
    return run_fixed_rnlp(run_tag) if FIX_RNLP else run_independent(run_tag)


def measurements() -> list[ScalingRecord]:
    records = []
    for repetition in range(REPETITIONS):
        if RUN_WARMUP:
            run_once(f"warmup_{repetition}")
        print(f"Running repetition {repetition + 1}/{REPETITIONS} ...")
        records.append(run_once(f"repeat_{repetition}"))
    return records


def eligible_times(
    records: list[ScalingRecord],
    M_f: int,
    solve_in_preparation: bool,
) -> np.ndarray:
    key = variant_key(M_f, solve_in_preparation)
    values = []
    for record in records:
        timing = np.asarray(record.feedback_miqp_times[key], dtype=float)
        integral = np.asarray(record.integer_found_in_rnlp[key], dtype=bool)
        values.extend(timing[~integral].tolist())
    return np.asarray(values, dtype=float)


def print_summary(records: list[ScalingRecord]) -> None:
    print("\nM_f scaling summary")
    print("+------+-------------+------------+")
    print("| M_f  | Prep MIQP   | Mean [s]   |")
    print("+------+-------------+------------+")
    for M_f in MF_VALUES:
        for solve_in_preparation in (False, True):
            values = eligible_times(records, M_f, solve_in_preparation)
            mean = "-" if values.size == 0 else f"{np.mean(values):.6f}"
            print(
                f"| {M_f:>4} | {str(solve_in_preparation):<11} | {mean:>10} |"
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
            means = [
                np.mean(eligible_times(records, M_f, solve_in_preparation))
                for M_f in MF_VALUES
            ]
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
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    validate_configuration()
    records = measurements()
    print_summary(records)
    plot_scaling(records)


if __name__ == "__main__":
    main()
