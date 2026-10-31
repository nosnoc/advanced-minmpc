"""PyCombina and horizon-tail MPC comparisons for Lotka-Volterra fishing.

Model/transcription reference: CasADi, lotka_volterra_minlp.py.
https://github.com/casadi/casadi/blob/main/docs/examples/python/lotka_volterra_minlp.py
Benchmark: mintOC, Lotka Volterra fishing problem.
https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem
"""
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

from lotka_volterra_model import LotkaVolterraProblem, build_lotka_volterra_problem
from mpc_implementation import AdvancedStepMPC, ThreePhaseOptions


N_MPC = 5
MIQP_SOLVER = "daqp"


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    M_f: int | None = None
    M_o: int | None = None


@dataclass(slots=True)
class Result:
    states: np.ndarray
    controls: np.ndarray
    preparation_times: np.ndarray
    feedback_times: np.ndarray
    cumulative_cost: np.ndarray


VARIANTS = (
    Variant("AS-MIQP full"),
    Variant("AS-MIQP Mf=2", M_f=2),
    Variant("AS-MIQP Mf=10", M_f=10),
    Variant("AS-MIQP Mo=2", M_o=2),
    Variant("AS-MIQP Mo=10", M_o=10),
)


def solver_options(
    problem: LotkaVolterraProblem,
    variant: Variant,
) -> ThreePhaseOptions:
    daqp_mo = MIQP_SOLVER == "daqp" and variant.M_o is not None
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER,
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=variant.M_o is not None,
        n_x=2,
        n_u=0,
        n_v=1,
        N=problem.N,
        M_f=variant.M_o if daqp_mo else variant.M_f,
        M_o=None if daqp_mo else variant.M_o,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=[],
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=False,
        solve_pnlp_in_feedback=False,
        pycombina_options={
            "time_grid": np.linspace(0.0, problem.T, problem.N + 1)
        },
    )


def first_control(problem: LotkaVolterraProblem, decision: Any) -> float:
    value = float(ca.DM(decision)[problem.binary_indices[0]])
    rounded = float(np.rint(value))
    if abs(value - rounded) > 1e-5:
        raise RuntimeError(f"The applied binary control is fractional: {value}.")
    return rounded


def run_variant(
    problem: LotkaVolterraProblem,
    variant: Variant,
) -> Result:
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(x0=x, v=v[0])["xf"]

    name = (
        variant.label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("=", "_")
    )
    options = solver_options(problem, variant)
    controller = AdvancedStepMPC(
        f"lotka_volterra_pycombina_{name}",
        problem.nlp,
        problem.discrete,
        options,
        integrator=integrate,
    )
    if MIQP_SOLVER == "daqp" and variant.M_o is not None:
        # DAQP requires fixed PyCombina tail variables to be marked continuous.
        options.M_f = None
        options.M_o = variant.M_o
    state = ca.DM(problem.x0[problem.state_indices[0]])
    guess = ca.DM(problem.x0)
    previous_solution = None
    states = [np.asarray(state, dtype=float).reshape(-1)]
    controls: list[float] = []
    preparation_times: list[float] = []
    feedback_times: list[float] = []
    cumulative_cost: list[float] = []
    realized_cost = 0.0

    for _ in range(N_MPC):
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous_solution is None else None,
            previous_solution=previous_solution,
        )
        feedback = controller.feedback_phase(state, preparation)
        guess = ca.DM(feedback.solution["x"])
        previous_solution = guess
        control = first_control(problem, guess)
        integration = problem.integrator(x0=state, v=control)
        state = ca.DM(integration["xf"])
        realized_cost += float(integration["qf"])

        states.append(np.asarray(state, dtype=float).reshape(-1))
        controls.append(control)
        preparation_times.append(preparation.timings["total_preparation"])
        feedback_times.append(feedback.timings["total_feedback"])
        cumulative_cost.append(realized_cost)

    return Result(
        states=np.asarray(states),
        controls=np.asarray(controls),
        preparation_times=np.asarray(preparation_times),
        feedback_times=np.asarray(feedback_times),
        cumulative_cost=np.asarray(cumulative_cost),
    )


def format_table(headings: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(headings[column]), *(len(row[column]) for row in rows))
        for column in range(len(headings))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    return "\n".join((separator, row(headings), separator, *(row(r) for r in rows), separator))


def print_summary(results: dict[Variant, Result]) -> None:
    headings = (
        "Algorithm",
        "Mean prep [s]",
        "Max prep [s]",
        "Mean feedback [s]",
        "Max feedback [s]",
        "Closed-loop cost",
    )
    rows = [
        (
            variant.label,
            f"{np.mean(result.preparation_times):.4f}",
            f"{np.max(result.preparation_times):.4f}",
            f"{np.mean(result.feedback_times):.4f}",
            f"{np.max(result.feedback_times):.4f}",
            f"{result.cumulative_cost[-1]:.6f}",
        )
        for variant, result in results.items()
    ]
    print("\nPyCombina MPC comparison")
    print(format_table(headings, rows))


def plot_results(
    problem: LotkaVolterraProblem,
    results: dict[Variant, Result],
) -> None:
    state_time = np.arange(N_MPC + 1) * problem.T / problem.N
    control_time = np.arange(N_MPC) * problem.T / problem.N
    figure, axes = plt.subplots(3, 2, figsize=(11, 10))
    axes = axes.flat

    for variant, result in results.items():
        axes[0].plot(state_time, result.states[:, 0], label=variant.label)
        axes[1].plot(state_time, result.states[:, 1], label=variant.label)
        axes[2].step(
            control_time,
            result.controls,
            where="post",
            label=variant.label,
        )
        axes[3].plot(
            np.arange(N_MPC),
            result.preparation_times,
            label=variant.label,
        )
        axes[4].plot(
            np.arange(N_MPC),
            result.feedback_times,
            label=variant.label,
        )
        axes[5].plot(
            control_time,
            result.cumulative_cost,
            label=variant.label,
        )

    titles = (
        "Prey state",
        "Predator state",
        "Applied binary control",
        "Preparation time",
        "Feedback time",
        "Realized closed-loop cost",
    )
    ylabels = (
        "$x_1$",
        "$x_2$",
        "$v$",
        "time [s]",
        "time [s]",
        "accumulated cost",
    )
    for index, (axis, title, ylabel) in enumerate(zip(axes, titles, ylabels)):
        axis.set_title(title)
        axis.set_xlabel(
            "time [s]" if index in (0, 1, 2, 5) else "MPC step"
        )
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend(fontsize="small")
    axes[2].set_yticks((0, 1))
    figure.tight_layout()
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not ca.has_conic(MIQP_SOLVER):
        raise RuntimeError(
            f"The CasADi {MIQP_SOLVER} conic plugin is unavailable in WSL."
        )
    if importlib.util.find_spec("pycombina") is None:
        raise RuntimeError(
            "PyCombina is unavailable; follow README.md's WSL installation steps."
        )

    problem = build_lotka_volterra_problem()
    results = {
        variant: run_variant(problem, variant)
        for variant in VARIANTS
    }
    print_summary(results)
    plot_results(problem, results)


if __name__ == "__main__":
    main()
