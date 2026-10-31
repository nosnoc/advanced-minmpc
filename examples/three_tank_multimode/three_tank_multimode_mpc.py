#!/usr/bin/env python3
# Adapted from the mintOC Three-Tank multimode benchmark.
# https://mintoc.de/index.php?title=Three_Tank_multimode_problem

from __future__ import annotations

import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

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
from three_tank_multimode_ocp import (
    MIQP_SOLVER,
    ThreeTankProblem,
    build_three_tank_problem,
    validate_one_hot,
)


N = 50
N_MPC = 120
RUN_BONMIN = False
WARM_START_BONMIN = True


def controller_options(
    problem: ThreeTankProblem,
    M_f: int | None,
) -> ThreePhaseOptions:
    return ThreePhaseOptions(
        mip_solver=MIQP_SOLVER,
        enable_miqp=True,
        enable_milp=False,
        enable_pycombina=False,
        binary_controls_one_hot=True,
        n_x=3,
        n_u=0,
        n_v=3,
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
        gurobi_options={
            "gurobi": {
                "OutputFlag": 0,
                "LogToConsole": 0,
                "IntFeasTol": 1e-9,
            }
        },
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
) -> ca.DM:
    return ca.DM(
        problem.integrator(
            x0=state,
            w=ca.DM(control),
            reference=problem.reference,
        )["xf"]
    )


def empty_record(problem: ThreeTankProblem) -> dict[str, list[Any]]:
    return {
        "states": [np.asarray(problem.initial_state, dtype=float).reshape(-1)],
        "binary": [],
        "total_preparation": [],
        "total_feedback": [],
    }


def append_step(
    problem: ThreeTankProblem,
    record: dict[str, list[Any]],
    state: ca.DM,
    control: np.ndarray,
    preparation_time: float,
    feedback_time: float,
) -> ca.DM:
    next_state = apply_control(problem, state, control)
    levels = np.asarray(next_state, dtype=float).reshape(-1)
    if np.any(levels < -1e-8):
        raise RuntimeError("The simulated tank levels became negative.")
    record["states"].append(levels)
    record["binary"].append(control)
    record["total_preparation"].append(preparation_time)
    record["total_feedback"].append(feedback_time)
    return next_state


def run_bonmin(problem: ThreeTankProblem) -> dict[str, list[Any]]:
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "three_tank_multimode_mpc_bonmin",
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
    initial_indices = np.asarray(problem.state_indices[0], dtype=int)
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
        lbx[initial_indices] = state
        ubx[initial_indices] = state
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
        state = append_step(problem, record, state, control, 0.0, elapsed)
    validate_record(problem, record)
    return record


def run_three_phase(problem: ThreeTankProblem) -> dict[str, list[Any]]:
    solver = ThreePhaseSolver(
        "three_tank_multimode_three_phase_mpc",
        problem.nlp,
        problem.discrete,
        controller_options(problem, None),
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    initial_indices = np.asarray(problem.state_indices[0], dtype=int)
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
        lbx[initial_indices] = state
        ubx[initial_indices] = state
        result = solver.solve(
            x0=guess,
            lbx=lbx,
            ubx=ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
            p=problem.p,
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
        )
    validate_record(problem, record)
    return record


def run_advanced(
    problem: ThreeTankProblem,
    M_f: int | None,
    name: str,
) -> dict[str, list[Any]]:
    def integrate(x: ca.DM, _u: ca.DM, v: ca.DM) -> ca.DM:
        return problem.integrator(
            x0=x,
            w=v,
            reference=problem.reference,
        )["xf"]

    controller = AdvancedStepMPC(
        name,
        problem.nlp,
        problem.discrete,
        controller_options(problem, M_f),
        integrator=integrate,
    )
    state = ca.DM(problem.initial_state)
    guess = ca.DM(problem.x0)
    previous_solution = None
    record = empty_record(problem)

    for _step in range(N_MPC):
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
        feedback = controller.feedback_phase(state, preparation, p=problem.p)
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
        )
    validate_record(problem, record)
    return record


def validate_record(
    problem: ThreeTankProblem,
    record: dict[str, list[Any]],
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
    states = np.asarray(record["states"], dtype=float)
    if states.shape != (N_MPC + 1, 3) or np.any(states < -1e-8):
        raise RuntimeError("The closed-loop state record is invalid.")
    validate_one_hot(record["binary"])


def as_array(record: dict[str, list[Any]], key: str) -> np.ndarray:
    return np.asarray(record[key], dtype=float)


def plot_closed_loop(
    problem: ThreeTankProblem,
    results: dict[str, dict[str, list[Any]]],
) -> None:
    state_time = np.arange(N_MPC + 1) * problem.dt
    control_time = np.arange(N_MPC) * problem.dt
    mpc_steps = np.arange(N_MPC)
    figure, axes = plt.subplots(5, 2, figsize=(13, 16))
    axes = axes.flat
    styles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    for (label, record), style in zip(results.items(), styles):
        states = as_array(record, "states")
        controls = as_array(record, "binary")
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
        axes[7].plot(
            mpc_steps,
            as_array(record, "total_preparation"),
            linestyle=style,
            label=label,
        )
        axes[8].plot(
            mpc_steps,
            as_array(record, "total_feedback"),
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
    axes[9].axis("off")
    titles = (
        "Tank 1 closed-loop level",
        "Tank 2 closed-loop level",
        "Tank 3 closed-loop level",
        "Applied mode $w_1$",
        "Applied mode $w_2$",
        "Applied mode $w_3$",
        "Selected mode",
        "Preparation time",
        "Feedback time",
    )
    ylabels = (
        "$x_1$",
        "$x_2$",
        "$x_3$",
        "$w_1$",
        "$w_2$",
        "$w_3$",
        "mode",
        "time [s]",
        "time [s]",
    )
    for index, (axis, title, ylabel) in enumerate(
        zip(axes[:9], titles, ylabels)
    ):
        axis.set_title(title)
        axis.set_xlabel("MPC step" if index >= 7 else "time")
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
    problem = build_three_tank_problem(N=N)
    results = {
        "Classic three-phase MIQP": run_three_phase(problem),
        "Advanced MIQP": run_advanced(
            problem,
            None,
            "three_tank_multimode_advanced_full",
        ),
        "Advanced MIQP ($M_f=2$)": run_advanced(
            problem,
            2,
            "three_tank_multimode_advanced_mf2",
        ),
        "Advanced MIQP ($M_f=10$)": run_advanced(
            problem,
            10,
            "three_tank_multimode_advanced_mf10",
        ),
    }
    if RUN_BONMIN:
        results["Bonmin"] = run_bonmin(problem)
    plot_closed_loop(problem, results)


if __name__ == "__main__":
    main()
