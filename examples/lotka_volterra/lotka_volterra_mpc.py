"""Standard and advanced-step MPC for the Lotka-Volterra fishing problem.

Model/transcription reference: CasADi, lotka_volterra_minlp.py.
https://github.com/casadi/casadi/blob/main/docs/examples/python/lotka_volterra_minlp.py
Benchmark: mintOC, Lotka Volterra fishing problem.
https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem
"""
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
from lotka_volterra_model import build_lotka_volterra_problem


N_MPC = 50
MIQP_SOLVER = "daqp"
MILP_SOLVER = "highs"
THREE_PHASE_M_F = 10
ADVANCED_M_F = 4
RUN_MILP_COMPARISONS = False
RUN_BONMIN = False
WARM_START_BONMIN = True


def options(
    problem, advanced: bool = False, solve_qp_after_milp: bool = True,
    use_milp: bool = False,
) -> ThreePhaseOptions:
    return ThreePhaseOptions(
        mip_solver=MILP_SOLVER if use_milp else MIQP_SOLVER,
        enable_miqp=not use_milp,
        enable_milp=use_milp,
        enable_pycombina=False,
        n_x=2,
        n_u=0,
        n_v=1,
        N=problem.N,
        M_f=ADVANCED_M_F if advanced else THREE_PHASE_M_F,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=[],
        initial_state_indices=problem.state_indices[0],
        solve_pnlp_in_feedback=False,
        solve_miqp_in_preparation=False,
        solve_milp_in_preparation=False,
        solve_qp_after_milp=solve_qp_after_milp,
    )


def simulate_step(problem, state: ca.DM, decision: ca.DM) -> tuple[ca.DM, float]:
    binary = float(decision[problem.binary_indices[0]])
    next_state = problem.integrator(x0=state, v=binary)["xf"]
    return next_state, binary


def run_standard(problem):
    solver = ThreePhaseSolver(
        "standard_mpc", problem.nlp, problem.discrete, options(problem)
    )
    state = ca.DM([0.5, 0.7])
    guess = problem.x0
    states, controls, timings = [np.asarray(state).reshape(-1)], [], []
    initial_indices = problem.state_indices[0]
    for step in range(N_MPC):
        if step:
            guess = shift_mpc_trajectory(
                guess,
                problem.state_indices,
                [],
                problem.binary_indices,
                state,
            )
        lbx, ubx = ca.DM(problem.lbx), ca.DM(problem.ubx)
        lbx[initial_indices] = state
        ubx[initial_indices] = state
        result = solver.solve(guess, lbx, ubx, problem.lbg, problem.ubg)
        guess = result.solution["x"]
        state, control = simulate_step(problem, state, guess)
        states.append(np.asarray(state).reshape(-1))
        controls.append(control)
        timings.append(
            {
                "total_preparation": 0.0,
                "total_feedback": result.timings["total"],
                "function_evaluation": 0.0,
                "miqp_update_construction": result.timings["miqp_construction"],
                "pycombina_solution": result.timings["pycombina_solution"],
                "miqp_solution": result.timings["miqp_solution"],
                "fixed_qp_construction": 0.0,
                "fixed_qp_solution": 0.0,
                "pnlp_solution": result.timings["pnlp_solver"],
            }
        )
    return np.asarray(states), np.asarray(controls), timings


def run_advanced(
    problem, solve_qp_after_milp: bool = False, use_milp: bool = False,
):
    def integrate(x, _u, v):
        return problem.integrator(x0=x, v=v[0])["xf"]

    method = f"milp_{'qp' if solve_qp_after_milp else 'raw'}" if use_milp else "miqp"
    controller = AdvancedStepMPC(
        f"advanced_{method}",
        problem.nlp,
        problem.discrete,
        options(
            problem,
            advanced=True,
            solve_qp_after_milp=solve_qp_after_milp,
            use_milp=use_milp,
        ),
        integrator=integrate,
    )
    state = ca.DM([0.5, 0.7])
    guess = problem.x0
    previous = None
    states, controls, timings = [np.asarray(state).reshape(-1)], [], []
    for _ in range(N_MPC):
        preparation = controller.preparation_phase(
            guess,
            problem.lbx,
            problem.ubx,
            problem.lbg,
            problem.ubg,
            predicted_state=state if previous is None else None,
            previous_solution=previous,
        )
        feedback = controller.feedback_phase(state, preparation)
        guess = feedback.solution["x"]
        previous = guess
        state, control = simulate_step(problem, state, guess)
        states.append(np.asarray(state).reshape(-1))
        controls.append(control)
        timings.append({**preparation.timings, **feedback.timings})
    return np.asarray(states), np.asarray(controls), timings


def run_bonmin(problem):
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "lotka_volterra_mpc_bonmin",
        "bonmin",
        problem.nlp,
        {
            "discrete": problem.discrete,
            "bonmin.bb_log_level": 0,
            "bonmin.nlp_log_level": 0,
            "print_time": False,
        },
    )
    state = ca.DM(problem.x0[problem.state_indices[0]])
    previous_solution = None
    states, controls, timings = [np.asarray(state).reshape(-1)], [], []
    initial_indices = problem.state_indices[0]
    for step in range(N_MPC):
        guess = prepare_mpc_initial_guess(
            problem.x0,
            previous_solution,
            problem.state_indices,
            [],
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
        )
        elapsed = perf_counter() - start
        stats = solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(
                f"Bonmin failed at MPC step {step}: "
                f"{stats.get('return_status', 'unknown status')}"
            )
        previous_solution = ca.DM(solution["x"])
        state, control = simulate_step(problem, state, previous_solution)
        states.append(np.asarray(state).reshape(-1))
        controls.append(control)
        timings.append(
            {
                "total_preparation": 0.0,
                "total_feedback": elapsed,
                "milp_solution": 0.0,
                "fixed_qp_solution": 0.0,
            }
        )
    return np.asarray(states), np.asarray(controls), timings


def timing_array(records, key):
    return np.asarray([entry.get(key, 0.0) for entry in records])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_lotka_volterra_problem()
    standard_states, standard_controls, standard_times = run_standard(problem)
    advanced_miqp = run_advanced(problem)
    results = [
        (
            standard_states,
            standard_controls,
            standard_times,
            "three-phase MIQP",
            "--",
        ),
        (*advanced_miqp, "AS-MINMPC", "-"),
    ]
    feedback_components = [
        (advanced_miqp[2], "AS-MINMPC: MIQP", "miqp_solution", "-"),
    ]
    if RUN_MILP_COMPARISONS:
        advanced_raw = run_advanced(problem, use_milp=True)
        advanced_qp = run_advanced(problem, solve_qp_after_milp=True, use_milp=True)
        results.extend([
            (*advanced_raw, "AS-MILP", "-."),
            (*advanced_qp, "AS-MILP+QP", ":"),
        ])
        feedback_components.extend([
            (advanced_raw[2], "AS-MILP: MILP", "milp_solution", "-."),
            (advanced_qp[2], "AS-MILP+QP: MILP", "milp_solution", "--"),
            (advanced_qp[2], "AS-MILP+QP: fixed QP", "fixed_qp_solution", ":"),
        ])
    if RUN_BONMIN:
        results.append((*run_bonmin(problem), "Bonmin", ":"))

    samples = np.arange(N_MPC + 1)
    control_samples = np.arange(N_MPC)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for states, controls, timing, label, style in results:
        axes[0, 0].plot(samples, states[:, 0], style, label=label)
        axes[0, 1].plot(samples, states[:, 1], style, label=label)
        axes[1, 0].step(control_samples, controls, where="post", label=label)
        axes[1, 1].plot(timing_array(timing, "total_preparation"), label=label)
        axes[2, 0].plot(timing_array(timing, "total_feedback"), label=label)
    for timing, label, key, style in feedback_components:
        axes[2, 1].semilogy(
            timing_array(timing, key), style, label=label
        )

    titles = (
        "$x_0$ closed loop",
        "$x_1$ closed loop",
        "binary control $v$ (no continuous control in this model)",
        "preparation time",
        "feedback time",
        "advanced feedback components (log scale)",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("MPC step")
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
