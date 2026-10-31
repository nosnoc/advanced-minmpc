"""Closed-loop MPC for a six-state needle with insertion/rotation and spherical obstacles.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)–(39).
https://arxiv.org/pdf/2002.09658v3
"""
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from mpc_implementation import AdvancedStepMPC, CaminoSolver, ThreePhaseOptions, ThreePhaseSolver, prepare_mpc_initial_guess
from steerable_needle_model import NeedleConfig, build_problem
from steerable_needle_plotting import plot_records

DT = 0.1
N = 20
ELL = 5  # Also try 4 or 8 samples.
M_O = 2
M_F = 6
#MPC_STEPS = 150
MPC_STEPS = 100
X_INITIAL = (0., 0., 0., 0., 0., 0.)
X_TARGET = (-2., 3.5, 10., 0., 0., 0.)
INITIAL_MODE = 0  # Insertion; pre-simulation dwell is already fulfilled.
INITIAL_REMAINING_DWELL = 0
RUN_THREE_PHASE = True
RUN_AS_MO = True
RUN_AS_MF = True
RUN_BONMIN = False
RUN_CAMINO = False
# Numerical choices, independent of the paper parameters: RK4 substeps and solver settings.
N_int_ocp = 2
N_int_sim = 4
MIP_SOLVER = "daqp"
# MIP_SOLVER = "gurobi"
CAMINO_MAX_TIME = 10.0  # Seconds per MPC solve.
CAMINO_MAX_ITERATIONS = 50
CAMINO_MIP_SOLVER = "gurobi"
SOLVER_LOGS = False
SHOW_PLOTS = True
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
FEASIBILITY_TOL = 1e-1
OUTPUT_DIRECTORY = Path(__file__).with_name("results")


def solver_options(problem, *, M_o=None, M_f=None, mip_solver="gurobi"):
    return ThreePhaseOptions(
        mip_solver=mip_solver, enable_miqp=True, enable_pycombina=M_o is not None,
        binary_controls_one_hot=True, n_x=6, n_u=2, n_v=2, N=problem.N,
        M_o=M_o, M_f=M_f, binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices, initial_state_indices=problem.state_indices[0],
        continuous_control_indices=problem.continuous_control_indices,
        exploit_structure_in_hessian_cvx=True,
        solve_miqp_in_preparation=False, solve_pnlp_in_feedback=False,
        ipopt_options={"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 1000,
                       "ipopt.tol": 1e-8, "ipopt.bound_relax_factor": 0.0},
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0,
                                   "IntFeasTol": 1e-9, "Seed": 210}},
        pycombina_options=problem.reconstruction_options(problem.history()))


def check_stats(stats):
    failures = [f"{phase}: {s.get('return_status', 'unknown')}" for phase, s in stats.items()
                if isinstance(s, dict) and s.get("success") is False]
    if failures:
        raise RuntimeError("; ".join(failures))


def make_bonmin(problem, name):
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin plugin is unavailable.")
    return ca.nlpsol(name, "bonmin", problem.nlp,
        {"discrete": problem.discrete, "print_time": False,
         "bonmin.bb_log_level": 0, "bonmin.nlp_log_level": 0})


def new_record(state, kind, *, M_o=None, M_f=None):
    return {"kind": kind, "M_o": M_o, "M_f": M_f, "states": [np.array(state)],
            "continuous": [], "binary": [], "objectives": [], "preparation": [], "feedback": [],
            "prediction_violation": [], "remaining_dwell": [], "failure": None,
            "setup_seconds": 0., "runtime_seconds": 0.}


def run_controller(problem, count, kind, *, M_o=None, M_f=None, mip_solver="gurobi"):
    c = problem.model.config
    state, history = np.array(c.x_initial), problem.history()
    record = new_record(state, kind, M_o=M_o, M_f=M_f)
    start = perf_counter()
    last, predicted, controller, solver = None, state.copy(), None, None
    k = 0
    try:
        if kind == "bonmin":
            solver = make_bonmin(problem, "needle_mpc_bonmin")
        elif kind == "camino":
            record["camino_settings"] = {"max_time": CAMINO_MAX_TIME,
                "max_iterations": CAMINO_MAX_ITERATIONS, "mip_solver": CAMINO_MIP_SOLVER}
            solver = CaminoSolver("needle_mpc_camino", problem.nlp, problem.discrete,
                **record["camino_settings"], hessian_not_psd=True)
        elif kind == "three_phase":
            solver = ThreePhaseSolver("needle_mpc_3p", problem.nlp, problem.discrete,
                                       solver_options(problem, mip_solver=mip_solver))
        else:
            controller = AdvancedStepMPC("needle_as_"+kind, problem.nlp, problem.discrete,
                solver_options(problem, M_o=M_o, M_f=M_f, mip_solver=mip_solver))
            solver = controller.solver
        record["setup_seconds"] = perf_counter()-start
        for k in range(count):
            prep_time = 0.
            lower, upper = problem.bounds(state, history)
            parameter = history.previous
            if kind not in ("bonmin", "camino"):
                solver.options.pycombina_options = problem.reconstruction_options(history)
            if controller is not None:
                prepared = controller.preparation_phase(
                    problem.x0, lower, upper, problem.lbg, problem.ubg,
                    predicted_state=predicted, previous_solution=last, p=parameter)
                prep_time = prepared.timings["total_preparation"]
                check_stats(prepared.stats)
                feedback = controller.feedback_phase(state, prepared, p=parameter)
                check_stats(feedback.stats)
                decision = feedback.solution["x"]
                feedback_time = feedback.timings["total_feedback"]
            else:
                guess = prepare_mpc_initial_guess(problem.x0, last, problem.state_indices,
                    problem.continuous_control_indices, problem.binary_indices, state, True)
                solve_start = perf_counter()
                args = dict(x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                result = solver(**args) if kind == "bonmin" else solver.solve(**args)
                feedback_time = perf_counter()-solve_start
                stats = {"bonmin": solver.stats()} if kind == "bonmin" else result.stats
                check_stats({"camino": stats} if kind == "camino" else stats)
                decision = result["x"] if kind == "bonmin" else result.solution["x"]
            violation = problem.check(decision, state, history, M_f=M_f,
                                      nonlinear=controller is None, tol=FEASIBILITY_TOL)
            _, u, v = problem.trajectories(decision)
            continuous, mode = u[0], np.rint(v[0])
            history.advance(mode, c.ell)
            predicted = np.asarray(problem.model.prediction_integrator(state, continuous, mode)).reshape(6)
            state = problem.model.step(state, continuous, mode)
            record["states"].append(state.copy())
            for key, value in (("continuous", continuous), ("binary", mode),
                ("objectives", float(problem.evaluate(decision, parameter)[0])),
                ("preparation", prep_time), ("feedback", feedback_time),
                ("prediction_violation", violation), ("remaining_dwell", history.remaining)):
                record[key].append(value)
            last = ca.DM(decision)
            if np.min(problem.model.clearance(state)) < -FEASIBILITY_TOL:
                raise RuntimeError("Nonlinear plant violates an obstacle constraint at a sampling node.")
            if (k+1) % 10 == 0:
                print(f"{kind}: {k+1}/{count} MPC steps", flush=True)
    except (RuntimeError, ValueError, ImportError) as exc:
        record["failure"] = f"step {k}: {exc}"
        print(f"{kind} FAILED / incomplete: {record['failure']}", flush=True)
    record["runtime_seconds"] = perf_counter()-start
    return record


def save_records(directory, problem, records, requested_steps, experiment, *, save_results=False):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    c, summary = problem.model.config, {}
    for label, r in records.items():
        states = np.asarray(r["states"])
        u = np.asarray(r["continuous"]).reshape(-1, 2)
        K = len(u)
        error = states[-1]-np.asarray(c.x_target)
        complete = K == requested_steps and r["failure"] is None
        metrics = {"completed_steps": K, "requested_steps": requested_steps,
            "status": "complete" if complete else "FAILED / incomplete", "failure": r["failure"],
            "objective_sum": float(sum(r["objectives"])),
            "realized_cost": float(0.01*np.sum(u*u)+10*np.dot(error, error)),
            "final_target_error": float(np.linalg.norm(error)),
            "final_position_error": float(np.linalg.norm(error[:3])),
            "minimum_node_clearance": float(np.min(problem.model.clearance(states))),
            "runtime_seconds": r["runtime_seconds"], "setup_seconds": r["setup_seconds"],
            "preparation_total_seconds": float(sum(r["preparation"])),
            "feedback_total_seconds": float(sum(r["feedback"])),
            "preparation_mean_seconds": float(np.mean(r["preparation"])) if K else 0.,
            "feedback_mean_seconds": float(np.mean(r["feedback"])) if K else 0.,
            "preparation_max_seconds": float(max(r["preparation"], default=0.)),
            "feedback_max_seconds": float(max(r["feedback"], default=0.)),
            "max_prediction_violation": max(r["prediction_violation"], default=0.),
            "M_o": r["M_o"], "M_f": r["M_f"]}
        if "camino_settings" in r:
            metrics["camino_settings"] = r["camino_settings"]
        summary[label] = metrics
        arrays = {key: np.asarray(r[key]) for key in ("states", "continuous", "binary",
            "objectives", "preparation", "feedback", "prediction_violation", "remaining_dwell")}
        for key in ("continuous", "binary"):
            arrays[key] = arrays[key].reshape(-1, 2)
        if "simulation_states" in r:
            arrays["simulation_states"] = r["simulation_states"]
            metrics["simulation_minimum_node_clearance"] = float(np.min(
                problem.model.clearance(r["simulation_states"])))
            metrics["max_simulation_prediction_difference"] = float(np.max(
                np.abs(states-r["simulation_states"])))
        if save_results:
            np.savez_compressed(directory/(r["kind"]+".npz"), **arrays, dt=c.dt,
                target=c.x_target, centers=c.centers, radius=c.radius, ell=c.ell,
                M_o=-1 if r["M_o"] is None else r["M_o"], M_f=-1 if r["M_f"] is None else r["M_f"],
                status=metrics["status"], failure=r["failure"] or "", label=label,
                config_json=json.dumps(asdict(c)), metrics_json=json.dumps(metrics))
        print(f"{label}: {metrics['status']} ({K}/{requested_steps}), "
            f"{'sum J (39)' if experiment == 'mpc' else 'J (38)'}={metrics['objective_sum']:.6f}, "
            f"realized J={metrics['realized_cost']:.6f}, target error={metrics['final_target_error']:.5f}, "
            f"min node clearance={metrics['minimum_node_clearance']:.6g}, "
            f"max prediction residual={metrics['max_prediction_violation']:.3g}, "
            f"runtime={r['runtime_seconds']:.3f}s; prep/feedback totals="
            f"{metrics['preparation_total_seconds']:.3f}/{metrics['feedback_total_seconds']:.3f}s "
            f"(means {metrics['preparation_mean_seconds']:.4f}/{metrics['feedback_mean_seconds']:.4f}s)", flush=True)
    if save_results:
        (directory/"summary.json").write_text(json.dumps({"config": asdict(c), "N": problem.N,
            "experiment": experiment, "metrics": summary}, indent=2), encoding="utf-8")
    return summary


def main():
    logging.basicConfig(level=logging.INFO if SOLVER_LOGS else logging.WARNING, format="%(message)s")
    c = NeedleConfig(dt=DT, ell=ELL, x_initial=X_INITIAL, x_target=X_TARGET,
        initial_mode=INITIAL_MODE, initial_remaining_dwell=INITIAL_REMAINING_DWELL,
        N_int_ocp=N_int_ocp, N_int_sim=N_int_sim)
    problem = build_problem(c, N)
    records = {}
    for label, kind, enabled, partial in (
        ("3P-MIQP", "three_phase", RUN_THREE_PHASE, {}),
        (f"AS-MIQP Mo={M_O}", "mo", RUN_AS_MO, {"M_o": M_O}),
        (f"AS-MIQP Mf={M_F}", "mf", RUN_AS_MF, {"M_f": M_F}),
        ("Bonmin", "bonmin", RUN_BONMIN, {}),
        ("CAMINO SBMIQP", "camino", RUN_CAMINO, {})):
        if enabled:
            records[label] = run_controller(problem, MPC_STEPS, kind, mip_solver=MIP_SOLVER, **partial)
    if not records:
        raise ValueError("Enable at least one comparison.")
    directory = OUTPUT_DIRECTORY/f"mpc_ell{ELL}_N{N}"
    save_records(directory, problem, records, MPC_STEPS, "mpc", save_results=SAVE_RESULTS)
    plot_records(problem, records, directory, "MPC", show=SHOW_PLOTS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
