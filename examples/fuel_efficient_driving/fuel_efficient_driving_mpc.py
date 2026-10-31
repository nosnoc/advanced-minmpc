"""Fuel-efficient MPC: longitudinal position/speed, torque, braking and six gears.

S. Mallick et al., Learning-Based MPC for Fuel Efficient Control of Autonomous
Vehicles with Discrete Gear Selection (2025), equations (1)–(12), Section VI.
https://arxiv.org/pdf/2503.11359v3
Source: https://github.com/SamuelMallick/mpcrl-vehicle-gears/tree/paper-2025
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
from mpc_implementation import AdvancedStepMPC, ThreePhaseOptions, ThreePhaseSolver, prepare_mpc_initial_guess
from fuel_efficient_driving_model import DrivingConfig, GEAR_NUMBERS, SOURCE_REVISION, build_problem, driving_record
from fuel_efficient_driving_plotting import plot_records

DT = 1.
N = 15
M_O = 4
M_F = 6
SIMULATION_TIME = 100.
SEED = 10
INITIAL_STATE = None  # Authors' initial-state sampling with seed SEED.
INITIAL_GEAR = None  # Highest rpm-feasible gear; overrides are numbered 1 through 6.
INITIAL_TORQUE = None  # Feasible steady-speed torque in the initial gear.
RUN_THREE_PHASE = True
RUN_AS_MO = True
RUN_AS_MF = True
RUN_BONMIN = False
# Numerical choices: requested RK4 replaces Euler; fuel/tracking costs remain sampled.
N_int_ocp = 2
N_int_sim = 4
MIP_SOLVER = "gurobi"
PYCOMBINA_BINARY_THRESHOLD = 1e-6
SOLVER_LOGS = False
SHOW_PLOTS = True
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
FEASIBILITY_TOL = 1e-5
OUTPUT_DIRECTORY = Path(__file__).with_name("results")


def solver_options(problem, *, M_o=None, M_f=None, mip_solver="gurobi",
                   binary_threshold=PYCOMBINA_BINARY_THRESHOLD):
    return ThreePhaseOptions(mip_solver=mip_solver, enable_miqp=True,
        enable_pycombina=M_o is not None, binary_controls_one_hot=True,
        integrality_tol=1e-6,  # Norm tolerance over the complete vector of gear binaries.
        pycombina_binary_threshold=binary_threshold,
        n_x=2, n_u=2, n_v=6, N=problem.N, M_o=M_o, M_f=M_f,
        binary_intervals=problem.binary_intervals, state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        solve_miqp_in_preparation=False, solve_pnlp_in_feedback=False,
        ipopt_options={"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 1000,
                       "ipopt.tol": 1e-8, "ipopt.bound_relax_factor": 0.},
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0,
                                   "IntFeasTol": 1e-9, "Seed": 10}},
        pycombina_options=problem.reconstruction_options(problem.initial_gear))


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
        "continuous": [], "binary": [], "fuel": [], "stage_cost": [], "rpm": [],
        "objectives": [], "preparation": [], "feedback": [], "prediction_violation": [],
        "violations": [], "failure": None, "failure_stats": {}, "setup_seconds": 0., "runtime_seconds": 0.}


def append_interval(record, model, state, u, v, next_state, reference):
    rpm, rate = model.outputs(state, u, v)
    fuel = float(rate)*model.config.dt
    record["states"].append(np.array(next_state))
    record["continuous"].append(np.array(u))
    record["binary"].append(np.array(v))
    record["rpm"].append([float(rpm), float(model.outputs(next_state, u, v)[0])])
    record["fuel"].append(fuel)
    record["stage_cost"].append(model.tracking_cost(state, reference)+fuel)


def run_controller(problem, reference, count, kind, *, M_o=None, M_f=None, mip_solver="gurobi",
                   binary_threshold=PYCOMBINA_BINARY_THRESHOLD):
    model, c = problem.model, problem.model.config
    state, previous_torque, previous_gear = problem.initial_state.copy(), problem.initial_input[0], problem.initial_gear.copy()
    record = new_record(state, kind, M_o=M_o, M_f=M_f)
    record["pycombina_binary_threshold"] = binary_threshold
    start = perf_counter()
    controller, solver, last, predicted, k = None, None, None, state.copy(), 0
    try:
        options = None if kind == "bonmin" else solver_options(problem, M_o=M_o, M_f=M_f,
            mip_solver=mip_solver, binary_threshold=binary_threshold)
        if kind == "bonmin":
            solver = make_bonmin(problem, "driving_mpc_bonmin")
        elif kind == "three_phase":
            solver = ThreePhaseSolver("driving_3p", problem.nlp, problem.discrete, options)
        else:
            controller = AdvancedStepMPC("driving_as_"+kind, problem.nlp, problem.discrete, options)
            solver = controller.solver
        record["setup_seconds"] = perf_counter()-start
        for k in range(count):
            parameter = problem.parameters(reference[k:k+problem.N+1], previous_torque, previous_gear)
            lower, upper = problem.bounds(state)
            prep_time = 0.
            if kind != "bonmin":
                solver.options.pycombina_options = problem.reconstruction_options(previous_gear)
            if controller is not None:
                prepared = controller.preparation_phase(problem.x0, lower, upper, problem.lbg, problem.ubg,
                    predicted_state=predicted, previous_solution=last, p=parameter)
                prep_time = prepared.timings["total_preparation"]
                check_stats(prepared.stats)
                feedback = controller.feedback_phase(state, prepared, p=parameter)
                check_stats(feedback.stats)
                decision, feedback_time = feedback.solution["x"], feedback.timings["total_feedback"]
            else:
                guess = prepare_mpc_initial_guess(problem.x0, last, problem.state_indices,
                    problem.continuous_control_indices, problem.binary_indices, state, True)
                solve_start = perf_counter()
                args = dict(x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                result = solver(**args) if kind == "bonmin" else solver.solve(**args)
                feedback_time = perf_counter()-solve_start
                check_stats({"bonmin": solver.stats()} if kind == "bonmin" else result.stats)
                decision = result["x"] if kind == "bonmin" else result.solution["x"]
            residual = problem.check(decision, parameter, state, M_f=M_f,
                                     nonlinear=controller is None, tol=FEASIBILITY_TOL)
            _, u, v = problem.trajectories(decision)
            continuous, gear = u[0], np.rint(v[0])
            next_state = model.step(state, continuous, gear)
            violations = model.physical_violations(state, continuous, gear, next_state, previous_torque, previous_gear)
            append_interval(record, model, state, continuous, gear, next_state, reference[k])
            for key, value in (("preparation", prep_time), ("feedback", feedback_time),
                ("prediction_violation", residual), ("violations", violations),
                ("objectives", float(problem.evaluate(decision, parameter)[0]))):
                record[key].append(value)
            predicted = np.asarray(model.prediction_integrator(state, continuous, gear)).reshape(2)
            state, previous_torque, previous_gear, last = next_state, continuous[0], gear, ca.DM(decision)
            scales = {"torque_Nm": c.torque_max, "brake_N": c.brake_max, "rpm": c.rpm_max,
                      "speed_m_s": c.speed_bounds[1], "speed_increment_m_s": c.acceleration_max*c.dt,
                      "torque_increment_Nm": c.torque_rate*c.dt, "gear": 1.}
            if any(value/scales[key] > FEASIBILITY_TOL for key, value in violations.items()):
                raise RuntimeError(f"Applied nonlinear plant constraint violations: {violations}")
            if (k+1) % 10 == 0:
                print(f"{kind}: {k+1}/{count} MPC steps", flush=True)
    except (RuntimeError, ValueError, ImportError) as exc:
        record["failure"] = f"step {k}: {exc}"
        if isinstance(solver, ThreePhaseSolver):
            for phase, function in (("nlp", solver._nlp_solver), ("miqp", solver._mip_solver)):
                try:
                    stats = function.stats()
                    record["failure_stats"][phase] = {key: stats.get(key) for key in ("success", "return_status")}
                except RuntimeError:
                    pass
        print(f"{kind} FAILED / incomplete: {record['failure']}", flush=True)
        if record["failure_stats"]:
            print(f"{kind} failure statuses: {record['failure_stats']}", flush=True)
    record["runtime_seconds"] = perf_counter()-start
    return record


def save_records(directory, problem, records, reference, count, experiment, changes, accelerations, *, save_results=False):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    model, c, summary = problem.model, problem.model.config, {}
    for label, r in records.items():
        K = len(r["binary"])
        states = np.asarray(r["states"])
        error = states-reference[:K+1]
        binary = np.asarray(r["binary"]).reshape(-1, 6)
        stage_sum = float(sum(r["stage_cost"]))
        terminal = model.tracking_cost(states[-1], reference[K])
        integral = np.all(np.abs(binary-np.rint(binary)) < 1e-6)
        gears = np.r_[GEAR_NUMBERS@problem.initial_gear, binary@GEAR_NUMBERS]
        metrics = {"completed_steps": K, "requested_steps": count,
            "status": "complete" if K == count and r["failure"] is None else "FAILED / incomplete",
            "failure": r["failure"], "failure_stats": r["failure_stats"],
            "cost": stage_sum+(terminal if experiment == "ocp" else 0.),
            "stage_cost_sum": stage_sum, "terminal_tracking_cost": terminal,
            "fuel_g": float(sum(r["fuel"])), "position_RMSE_m": float(np.sqrt(np.mean(error[:, 0]**2))),
            "speed_RMSE_m_s": float(np.sqrt(np.mean(error[:, 1]**2))),
            "gear_changes": int(np.count_nonzero(np.abs(np.diff(gears)) > 0.5)) if integral else None,
            "relaxed_gear_total_variation": float(np.sum(np.abs(np.diff(gears)))),
            "predicted_objective_sum": float(sum(r["objectives"])),
            "runtime_seconds": r["runtime_seconds"], "setup_seconds": r["setup_seconds"],
            "max_prediction_violation": float(max(r["prediction_violation"], default=0.)),
            "M_o": r["M_o"], "M_f": r["M_f"],
            "pycombina_binary_threshold": r.get("pycombina_binary_threshold")}
        for phase in ("preparation", "feedback"):
            values = r[phase]
            metrics.update({phase+"_total_seconds": float(sum(values)),
                phase+"_mean_seconds": float(np.mean(values)) if len(values) else 0.,
                phase+"_max_seconds": float(max(values, default=0.))})
        metrics["physical_violations"] = {key: max(row[key] for row in r["violations"])
            for key in (r["violations"][0] if r["violations"] else {})}
        if "rollout_states" in r:
            metrics["max_rollout_prediction_difference"] = float(np.max(np.abs(states-r["rollout_states"])))
        arrays = {key: np.asarray(r[key]) for key in ("states", "continuous", "binary", "rpm", "fuel",
            "stage_cost", "objectives", "preparation", "feedback", "prediction_violation")}
        for key, width in (("continuous", 2), ("binary", 6), ("rpm", 2)):
            arrays[key] = arrays[key].reshape(-1, width)
        if "rollout_states" in r:
            arrays["rollout_states"] = r["rollout_states"]
        if save_results:
            np.savez_compressed(directory/(r["kind"]+".npz"), **arrays,
                reference=reference, dt=c.dt, initial_state=problem.initial_state,
                initial_input=problem.initial_input, initial_gear=problem.initial_gear,
                change_points=changes, reference_accelerations=accelerations,
                M_o=-1 if r["M_o"] is None else r["M_o"], M_f=-1 if r["M_f"] is None else r["M_f"],
                status=metrics["status"], failure=r["failure"] or "", label=label,
                config_json=json.dumps(asdict(c)), metrics_json=json.dumps(metrics), source_revision=SOURCE_REVISION)
        summary[label] = metrics
        print(f"{label}: {metrics['status']} ({K}/{count}), cost={metrics['cost']:.4f}, "
            f"fuel={metrics['fuel_g']:.4f}g, position/speed RMSE="
            f"{metrics['position_RMSE_m']:.4f}m/{metrics['speed_RMSE_m_s']:.4f}m/s, "
            f"gear changes={metrics['gear_changes'] if integral else 'fractional tail'}, "
            f"runtime={r['runtime_seconds']:.3f}s, prep/feedback total="
            f"{metrics['preparation_total_seconds']:.3f}/{metrics['feedback_total_seconds']:.3f}s, "
            f"mean={metrics['preparation_mean_seconds']:.4f}/{metrics['feedback_mean_seconds']:.4f}s, "
            f"max prediction residual={metrics['max_prediction_violation']:.3g}", flush=True)
    if save_results:
        (directory/"summary.json").write_text(json.dumps({"config": asdict(c), "N": problem.N,
            "experiment": experiment, "source_revision": SOURCE_REVISION, "metrics": summary}, indent=2), encoding="utf-8")
    return summary


def main():
    logging.basicConfig(level=logging.INFO if SOLVER_LOGS else logging.WARNING, format="%(message)s")
    c = DrivingConfig(dt=DT, seed=SEED, N_int_ocp=N_int_ocp, N_int_sim=N_int_sim)
    count = int(round(SIMULATION_TIME/DT))
    state, reference, changes, accelerations = driving_record(c, count, N, INITIAL_STATE)
    problem = build_problem(c, N, state, INITIAL_GEAR, INITIAL_TORQUE)
    records = {}
    for label, kind, enabled, partial in (
        ("3P-MIQP", "three_phase", RUN_THREE_PHASE, {}),
        (f"AS-MIQP Mo={M_O}", "mo", RUN_AS_MO, {"M_o": M_O}),
        (f"AS-MIQP Mf={M_F}", "mf", RUN_AS_MF, {"M_f": M_F}),
        ("Bonmin", "bonmin", RUN_BONMIN, {})):
        if enabled:
            records[label] = run_controller(problem, reference, count, kind, mip_solver=MIP_SOLVER,
                binary_threshold=PYCOMBINA_BINARY_THRESHOLD, **partial)
    if not records:
        raise ValueError("Enable at least one comparison.")
    directory = OUTPUT_DIRECTORY/f"mpc_seed{SEED}_N{N}"
    save_records(directory, problem, records, reference, count, "mpc", changes, accelerations, save_results=SAVE_RESULTS)
    plot_records(problem, records, reference, directory, "MPC", show=SHOW_PLOTS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
