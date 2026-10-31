"""Closed-loop needle benchmarks: nonlinear RK4, insertion/rotation and dwell times.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)-(39).
https://arxiv.org/pdf/2002.09658v3
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import warnings

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
PROJECT = EXAMPLE.parents[1]
ROOT = PROJECT.parent
sys.path[:0] = [str(EXAMPLE), str(PROJECT / "src")]
import steerable_needle_mpc as mpc
from steerable_needle_model import NeedleConfig, build_problem
from mpc_implementation import AdvancedStepMPC, CaminoSolver, ThreePhaseSolver, prepare_mpc_initial_guess

ELL_VALUES = (5, 5)  # Minimum dwell lengths [samples]; cases are (ell, N).
N_VALUES = (20, 40)  # Prediction horizons [control intervals].
MF_VALUES = (1, 2, 5, 8, 10)  # Integer prefixes; the remaining tail is relaxed.
MO_VALUES = (1, 2, 5, 8, 10)  # Optimized integer prefixes; the tail is fixed to PyCombina.
SWEEP_CASE = (mpc.ELL, 40)
SWEEP_VALUES = tuple(range(1, 40+1))
SELECTED_BATCHES = ("pareto", "sweep", "camino")
METHODS = ("ThreePhase", "CAMINO", "AS-MINMPC", "Mo", "Mf")
N_MPC = 150  # Applied control moves; simulated duration = N_MPC*DT seconds.
REPETITIONS = 1
OVERWRITE = False  # Reuse matching completed runs; retry failed/unavailable runs.
FRESH_PROCESS = True  # Sequential isolated workers, matching the chiller benchmark.
RESULTS_DIRECTORY = HERE / "results"

DT = mpc.DT
X_INITIAL, X_TARGET = mpc.X_INITIAL, mpc.X_TARGET
INITIAL_MODE = mpc.INITIAL_MODE
INITIAL_REMAINING_DWELL = mpc.INITIAL_REMAINING_DWELL
N_INT_OCP, N_INT_SIM = mpc.N_int_ocp, mpc.N_int_sim
MIP_SOLVER = "daqp"  # ThreePhase/AS/Mo/Mf; CAMINO retains its own backend below.
HESSIAN_EPSILON = 1e-4  # Numerical convexification floor for DAQP conditioning; shared by all three-phase/AS methods.
GUROBI_SEED = 210
GUROBI_SINGLE_THREAD = False
COLLECT_GUROBI_NODES = MIP_SOLVER == "gurobi"
PYCOMBINA_WARM_START = False  # Mo always uses it; full-horizon AS-MINMPC never does.
PYCOMBINA_BINARY_THRESHOLD = 1e-6
SOLVE_MIQP_IN_PREPARATION = False
SHIFT_INITIAL_GUESS = True
IPOPT_TOL, IPOPT_MAX_ITER = 1e-8, 1000
IPOPT_MAX_TIME = None
GUROBI_TIME_LIMIT = None
GUROBI_OPTIONS = {"IntFeasTol": 1e-6, "MIPGap": 0., "MIPGapAbs": 1e-6}
CAMINO_MAX_TIME = mpc.CAMINO_MAX_TIME  # Per MPC solve, using the current needle script setting.
CAMINO_MAX_ITERATIONS = mpc.CAMINO_MAX_ITERATIONS
CAMINO_MIP_SOLVER = mpc.CAMINO_MIP_SOLVER
FEASIBILITY_TOL = mpc.FEASIBILITY_TOL
SOLVER_LOGS = False
SCHEMA = 1


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()
                if k not in ("pool_solutions", "pool_obj_val", "iterations")}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.ndarray, ca.DM)):
        return json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_text(value):
    return json.dumps(json_value(value), sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()[:16]


def settings():
    return dict(dt=DT, x_initial=X_INITIAL, x_target=X_TARGET, initial_mode=INITIAL_MODE,
        initial_remaining_dwell=INITIAL_REMAINING_DWELL, N_int_ocp=N_INT_OCP, N_int_sim=N_INT_SIM,
        fresh_process=FRESH_PROCESS, mip_solver=MIP_SOLVER, hessian_epsilon=HESSIAN_EPSILON, seed=GUROBI_SEED,
        single_thread=GUROBI_SINGLE_THREAD, collect_gurobi_nodes=COLLECT_GUROBI_NODES,
        pycombina_warm_start=PYCOMBINA_WARM_START, binary_threshold=PYCOMBINA_BINARY_THRESHOLD,
        prep_miqp=SOLVE_MIQP_IN_PREPARATION, shift=SHIFT_INITIAL_GUESS,
        ipopt_tol=IPOPT_TOL, ipopt_max_iter=IPOPT_MAX_ITER, ipopt_max_time=IPOPT_MAX_TIME,
        gurobi_options=dict(GUROBI_OPTIONS), gurobi_time_limit=GUROBI_TIME_LIMIT,
        camino_time=CAMINO_MAX_TIME, camino_iterations=CAMINO_MAX_ITERATIONS,
        camino_backend=CAMINO_MIP_SOLVER, feasibility_tol=FEASIBILITY_TOL, solver_logs=SOLVER_LOGS)


def describe(options):
    print(f"Needle: dt={options['dt']:g}s, RK4={options['N_int_ocp']}/{options['N_int_sim']}; "
          f"backend={options['mip_solver']}, Hessian floor={options['hessian_epsilon']:g}, "
          f"seed={options['seed']}, single_thread={options['single_thread']}.")
    print("Cost: 0.01*sum ||u_applied||^2 + 10*||x_final-x_target||^2; no dt factor.")
    print(f"3P/CAMINO: all feedback; AS: prep MIQP={options['prep_miqp']}, feedback PNLP=False. "
          "All methods carry mode history and remaining dwell.", flush=True)


def make_jobs(cases, methods, mf_values, mo_values, repetitions, steps, options=None):
    if repetitions < 1 or steps < 1:
        raise ValueError("Repetitions and N_MPC must be positive.")
    options = settings() if options is None else options
    jobs = []
    for ell, n in dict.fromkeys(cases):
        if not isinstance(ell, int) or ell < 1 or not isinstance(n, int) or n < 2:
            raise ValueError("Require integer ell >= 1 and N >= 2.")
        for method in methods:
            if method not in METHODS:
                raise ValueError(f"Unknown needle method: {method}")
            horizons = mo_values if method == "Mo" else mf_values if method == "Mf" else (None,)
            for h in horizons:
                if h is not None and (not isinstance(h, int) or not 1 <= h <= n):
                    raise ValueError(f"{method} horizon {h} must lie in [1, {n}].")
                for rep in range(repetitions):
                    jobs.append(dict(ell=ell, N=n, steps=steps, method=method, horizon=h,
                                     repetition=rep, options=dict(options)))
    return jobs


def identity(job):
    o, method = job["options"], job["method"]
    config = NeedleConfig(dt=o["dt"], ell=job["ell"], x_initial=o["x_initial"], x_target=o["x_target"],
        initial_mode=o["initial_mode"], initial_remaining_dwell=o["initial_remaining_dwell"],
        N_int_ocp=o["N_int_ocp"], N_int_sim=o["N_int_sim"])
    files = [Path(__file__), EXAMPLE / "steerable_needle_model.py", EXAMPLE / "steerable_needle_mpc.py"]
    files += list((PROJECT / "src" / "mpc_implementation").glob("*.py"))
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    scenario = dict(config=asdict(config), N=job["N"], steps=job["steps"], source=source,
        casadi=ca.__version__, numpy=np.__version__, schema=SCHEMA,
        fresh_process=o["fresh_process"])
    keys = ["shift", "feasibility_tol"]
    if method == "CAMINO":
        keys += ["camino_time", "camino_iterations", "camino_backend"]
    else:
        keys += ["mip_solver", "hessian_epsilon", "seed", "single_thread", "gurobi_options", "gurobi_time_limit",
                 "ipopt_tol", "ipopt_max_iter", "ipopt_max_time", "collect_gurobi_nodes",
                 "pycombina_warm_start", "binary_threshold"]
        if method in ("AS-MINMPC", "Mo", "Mf"):
            keys += ["prep_miqp"]
    algorithm = {k: o[k] for k in keys}
    if method == "AS-MINMPC":
        algorithm["pycombina_warm_start"] = False
    effective = dict(scenario=scenario, method=method, horizon=job["horizon"], algorithm=algorithm)
    return effective, digest(scenario), digest(effective)


def run_path(job, directory):
    _, _, key = identity(job)
    h = "" if job["horizon"] is None else f"_{job['horizon']}"
    return Path(directory) / f"ell{job['ell']}_N{job['N']}_{job['method']}{h}_{key}_r{job['repetition']:03d}.npz"


def make_options(problem, job, result_file=None):
    o, method = job["options"], job["method"]
    base = mpc.solver_options(problem, M_o=job["horizon"] if method == "Mo" else None,
                              M_f=job["horizon"] if method == "Mf" else None, mip_solver=o["mip_solver"])
    ipopt = {**base.ipopt_options, "ipopt.tol": o["ipopt_tol"], "ipopt.max_iter": o["ipopt_max_iter"]}
    if o["ipopt_max_time"] is not None:
        ipopt["ipopt.max_cpu_time"] = o["ipopt_max_time"]
    gurobi = {**base.gurobi_options["gurobi"], **o["gurobi_options"], "Seed": o["seed"]}
    if o["single_thread"]:
        gurobi["Threads"] = 1
    if o["gurobi_time_limit"] is not None:
        gurobi["TimeLimit"] = o["gurobi_time_limit"]
    if result_file is not None:
        gurobi["ResultFile"] = str(result_file)
    return replace(base, enable_pycombina=False if method == "AS-MINMPC" else method == "Mo" or o["pycombina_warm_start"],
        pycombina_binary_threshold=o["binary_threshold"],
        solve_miqp_in_preparation=o["prep_miqp"] if method != "ThreePhase" else False,
        shift_initial_guess_enabled=False, hessian_epsilon=o["hessian_epsilon"],
        ipopt_options=ipopt, gurobi_options={"gurobi": gurobi})


def gurobi_node_count(path, stats):
    if path is None:
        return np.nan
    mip = stats.get("miqp")
    if stats.get("integer_found_in_rnlp") or mip is None or mip.get("skipped", False):
        return 0.
    if not path.exists():
        warnings.warn("Gurobi did not produce node-count data for this MIQP.", RuntimeWarning)
        return np.nan
    value = json.loads(path.read_text())["SolutionInfo"].get("NodeCount")
    if value is None:
        warnings.warn("Gurobi NodeCount is unavailable.", RuntimeWarning)
        return np.nan
    return float(value)


def new_record(state):
    return dict(states=[np.array(state)], continuous=[], binary=[], stage_cost=[], objectives=[],
        preparation=[], feedback=[], remaining_dwell=[], prediction_violation=[], steps=[], failure=None)


def simulate(job):
    effective, scenario_key, setting_key = identity(job)
    metadata = dict(job=job, effective=effective, scenario_key=scenario_key, setting_key=setting_key,
        preparation_applicable=job["method"] in ("AS-MINMPC", "Mo", "Mf"), fallback_count=0, fallback_used=False,
        cost_convention="0.01*sum applied squared inputs + 10*final full-state squared target error; no dt factor",
        timing_convention="Wall seconds, shifting/checks/node I/O included; plant integration and persistence excluded; setup separate")
    o, method = job["options"], job["method"]
    c = NeedleConfig(**effective["scenario"]["config"])
    state, predicted, record = np.array(c.x_initial), np.array(c.x_initial), new_record(c.x_initial)
    node_file = None
    initialization_start = perf_counter()
    try:
        problem = build_problem(c, job["N"])
        history, controller, solver = problem.history(), None, None
        logging.basicConfig(level=logging.INFO if o["solver_logs"] else logging.ERROR, format="%(message)s", force=True)
        if method == "CAMINO":
            solver = CaminoSolver("needle_benchmark_camino", problem.nlp, problem.discrete,
                mip_solver=o["camino_backend"], max_time=o["camino_time"],
                max_iterations=o["camino_iterations"], hessian_not_psd=True)
        else:
            if o["collect_gurobi_nodes"] and o["mip_solver"] == "gurobi":
                handle, filename = tempfile.mkstemp(prefix="gurobi_nodes_", suffix=".json")
                os.close(handle)
                node_file = Path(filename)
            options = make_options(problem, job, node_file)
            metadata["effective_solver_options"] = asdict(options)
            if method == "ThreePhase":
                solver = ThreePhaseSolver("needle_benchmark_3p", problem.nlp, problem.discrete, options)
            else:
                controller = AdvancedStepMPC("needle_benchmark_as", problem.nlp, problem.discrete, options)
                solver = controller.solver
        metadata["initialization_s"] = perf_counter()-initialization_start
        last = None
        for k in range(job["steps"]):
            start, prep, feedback, details = perf_counter(), 0., 0., {}
            phase, feedback_start = ("preparation" if controller else "feedback"), start
            try:
                if node_file is not None:
                    node_file.unlink(missing_ok=True)
                lower, upper = problem.bounds(state, history)
                parameter = history.previous
                guess = prepare_mpc_initial_guess(problem.x0, last, problem.state_indices,
                    problem.continuous_control_indices, problem.binary_indices, state, o["shift"])
                if method != "CAMINO":
                    solver.options.pycombina_options = problem.reconstruction_options(history)
                if controller:
                    prepared = controller.preparation_phase(guess, lower, upper, problem.lbg, problem.ubg,
                        predicted_state=predicted, previous_solution=last, p=parameter)
                    details = dict(preparation_stats=prepared.stats, timings=dict(prepared.timings))
                    mpc.check_stats(prepared.stats)
                    details["preparation_miqp_nodes"] = gurobi_node_count(node_file, prepared.stats)
                    prep = perf_counter()-start
                    phase, feedback_start = "feedback", perf_counter()
                    if node_file is not None:
                        node_file.unlink(missing_ok=True)
                    result = controller.feedback_phase(state, prepared, p=parameter)
                    details["stats"] = result.stats
                    details["timings"].update(result.timings)
                else:
                    result = solver.solve(x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                    details = dict(stats={"camino": result.stats} if method == "CAMINO" else result.stats,
                                   timings=dict(result.timings))
                mpc.check_stats(details["stats"])
                details["feedback_miqp_nodes"] = gurobi_node_count(node_file, details["stats"])
                decision = result.solution["x"]
                violation = problem.check(decision, state, history,
                    M_f=job["horizon"] if method == "Mf" else None,
                    nonlinear=controller is None, tol=o["feasibility_tol"])
                _, u, v = problem.trajectories(decision)
                mode = np.rint(v[0])
                if np.max(np.abs(v[0]-mode)) > 1e-6 or mode.sum() != 1 or not np.all(np.isin(mode, (0, 1))):
                    raise RuntimeError("The applied mode is not integral and one-hot.")
                history.advance(mode, c.ell)
                feedback = perf_counter()-feedback_start
                details["timings"].update(measured_preparation=prep, measured_feedback=feedback)
                phase = "simulation"
                predicted = np.asarray(problem.model.prediction_integrator(state, u[0], mode)).reshape(6)
                state = problem.model.step(state, u[0], mode)
                record["states"].append(state.copy())
                for key, value in (("continuous", u[0]), ("binary", mode), ("stage_cost", .01*float(u[0]@u[0])),
                    ("objectives", float(problem.evaluate(decision, parameter)[0])), ("preparation", prep),
                    ("feedback", feedback), ("remaining_dwell", history.remaining),
                    ("prediction_violation", violation), ("steps", details)):
                    record[key].append(value)
                last = ca.DM(decision)
                if np.min(problem.model.clearance(state)) < -o["feasibility_tol"]:
                    raise RuntimeError("The nonlinear plant violates an obstacle constraint at a prediction node.")
            except Exception as error:
                record["failure"] = dict(step=k, type=type(error).__name__, message=str(error), phase=phase,
                    elapsed_s=perf_counter()-start, preparation_s=perf_counter()-start if phase == "preparation" else prep,
                    feedback_s=0. if phase == "preparation" else perf_counter()-feedback_start, details=json_value(details))
                break
        error = state-np.asarray(c.x_target)
        metadata["metrics"] = dict(final_target_error=float(np.linalg.norm(error)),
            final_position_error=float(np.linalg.norm(error[:3])),
            minimum_node_clearance=float(np.min(problem.model.clearance(record["states"]))))
    except Exception as error:
        record["failure"] = dict(step=0, type=type(error).__name__, message=str(error),
                                 phase="initialization", elapsed_s=perf_counter()-initialization_start)
    finally:
        if node_file is not None:
            node_file.unlink(missing_ok=True)
    return metadata, record


def save_run(path, metadata, record, messages=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    k = len(record["stage_cost"])
    names = ("states", "continuous", "binary", "stage_cost", "objectives", "preparation", "feedback",
             "remaining_dwell", "prediction_violation")
    arrays = {key: np.asarray(record[key], dtype=float) for key in names}
    for key, width in (("states", 6), ("continuous", 2), ("binary", 2)):
        arrays[key] = arrays[key].reshape(-1, width)
    if record["failure"] is None and not all(np.all(np.isfinite(a)) for a in arrays.values()):
        record["failure"] = dict(step=k, type="NonfiniteResult", message="Nonfinite trajectory, cost, or timing.")
    complete = record["failure"] is None and k == metadata["job"]["steps"]
    failure = record["failure"]
    status = "completed" if complete else "unavailable" if failure and any(s in failure["message"].lower()
        for s in ("unavailable", "not found", "no module", "license", "licence", "cannot load")) else "failed"
    metadata.update(status=status, completed_steps=k, warnings=list(messages), failure=failure)
    target = np.asarray(metadata["effective"]["scenario"]["config"]["x_target"])
    terminal = 10*float(np.sum((arrays["states"][-1]-target)**2))
    cumulative_input = np.r_[0., np.cumsum(arrays["stage_cost"])]
    cumulative = cumulative_input.copy()
    cumulative[-1] += terminal
    for key in ("preparation_miqp_nodes", "feedback_miqp_nodes"):
        arrays[key] = np.asarray([s.get(key, np.nan) for s in record["steps"]], dtype=float)
    arrays["integer_found_in_rnlp"] = np.asarray([bool(s.get("stats", {}).get("integer_found_in_rnlp", False))
                                                 for s in record["steps"]])
    components = sorted({key for s in record["steps"] for key in s.get("timings", {})})
    times = np.array([[s.get("timings", {}).get(key, 0.) for key in components] for s in record["steps"]])
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays, cumulative_cost=cumulative, cumulative_input_cost=cumulative_input,
        terminal_cost=terminal, total_cost=cumulative[-1], completed=complete, fallback_used=False,
        component_names=np.asarray(components, dtype=str), component_times=times,
        metadata=json_text(metadata), step_details=json_text(record["steps"]))
    os.replace(temporary, path)
    return status


def execute(job, directory):
    cache = ROOT / "tmp" / "needle_benchmarks"
    cache.mkdir(parents=True, exist_ok=True)
    original_temp = tempfile.tempdir
    with tempfile.TemporaryDirectory(dir=cache, prefix="run_", ignore_cleanup_errors=True) as work:
        try:
            tempfile.tempdir = work
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metadata, record = simulate(job)
            return save_run(run_path(job, directory), metadata, record, sorted({str(w.message) for w in caught}))
        finally:
            tempfile.tempdir = original_temp


def run_campaign(jobs, directory=RESULTS_DIRECTORY, *, overwrite=OVERWRITE, fresh_process=FRESH_PROCESS):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job["options"] = {**job["options"], "fresh_process": bool(fresh_process)}
    unique = {str(run_path(job, directory)): job for job in jobs}
    save_batch_manifests(jobs, directory)
    counts = dict(completed=0, skipped=0, failed=0, unavailable=0)
    for i, (filename, job) in enumerate(unique.items(), 1):
        path = Path(filename)
        if path.exists() and not overwrite:
            with np.load(path, allow_pickle=False) as data:
                if bool(data["completed"]):
                    counts["skipped"] += 1
                    continue
        if fresh_process:
            process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                                      json_text(job), str(directory)], capture_output=True, text=True)
            if process.returncode or not path.exists():
                effective, scenario, key = identity(job)
                record = new_record(effective["scenario"]["config"]["x_initial"])
                record["failure"] = dict(step=0, message=process.stderr.strip() or process.stdout.strip()
                    or f"Worker exited {process.returncode}", type="WorkerFailure")
                save_run(path, dict(job=job, effective=effective, scenario_key=scenario, setting_key=key,
                    fallback_count=0, fallback_used=False), record)
        else:
            execute(job, directory)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        counts[meta["status"]] += 1
        reason = "" if meta["status"] == "completed" else ": " + meta["failure"]["message"].splitlines()[-1]
        print(f"[{i}/{len(unique)}] ell={job['ell']} N={job['N']} {job['method']} {job['horizon'] or ''} "
              f"r={job['repetition']}: {meta['status']} {meta['completed_steps']}/{job['steps']}{reason}", flush=True)
    return counts


def make_batch_jobs(cases, methods, mf_values, mo_values, repetitions, steps, options=None, *,
                    batches, sweep_case, sweep_values):
    jobs = []
    for batch in batches:
        first = len(jobs)
        if batch == "pareto":
            jobs += make_jobs(cases, [m for m in methods if m != "CAMINO"], mf_values, mo_values, repetitions, steps, options)
        elif batch == "sweep":
            jobs += make_jobs([sweep_case], [m for m in methods if m in ("AS-MINMPC", "Mo", "Mf")],
                              sweep_values, sweep_values, repetitions, steps, options)
        elif batch == "camino":
            if "CAMINO" in methods:
                jobs += make_jobs(cases, ["CAMINO"], (), (), repetitions, steps, options)
        else:
            raise ValueError(f"Unknown batch: {batch}")
        for job in jobs[first:]:
            job["batch"] = batch
    if not jobs:
        raise ValueError("No benchmark runs selected.")
    return jobs


def save_batch_manifests(jobs, directory):
    """Record exact batch membership, including cached runs shared by two batches."""
    for batch in dict.fromkeys(job["batch"] for job in jobs if "batch" in job):
        selected = [job for job in jobs if job.get("batch") == batch]
        manifest = dict(batch=batch, cases=list(dict.fromkeys((j["ell"], j["N"]) for j in selected)),
                        paths=list(dict.fromkeys(run_path(j, directory).name for j in selected)))
        path = Path(directory) / f"{batch}_runs.json"
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json_text(manifest), encoding="utf-8")
        os.replace(temporary, path)


def main():
    describe(settings())
    cases = [(ell, n) for ell in ELL_VALUES for n in N_VALUES]
    jobs = make_batch_jobs(cases, METHODS, MF_VALUES, MO_VALUES, REPETITIONS, N_MPC,
        batches=SELECTED_BATCHES, sweep_case=SWEEP_CASE, sweep_values=SWEEP_VALUES)
    counts = run_campaign(jobs)
    print(counts, flush=True)
    if counts["failed"] or counts["unavailable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        execute(json.loads(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
