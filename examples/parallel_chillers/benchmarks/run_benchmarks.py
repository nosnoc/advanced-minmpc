"""Closed-loop benchmarks using the existing nonlinear parallel-chiller example.

Model: J. Boldocky et al., Data Center Chiller Plant Optimization via Mixed-Integer
Nonlinear Differentiable Predictive Control, 2026, Eqs. (1)-(4).
https://arxiv.org/pdf/2603.28323v1
Demand provenance is retained in the shared model and demand.csv.
Model implementation and data: https://github.com/pnnl/MI-DPC
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
import parallel_chillers_mpc as mpc
from parallel_chillers_model import ChillerConfig, build_problem, demand_record, filtered_demand, applied_control
from parallel_chillers_rbc import RuleBasedController
from mpc_implementation import AdvancedStepMPC, CaminoSolver, ThreePhaseSolver, prepare_mpc_initial_guess

M_VALUES = (3, 3)  # Chiller counts for the pareto and camino batches.
N_VALUES = (15, 20)  # Prediction horizons [control intervals] for those batches; all (m, N) pairs run.
MF_VALUES = (1, 2, 5, 8, 10)  # Pareto Mf horizons: first Mf intervals integer, tail relaxed; each must be <= every N.
MO_VALUES = (1, 2, 5, 8, 10)  # Positive Mo values: first Mo intervals optimized, tail PyCombina-fixed.
SWEEP_CASE = (3, 30)  # Fixed (number of chillers, prediction horizon) for the feedback-time sweep.
SWEEP_VALUES = tuple(range(1, 31, 2))  # Positive Mf/Mo sweep horizons 1 through 30; each must be <= SWEEP_CASE[1].
# Batches: pareto=selected (m,N) pairs without CAMINO; sweep=SWEEP_CASE; camino=CAMINO only.
SELECTED_BATCHES = ("pareto", "sweep", "camino")  # Select one or several batches; matching completed runs are reused.
# AS-MINMPC=full-horizon advanced-step MIQP without PyCombina; CAMINO needs the camino batch.
# METHODS = ("ThreePhase", "RBC", "AS-MINMPC", "Mo", "Mf", "CAMINO")
METHODS = ("ThreePhase", "RBC", "AS-MINMPC", "Mo", "Mf")  # Remove names to disable methods within selected batches.
N_MPC = 240  # Applied control moves; duration=N_MPC*DT/86400 days (240 moves at 180 s = 0.5 days).
REPETITIONS = 1  # Complete closed-loop runs per setting; each repetition saves its own NPZ.
OVERWRITE = False  # True reruns matching completed runs; False reuses them; failed runs are retried.
FRESH_PROCESS = True  # True isolates each run in a new Python process; execution stays sequential.
RESULTS_DIRECTORY = HERE / "results"  # One NPZ per setting/repetition, including failed runs.

DT = mpc.DT  # Sampling interval [s], inherited from the main MPC example.
DEMAND_START_STEP = 0  # Starting index in the same varying demand.csv profile; 0 matches the MPC example.
PERMANENT_ON_UNIT = mpc.PERMANENT_ON_UNIT  # Zero-based always-on unit index; None allows any unit to stay on.
USE_ACTIVE_FLOW = mpc.USE_ACTIVE_FLOW  # True uses a=v*m controls; False uses mass flow m; the existing RBC needs True.
LIFT_Q_EXPRESSION = False  # Introduce auxiliary cooling variables at shooting nodes and RK4 stages.
WITH_RAMP_LIMITS = False  # Enable per-interval cooling ramp limits and cooling-memory states.
N_INT_OCP, N_INT_SIM = 1, 4  # RK4 substeps per control interval: prediction / plant simulation.
MIP_SOLVER = "daqp"  # ThreePhase/advanced-step MIQP/QP backend for this benchmark.
GUROBI_SEED = mpc.GUROBI_SEED  # ThreePhase/advanced-step Gurobi seed, held fixed across repetitions.
GUROBI_SINGLE_THREAD = mpc.GUROBI_SINGLE_THREAD  # True forces one Gurobi thread for those methods; False keeps default threading.
COLLECT_GUROBI_NODES = MIP_SOLVER == "gurobi"  # Read native NodeCount through temporary JSON files; their I/O is included in phase timings.
PYCOMBINA_WARM_START = False  # Optional ThreePhase/Mf warm start; Mo requires PyCombina, AS-MINMPC always disables it.
SOLVE_MIQP_IN_PREPARATION = False  # Also solve a predicted MIQP for advanced-step variants; never ThreePhase.
SHIFT_INITIAL_GUESS = True  # Shift the previous final trajectory; False uses the nominal NLP guess each step.
HESSIAN_EPSILON = 1e-4  # Minimum eigenvalue for the ThreePhase/advanced-step MIQP and QP Hessians.
MIQP_SLACK_MODE = mpc.MO_MIQP_SLACK_MODE  # Shared by every benchmark MIQP; current default is ineq_all.
MIQP_SLACK_PENALTY_NORM = mpc.MO_MIQP_SLACK_PENALTY_NORM  # Shared slack penalty norm; current default is l1.
MIQP_SLACK_PENALTY = mpc.MO_MIQP_SLACK_PENALTY  # Shared positive slack weight; current default is 1e4.
IPOPT_TOL = 1e-8  # IPOPT optimality tolerance for ThreePhase/advanced-step NLP phases.
IPOPT_MAX_ITER = 1000  # Maximum iterations per IPOPT solve.
IPOPT_MAX_TIME = None  # CPU-time limit [s] per IPOPT solve; None adds no benchmark cap.
GUROBI_TIME_LIMIT = None  # Time limit [s] per ThreePhase/advanced-step Gurobi solve; None adds no cap.
# These Gurobi settings apply to ThreePhase/advanced-step; CAMINO retains its native tolerances.
GUROBI_OPTIONS = {
    "IntFeasTol": 1e-6,  # Integer-variable integrality tolerance.
    "FeasibilityTol": 1e-6,  # Linear-constraint feasibility tolerance.
    "OptimalityTol": 1e-6,  # Continuous-solver dual feasibility tolerance.
    "Method": 1,  # Continuous relaxation algorithm: 1 selects dual simplex.
    "MIPGap": 0.0,  # Relative MIP optimality-gap target.
    "MIPGapAbs": 1e-6,  # Absolute MIP optimality-gap target.
}
CAMINO_MAX_TIME = 15  # Time budget [s] for each CAMINO MPC solve, not the whole simulation.
CAMINO_MAX_ITERATIONS = mpc.CAMINO_MAX_ITERATIONS  # SBMIQP iterations per MPC solve; None adds no iteration cap.
CAMINO_MIP_SOLVER = mpc.CAMINO_MIP_SOLVER  # CAMINO's own MIP backend, independent of MIP_SOLVER.
FEASIBILITY_TOL = mpc.FEASIBILITY_TOL  # Absolute tolerance used in benchmark solution checks.
SOLVER_LOGS = False  # Enable INFO solver summaries; visible when FRESH_PROCESS=False.
SCHEMA = 4  # Wall-clock timings, warning-only initial cooling guard, and optional native Gurobi node counts.


def json_text(value):
    return json.dumps(mpc.compact_stats(value), default=mpc.json_value, sort_keys=True)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()[:16]


def settings():
    return dict(dt=DT, start_step=DEMAND_START_STEP, fresh_process=FRESH_PROCESS, permanent_on_unit=PERMANENT_ON_UNIT,
                use_active_flow=USE_ACTIVE_FLOW, lift_Q_expression=LIFT_Q_EXPRESSION,
                with_ramp_limits=WITH_RAMP_LIMITS, N_int_ocp=N_INT_OCP, N_int_sim=N_INT_SIM,
                mip_solver=MIP_SOLVER, seed=GUROBI_SEED, single_thread=GUROBI_SINGLE_THREAD,
                collect_gurobi_nodes=COLLECT_GUROBI_NODES,
                pycombina_warm_start=PYCOMBINA_WARM_START, prep_miqp=SOLVE_MIQP_IN_PREPARATION,
                shift=SHIFT_INITIAL_GUESS, hessian_epsilon=HESSIAN_EPSILON,
                miqp_slack=(MIQP_SLACK_MODE, MIQP_SLACK_PENALTY_NORM, MIQP_SLACK_PENALTY),
                ipopt_tol=IPOPT_TOL, ipopt_max_iter=IPOPT_MAX_ITER, ipopt_max_time=IPOPT_MAX_TIME,
                gurobi_options=GUROBI_OPTIONS, gurobi_time_limit=GUROBI_TIME_LIMIT,
                camino_time=CAMINO_MAX_TIME, camino_iterations=CAMINO_MAX_ITERATIONS,
                camino_backend=CAMINO_MIP_SOLVER, feasibility_tol=FEASIBILITY_TOL,
                solver_logs=SOLVER_LOGS)


def make_jobs(cases, methods, mf_values, mo_values, repetitions, steps, options=None):
    if repetitions < 1 or steps < 1:
        raise ValueError("Repetitions and N_MPC must be positive.")
    options = settings() if options is None else options
    jobs = []
    for m, n in cases:
        if m < 1 or n < 2:
            raise ValueError("Require m >= 1 and N >= 2.")
        for method in methods:
            if method not in ("ThreePhase", "CAMINO", "RBC", "AS-MINMPC", "Mo", "Mf"):
                raise ValueError(f"Unknown method: {method}")
            horizons = mo_values if method == "Mo" else mf_values if method == "Mf" else (None,)
            for h in horizons:
                if h is not None and (not isinstance(h, int) or not 1 <= h <= n):
                    raise ValueError(f"{method} horizon {h} must be in [1, {n}].")
                for rep in range(repetitions):
                    jobs.append(dict(m=m, N=n, steps=steps, method=method, horizon=h,
                                     repetition=rep, options=dict(options)))
    return jobs


def describe(options):
    print(f"Defaults: {options['mip_solver']}, seed={options['seed']}, single_thread={options['single_thread']}; "
          f"RK4={options['N_int_ocp']}/{options['N_int_sim']}; dt={options['dt']:g}s.")
    print(f"3P: all feedback. AS: prep MIQP={options['prep_miqp']}, "
          f"PyCombina warm start={options['pycombina_warm_start']}; Mo requires PyCombina in prep.")
    print(f"All MIQP slacks={options['miqp_slack']}, Hessian epsilon={options['hessian_epsilon']:g}; "
          "AS-MINMPC uses the full horizon without PyCombina.")
    print(f"Limits: IPOPT={options['ipopt_max_time']}, Gurobi={options['gurobi_time_limit']}, "
          f"CAMINO={options['camino_time']}s/call, {options['camino_iterations']} iterations "
          "(CAMINO retains its own native tolerances).", flush=True)


def identity(job):
    o, method = job["options"], job["method"]
    config = ChillerConfig(M=job["m"], dt=o["dt"], permanent_on_unit=o["permanent_on_unit"],
                          use_active_flow=o["use_active_flow"], lift_Q_expression=o["lift_Q_expression"],
                          with_ramp_limits=o["with_ramp_limits"], N_int_ocp=o["N_int_ocp"], N_int_sim=o["N_int_sim"])
    files = [EXAMPLE / name for name in ("parallel_chillers_model.py", "parallel_chillers_mpc.py",
                                       "parallel_chillers_rbc.py")]
    if "custom_demand" not in job:
        files.append(EXAMPLE / "demand.csv")
    files += list((PROJECT / "src" / "mpc_implementation").glob("*.py"))
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    scenario = dict(config=asdict(config), N=job["N"], steps=job["steps"], start_step=o["start_step"],
                    source=source, casadi=ca.__version__, numpy=np.__version__, schema=SCHEMA,
                    fresh_process=o.get("fresh_process", FRESH_PROCESS))
    if "custom_demand" in job:
        scenario["custom_demand"] = job["custom_demand"]
    keys = ["shift", "feasibility_tol"] if method != "RBC" else []
    if method == "CAMINO":
        keys += ["camino_time", "camino_iterations", "camino_backend"]
    elif method != "RBC":
        keys += ["mip_solver", "seed", "single_thread", "gurobi_options", "gurobi_time_limit",
                 "ipopt_tol", "ipopt_max_iter", "ipopt_max_time", "pycombina_warm_start", "collect_gurobi_nodes"]
        keys += ["miqp_slack", "hessian_epsilon"]
        if method in ("AS-MINMPC", "Mf", "Mo"):
            keys += ["prep_miqp"]
    algorithm = {k: o[k] for k in keys}
    if method == "AS-MINMPC":
        algorithm["pycombina_warm_start"] = False
    effective = dict(scenario=scenario, method=method, horizon=job["horizon"],
                     algorithm=algorithm)
    return effective, digest(scenario), digest(effective)


def run_path(job, directory):
    _, _, key = identity(job)
    h = "" if job["horizon"] is None else f"_{job['horizon']}"
    return Path(directory) / f"m{job['m']}_N{job['N']}_{job['method']}{h}_{key}_r{job['repetition']:03d}.npz"


def make_options(problem, job, result_file=None):
    o, method = job["options"], job["method"]
    mo = job["horizon"] if method == "Mo" else None
    mf = job["horizon"] if method == "Mf" else None
    base = mpc.solver_options(problem, M_f=mf, M_o=mo)
    ipopt = {**base.ipopt_options, "ipopt.tol": o["ipopt_tol"], "ipopt.max_iter": o["ipopt_max_iter"]}
    if o["ipopt_max_time"] is not None:
        ipopt["ipopt.max_cpu_time"] = o["ipopt_max_time"]
    gurobi = {**base.gurobi_options["gurobi"], **o["gurobi_options"], "Seed": o["seed"]}
    gurobi.pop("Threads", None)
    if o["single_thread"]:
        gurobi["Threads"] = 1
    if o["gurobi_time_limit"] is not None:
        gurobi["TimeLimit"] = o["gurobi_time_limit"]
    if result_file is not None:
        gurobi["ResultFile"] = str(result_file)
    slack = o["miqp_slack"]
    return replace(base, mip_solver=o["mip_solver"], hessian_epsilon=o["hessian_epsilon"],
                   enable_pycombina=False if method == "AS-MINMPC" else mo is not None or o["pycombina_warm_start"],
                   solve_miqp_in_preparation=o["prep_miqp"] if method in ("AS-MINMPC", "Mo", "Mf") else False,
                   shift_initial_guess_enabled=False, ipopt_options=ipopt, gurobi_options={"gurobi": gurobi},
                   miqp_slack_mode=slack[0], miqp_slack_penalty_norm=slack[1], miqp_slack_penalty=slack[2])


def gurobi_node_count(path, stats):
    if path is None:
        return np.nan
    mip = stats.get("miqp")
    if stats.get("integer_found_in_rnlp") or mip is None or mip.get("skipped", False):
        return 0.0
    if not path.exists():
        warnings.warn("Gurobi did not produce node-count data for this MIQP.", RuntimeWarning, stacklevel=2)
        return np.nan
    count = json.loads(path.read_text())["SolutionInfo"].get("NodeCount")
    if count is None:
        warnings.warn("Gurobi's result has no NodeCount; recording it as unavailable.", RuntimeWarning, stacklevel=2)
        return np.nan
    mip["node_count"] = float(count)
    return float(count)


def simulate(job):
    effective, scenario_key, setting_key = identity(job)
    metadata = dict(job=job, effective=effective, scenario_key=scenario_key, setting_key=setting_key,
                    preparation_applicable=job["method"] in ("AS-MINMPC", "Mo", "Mf"),
                    cost_convention="Actual applied-interval stage sum; no terminal cost or DT factor",
                    timing_convention="Wall seconds; housekeeping and optional node JSON I/O included; plant/result persistence excluded; one-time setup recorded separately",
                    fallback_count=0, fallback_used=False)
    o, method = job["options"], job["method"]
    state = ChillerConfig(**effective["scenario"]["config"]).initial_state()
    record = None
    node_file = None
    raw = filtered = np.empty(0)
    initialization_start = perf_counter()
    try:
        config = ChillerConfig(**effective["scenario"]["config"])
        state, previous = config.initial_state(), config.initial_switches()
        record = mpc.new_record(state)
        if "custom_demand" in job:
            load = job["custom_demand"]
            raw = np.where(np.arange(job["steps"] + job["N"]) < load["jump_step"], load["low_kw"], load["high_kw"])
            filtered = filtered_demand(raw)
        else:
            raw, filtered = demand_record(config, o["start_step"] + job["steps"] + job["N"])
            raw, filtered = raw[o["start_step"]:], filtered[o["start_step"]:]
        problem = build_problem(config, job["N"])
        model = problem.model
        mpc.FEASIBILITY_TOL = o["feasibility_tol"]
        logging.basicConfig(level=logging.INFO if o["solver_logs"] else logging.ERROR, format="%(message)s", force=True)
        controller = rbc = solver = None
        if method == "RBC":
            if not config.use_active_flow:
                raise ValueError("The existing RBC requires active-flow controls; use USE_ACTIVE_FLOW=True.")
            rbc = RuleBasedController(model)
        elif method == "CAMINO":
            solver = CaminoSolver("benchmark_camino", problem.nlp, problem.discrete,
                                  mip_solver=o["camino_backend"], max_time=o["camino_time"],
                                  max_iterations=o["camino_iterations"], hessian_not_psd=True)
            native = solver._Settings()
            metadata["native_camino_defaults"] = {k: getattr(native, k) for k in
                ("IPOPT_SETTINGS", "MIP_SETTINGS", "MIP_GAP", "TIME_LIMIT") if hasattr(native, k)}
        else:
            if o["collect_gurobi_nodes"] and o["mip_solver"] == "gurobi":
                handle, filename = tempfile.mkstemp(prefix="gurobi_nodes_", suffix=".json")
                os.close(handle)
                node_file = Path(filename)
            options = make_options(problem, job, node_file)
            metadata["effective_solver_options"] = asdict(options)
            if method == "ThreePhase":
                solver = ThreePhaseSolver("benchmark_3p", problem.nlp, problem.discrete, options)
            else:
                controller = AdvancedStepMPC("benchmark_as", problem.nlp, problem.discrete, options)
        metadata["initialization_s"] = perf_counter() - initialization_start
        last = None
        for k in range(job["steps"]):
            start, prep, details = perf_counter(), 0.0, {}
            phase, feedback_start = ("preparation" if controller is not None else "feedback"), start
            try:
                if rbc is not None:
                    u, v = rbc(state)
                    feedback = perf_counter() - start
                    details = dict(stats={"rbc": {"success": True}}, timings={})
                else:
                    if node_file is not None:
                        node_file.unlink(missing_ok=True)
                    parameter = problem.parameters(raw[k:k+job["N"]], filtered[k:k+job["N"]], previous)
                    guess = prepare_mpc_initial_guess(problem.x0, last, problem.state_indices,
                        problem.warm_start_control_indices, problem.binary_indices, state, o["shift"])
                    try:
                        lower, upper = problem.bounds(state, cooling_tolerance=o["feasibility_tol"])
                    except RuntimeError as error:
                        if str(error) != "A permanently-on unit has negative initial cooling beyond tolerance.":
                            raise
                        warnings.warn(f"Step {k}: {error} Continuing with nominal control bounds.",
                                      RuntimeWarning, stacklevel=2)
                        lower, upper = ca.DM(problem.lbx), ca.DM(problem.ubx)
                        lower[problem.state_indices[0]] = state
                        upper[problem.state_indices[0]] = state
                    if controller is not None:
                        prepared = controller.preparation_phase(guess, lower, upper, problem.lbg, problem.ubg,
                                                               predicted_state=state, previous_solution=last, p=parameter)
                        details = dict(preparation_stats=prepared.stats, timings=dict(prepared.timings))
                        mpc.check_stats(prepared.stats)
                        details["preparation_miqp_nodes"] = gurobi_node_count(node_file, prepared.stats)
                        prep = perf_counter() - start
                        phase = "feedback"
                        feedback_start = perf_counter()
                        if node_file is not None:
                            node_file.unlink(missing_ok=True)
                        result = controller.feedback_phase(state, prepared, p=parameter)
                        details["stats"] = result.stats
                        details["timings"].update(result.timings)
                        kind = "mf" if method == "Mf" else "mo" if method == "Mo" else "advanced"
                    else:
                        result = solver.solve(x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                        details = dict(stats={"camino": result.stats} if method == "CAMINO" else result.stats,
                                       timings=dict(result.timings))
                        feedback_start = start
                        kind = "camino" if method == "CAMINO" else "three_phase"
                    mpc.check_stats(details["stats"])
                    details["feedback_miqp_nodes"] = gurobi_node_count(node_file, details["stats"])
                    decision = result.solution["x"]
                    mpc.check_solution(problem, decision, parameter, state, kind,
                                       M_f=job["horizon"] if method == "Mf" else None, bounds=(lower, upper))
                    u, v = applied_control(problem, decision)
                    last = ca.DM(decision)
                    feedback = perf_counter() - feedback_start
                details["timings"].update(measured_preparation=prep, measured_feedback=feedback)
                phase = "simulation"
                state, _ = mpc.append_interval(record, model, state, u, v, raw[k], filtered[k], previous,
                                               prep, feedback, details)
                previous = v.copy()
            except Exception as error:
                elapsed = perf_counter()-start
                record["failure"] = dict(step=k, type=type(error).__name__, message=str(error),
                    phase=phase, elapsed_s=elapsed, preparation_s=elapsed if phase == "preparation" else prep,
                    feedback_s=0.0 if phase == "preparation" else feedback if phase == "simulation" else perf_counter()-feedback_start,
                    details=mpc.compact_stats(details))
                if o["mip_solver"] == "daqp" and method not in ("RBC", "CAMINO"):
                    phase_solver = controller.solver if controller is not None else solver
                    for native in (phase_solver._mip_solver, phase_solver._fixed_qp_solver):
                        if native is not None and native.name() in str(error):
                            try:
                                record["failure"]["native_solver"] = dict(name=native.name(), **mpc.compact_stats(native.stats()))
                            except RuntimeError:
                                pass
                            break
                break
        metadata["metrics"] = mpc.metrics(record, raw, config)
    except Exception as error:
        if record is None:
            record = mpc.new_record(state)
        record["failure"] = dict(step=0, type=type(error).__name__, message=str(error),
                                 phase="initialization", elapsed_s=perf_counter()-initialization_start)
    finally:
        if node_file is not None:
            node_file.unlink(missing_ok=True)
    return metadata, record, raw, filtered


def save_run(path, metadata, record, raw, filtered, messages=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    k = len(record["stage_cost"])
    arrays = {key: np.asarray(record[key], dtype=float) for key in
        ("states", "continuous", "binary", "cooling", "chiller_power", "pump_power", "stage_cost", "preparation", "feedback")}
    if record["failure"] is None and not all(np.all(np.isfinite(a)) for a in arrays.values()):
        record["failure"] = dict(step=k, type="NonfiniteResult", message="Nonfinite trajectory, cost, or timing.")
    complete = record["failure"] is None and k == metadata["job"]["steps"]
    failure = record["failure"]
    status = "completed" if complete else "unavailable" if failure and any(s in failure["message"].lower()
        for s in ("unavailable", "not found", "no module", "license", "licence", "cannot load")) else "failed"
    metadata.update(status=status, completed_steps=k, warnings=list(messages), failure=failure)
    arrays["integer_found_in_rnlp"] = np.asarray([bool(s.get("stats", {}).get("integer_found_in_rnlp", False))
                                                for s in record["steps"]])
    for key in ("preparation_miqp_nodes", "feedback_miqp_nodes"):
        arrays[key] = np.asarray([s.get(key, np.nan) for s in record["steps"]], dtype=float)
    component_keys = sorted({key for s in record["steps"] for key in s.get("timings", {})})
    components = np.array([[s.get("timings", {}).get(key, 0.0) for key in component_keys] for s in record["steps"]])
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays, cumulative_cost=np.r_[0.0, np.cumsum(arrays["stage_cost"])],
        total_cost=float(np.sum(arrays["stage_cost"])), raw_demand=raw, filtered_demand=filtered,
        completed=complete, fallback_used=False, component_names=np.asarray(component_keys, dtype=str),
        component_times=components, metadata=json_text(metadata), step_details=json_text(record["steps"]))
    os.replace(temporary, path)
    return status


def execute(job, directory):
    cache = ROOT / "tmp" / "chiller_benchmarks"
    cache.mkdir(parents=True, exist_ok=True)
    original_temp = tempfile.tempdir
    with tempfile.TemporaryDirectory(dir=cache, prefix="run_", ignore_cleanup_errors=True) as work:
        try:
            tempfile.tempdir = work
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metadata, record, raw, filtered = simulate(job)
            messages = sorted({str(w.message) for w in caught})
            return save_run(run_path(job, directory), metadata, record, raw, filtered, messages)
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
                record = mpc.new_record(ChillerConfig(**effective["scenario"]["config"]).initial_state())
                record["failure"] = dict(step=0, message=process.stderr.strip() or process.stdout.strip()
                    or f"Worker exited {process.returncode}", type="WorkerFailure")
                save_run(path, dict(job=job, effective=effective, scenario_key=scenario, setting_key=key,
                    fallback_count=0, fallback_used=False, preparation_applicable=job["method"] in ("AS-MINMPC", "Mo", "Mf")),
                    record, [], [])
        else:
            execute(job, directory)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        counts[meta["status"]] += 1
        label = f"m={job['m']} N={job['N']} {job['method']} {job['horizon'] or ''} r={job['repetition']}"
        reason = "" if meta["status"] == "completed" else ": " + meta["failure"]["message"].splitlines()[-1]
        native = (meta.get("failure") or {}).get("native_solver")
        if native is not None:
            reason += f" (DAQP status {native.get('return_status', 'unknown')})"
        print(f"[{i}/{len(unique)}] {label}: {meta['status']} {meta['completed_steps']}/{job['steps']}{reason}", flush=True)
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
        manifest = dict(batch=batch, cases=list(dict.fromkeys((j["m"], j["N"]) for j in selected)),
                        paths=list(dict.fromkeys(run_path(j, directory).name for j in selected)))
        path = Path(directory) / f"{batch}_runs.json"
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json_text(manifest), encoding="utf-8")
        os.replace(temporary, path)


def main():
    describe(settings())
    cases = [(m, n) for m in M_VALUES for n in N_VALUES]
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
