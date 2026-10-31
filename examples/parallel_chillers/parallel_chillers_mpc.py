"""Closed-loop nonlinear MPC comparison, Eq. (4).

J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and J. Drgona,
Data Center Chiller Plant Optimization via Mixed-Integer Nonlinear
Differentiable Predictive Control, 2026. https://arxiv.org/pdf/2603.28323v1
Parameters: paper Table 1; data: pnnl/MI-DPC commit 31ce8e6dd68c5d27bb4ac202d44fe82c3c7fbb54.
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mpc_implementation import AdvancedStepMPC, CaminoSolver, ThreePhaseOptions, ThreePhaseSolver, prepare_mpc_initial_guess
from parallel_chillers_model import (
    ChillerConfig, ChillerProblem, COP_COEFFICIENTS, SEED, applied_control,
    build_problem, demand_record, physical_violations,
)
from parallel_chillers_rbc import RuleBasedController

M = 3 # number of chillers
USE_ACTIVE_FLOW = True  # True: a=v*m reformulation to reduce nonlinearity; False: paper's m controls (Eqs. (1), (3c), (4e)).
LIFT_Q_EXPRESSION = False  # Introduce auxiliary cooling at shooting nodes and RK4 stages, Eq. (1a).
PERMANENT_ON_UNIT = 0
WITH_RAMP_LIMITS = False

DT = 180.0
N = 15 # 15
M_O = 2
MO_MIQP_SLACK_MODE = "ineq_all"
# Mode	Softened constraints
# "none"	None 
# "eq"	General equalities
# "ineq_general"	General inequalities supplied through lbg/ubg
# "ineq_box"	Eligible continuous-variable bounds
# "ineq_all"	General inequalities and eligible box bounds
# "all"	General equalities, inequalities, and eligible box bounds
MO_MIQP_SLACK_PENALTY_NORM = "l2"
MO_MIQP_SLACK_PENALTY = 1e4
M_F = 4
MF_MIQP_SLACK_MODE = "ineq_all" # none
MF_MIQP_SLACK_PENALTY_NORM = "l2"
MF_MIQP_SLACK_PENALTY = 1e4
N_DAYS = 0.25
N_int_ocp = 1
N_int_sim = 4

RUN_THREE_PHASE = True
RUN_AS_MO = True
RUN_AS_MF = True
RUN_RBC = True
RUN_BONMIN = False
RUN_CAMINO = False

WARM_START_BONMIN = True

CAMINO_MAX_TIME = 20.0  # Seconds per MPC solve.
CAMINO_MAX_ITERATIONS = 50
CAMINO_MIP_SOLVER = "gurobi"



MIP_SOLVER = "daqp"
#MIP_SOLVER = "gurobi"
GUROBI_SEED = 210
GUROBI_SINGLE_THREAD = True  
GUROBI_OPTIONS = {  # ThreePhase/advanced-step only; CAMINO uses its own settings.
    "Heuristics": 0.01,  # Target 1% of runtime for feasibility heuristics (default: 5%).
    "Presolve": 0,  # Conservative presolve, still enabled.
    "PrePasses": 0,  # Limit presolve to three passes.
    "Cuts": 1,  # Moderate cut generation, still enabled.
}
SHOW_PLOTS = True
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
PLOT_INDIVIDUAL_CONTROLLERS = False
SOLVER_LOGS = False
FEASIBILITY_TOL = 1e-4
OUTPUT_DIRECTORY = Path(__file__).with_name("results")

SMOKE_RUN = False
SMOKE_STEPS = 30


def solver_options(
    problem: ChillerProblem, *, M_f: int | None = None, M_o: int | None = None,
    miqp_slack_mode: str = "none",
) -> ThreePhaseOptions:
    c = problem.model.config
    settings = {"OutputFlag": 0, "LogToConsole": 0, "Seed": GUROBI_SEED, "IntFeasTol": 1e-6, "OptimalityTol": 1e-6, "FeasibilityTol": 1e-6,
                "Method": 1, **GUROBI_OPTIONS}
    if GUROBI_SINGLE_THREAD:
        settings["Threads"] = 1
    return ThreePhaseOptions(
        mip_solver=MIP_SOLVER, enable_miqp=True, enable_pycombina=M_o is not None,
        binary_controls_one_hot=False, n_x=c.n_x, n_u=2*c.M, n_v=c.M, N=problem.N,
        M_o=M_o, M_f=M_f,
        miqp_slack_mode=miqp_slack_mode,
        miqp_slack_penalty_norm=MF_MIQP_SLACK_PENALTY_NORM if M_f is not None else MO_MIQP_SLACK_PENALTY_NORM,
        miqp_slack_penalty=MF_MIQP_SLACK_PENALTY if M_f is not None else MO_MIQP_SLACK_PENALTY,
        binary_intervals=problem.binary_intervals, state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        shift_initial_guess_enabled=not c.lift_Q_expression,  # Shift lifted auxiliaries explicitly in this example.
        solve_miqp_in_preparation=False, solve_pnlp_in_feedback=False,
        ipopt_options={"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 1000,
                       "ipopt.tol": 1e-7, "ipopt.bound_relax_factor": 1e-8},
        gurobi_options={"gurobi": settings},
        pycombina_options={"time_grid": np.arange(problem.N+1)*c.dt},
    )


def check_stats(stats):
    failures = [f"{phase}: {value.get('return_status', 'unknown')}"
                for phase, value in stats.items()
                if isinstance(value, dict) and "success" in value and not value["success"]]
    if failures:
        raise RuntimeError("; ".join(failures))


def compact_stats(value):
    if isinstance(value, list):
        return [compact_stats(item) for item in value]
    if isinstance(value, dict):
        result = {key: compact_stats(item) for key, item in value.items()
                  if key not in ("pool_solutions", "pool_obj_val", "iterations")}
        if isinstance(value.get("iterations"), dict):
            result["final_residuals"] = {key: item[-1] for key, item in value["iterations"].items()
                                         if key in ("inf_pr", "inf_du") and len(item)}
        return result
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def check_solution(problem, decision, parameter, state, kind, check_nlp=False, M_f=None, bounds=None):
    values = np.asarray(decision, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Solver returned nonfinite decision variables.")
    indices = np.asarray(problem.binary_indices)
    binary = values[indices[:M_f] if M_f is not None else indices]
    if np.max(np.abs(binary-np.rint(binary))) > 1e-6:
        raise RuntimeError("Reconstruction is fractional inside the integer horizon.")
    if kind in ("three_phase", "bonmin", "camino") or check_nlp:
        lower, upper = problem.bounds(state, cooling_tolerance=FEASIBILITY_TOL) if bounds is None else bounds
        residual = np.asarray(ca.Function("verify_g", [problem.nlp["x"], problem.nlp["p"]],
                                          [problem.nlp["g"]])(decision, parameter)).reshape(-1)
        violation = max(0.0, float(np.max(np.asarray(lower).reshape(-1)-values)),
                        float(np.max(values-np.asarray(upper).reshape(-1))),
                        float(np.max(np.asarray(problem.lbg).reshape(-1)-residual)),
                        float(np.max(residual-np.asarray(problem.ubg).reshape(-1))))
        if violation > FEASIBILITY_TOL:
            raise RuntimeError(f"Final NLP violates hard constraints by {violation:.3g}.")


def new_record(state, *, M_f=None, M_o=None):
    return {"states": [state.copy()], "continuous": [], "binary": [], "cooling": [],
            "chiller_power": [], "pump_power": [], "stage_cost": [], "preparation": [],
            "feedback": [], "steps": [], "failure": None, "M_f": M_f, "M_o": M_o}


def append_interval(record, model, state, continuous, switches, raw, filtered, previous,
                    preparation_time, feedback_time, details):
    outputs = model.evaluate(state, continuous, switches)
    q, chiller, pump = outputs
    next_state = model.step(state, continuous, switches, filtered)
    violations = physical_violations(model, state, continuous, switches, next_state, cooling=q)
    prediction = np.asarray(model.prediction_integrator(state,continuous,switches,filtered)).reshape(-1)
    requested = np.asarray(model.prediction_outputs(state,continuous,switches)[0]).reshape(-1)
    details = {**details, "actuator_prediction_difference_kW": float(np.max(np.abs(requested-q))),
               "thermal_prediction_difference_C": float(np.max(np.abs(prediction[:model.config.n_thermal]-next_state[:model.config.n_thermal])))}
    record["states"].append(next_state.copy())
    for key, value in (("continuous", continuous), ("binary", switches), ("cooling", q),
                       ("chiller_power", chiller), ("pump_power", pump)):
        record[key].append(value.copy())
    record["stage_cost"].append(model.stage_cost(state, continuous, switches, raw, previous, outputs=outputs))
    record["preparation"].append(preparation_time)
    record["feedback"].append(feedback_time)
    record["steps"].append({**compact_stats(details), "violations": violations})
    return next_state, violations


def run_controller(problem, raw, filtered, count, kind):
    if kind not in ("three_phase", "mo", "mf", "rbc", "bonmin", "camino"):
        raise ValueError(f"Unknown controller kind: {kind}")
    model, c = problem.model, problem.model.config
    state, previous = c.initial_state(), c.initial_switches()
    partial = {"M_f": M_F if kind == "mf" else None, "M_o": M_O if kind == "mo" else None}
    record = new_record(state, **partial)
    last_solution = None
    guess = ca.DM(problem.x0)
    solver = controller = rbc = None
    if kind == "rbc":
        rbc = RuleBasedController(model)
    elif kind == "bonmin":
        if not ca.has_nlpsol("bonmin"):
            raise RuntimeError("Bonmin plugin is unavailable.")
        solver = ca.nlpsol("chiller_bonmin", "bonmin", problem.nlp,
                           {"discrete": problem.discrete, "bonmin.bb_log_level": 0,
                            "bonmin.nlp_log_level": 0,
                            "print_time": False})
    elif kind == "camino":
        solver = CaminoSolver("chiller_camino", problem.nlp, problem.discrete,
                              mip_solver=CAMINO_MIP_SOLVER, max_time=CAMINO_MAX_TIME,
                              max_iterations=CAMINO_MAX_ITERATIONS, hessian_not_psd=True)
    elif kind == "three_phase":
        solver = ThreePhaseSolver("chiller_3p", problem.nlp, problem.discrete, solver_options(problem))
    else:
        controller = AdvancedStepMPC("chiller_as_"+kind, problem.nlp, problem.discrete,
                                     solver_options(problem, **partial, miqp_slack_mode=(
                                         MO_MIQP_SLACK_MODE if kind == "mo" else MF_MIQP_SLACK_MODE)))
    for k in range(count):
        start, prep_time = perf_counter(), 0.0
        details = {}
        try:
            parameter = problem.parameters(raw[k:k+problem.N], filtered[k:k+problem.N], previous)
            if rbc is not None:
                continuous, switches = rbc(state)
                feedback_time = perf_counter()-start
                details = {"status": "RBC applied", "timings": {"total_feedback": feedback_time}}
            elif controller is not None:
                if c.lift_Q_expression:
                    shift_start = perf_counter()
                    guess = prepare_mpc_initial_guess(
                        problem.x0, last_solution, problem.state_indices, problem.warm_start_control_indices,
                        problem.binary_indices, state, True)
                    shift_time = perf_counter()-shift_start
                lower, upper = problem.bounds(state, cooling_tolerance=FEASIBILITY_TOL)
                preparation = controller.preparation_phase(
                    guess, lower, upper, problem.lbg, problem.ubg,
                    predicted_state=state, previous_solution=last_solution, p=parameter)
                if c.lift_Q_expression:
                    preparation.timings["total_preparation"] += shift_time
                prep_time = preparation.timings["total_preparation"]
                details = {"preparation_stats": preparation.stats, "timings": preparation.timings}
                check_stats(preparation.stats)
                if preparation.pycombina_binary is not None and c.permanent_on_unit is not None:
                    py_binary = np.asarray(preparation.pycombina_binary).reshape(problem.N, c.M)
                    if not np.all(py_binary[:, c.permanent_on_unit] == 1):
                        raise RuntimeError("PyCombina switched off the permanently-on unit.")
                feedback = controller.feedback_phase(state, preparation, p=parameter)
                check_stats(feedback.stats)
                decision = feedback.solution["x"]
                check_solution(problem, decision, parameter, state, kind, M_f=controller.options.M_f)
                continuous, switches = applied_control(problem, decision)
                feedback_time = feedback.timings["total_feedback"]
                last_solution = ca.DM(decision)
                details = {"stats": feedback.stats, "preparation_stats": preparation.stats,
                           "timings": {**preparation.timings, **feedback.timings}}
            else:
                guess = prepare_mpc_initial_guess(
                    problem.x0, last_solution, problem.state_indices, problem.warm_start_control_indices,
                    problem.binary_indices, state, kind != "bonmin" or WARM_START_BONMIN)
                lower, upper = problem.bounds(state, cooling_tolerance=FEASIBILITY_TOL)
                solve_start = perf_counter()
                result = solver(x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter) if kind == "bonmin" else solver.solve(
                    x0=guess, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                feedback_time = perf_counter()-solve_start
                stats = {"bonmin": solver.stats()} if kind == "bonmin" else result.stats
                timings = {"total_feedback": feedback_time} if kind == "bonmin" else result.timings
                if kind == "camino":
                    stats = {"camino": stats}
                    timings = {**timings, "total_feedback": feedback_time}
                details = {"stats": stats, "timings": timings}
                check_stats(stats)
                decision = result["x"] if kind == "bonmin" else result.solution["x"]
                check_solution(problem, decision, parameter, state, kind)
                continuous, switches = applied_control(problem, decision)
                last_solution = ca.DM(decision)
            next_state, violations = append_interval(
                record, model, state, continuous, switches, raw[k], filtered[k], previous,
                prep_time, feedback_time, details)
            state, previous = next_state, switches
            if kind != "rbc" and max(violations.values()) > FEASIBILITY_TOL:
                logging.warning("%s step %d: simulated physical constraint violations: %s",
                                kind, k, violations)
        except (RuntimeError, ValueError) as exc:
            phase_solver = controller.solver if controller is not None else solver
            if phase_solver is not None and kind not in ("bonmin", "camino"):
                for phase, function in (("last_nlp", phase_solver._nlp_solver),
                                        ("last_miqp", phase_solver._mip_solver),
                                        ("last_fixed_qp", phase_solver._fixed_qp_solver)):
                    if function is None:
                        continue
                    try:
                        details[phase] = compact_stats(function.stats())
                    except RuntimeError:
                        pass
            record["failure"] = {"step": k, "message": str(exc), "elapsed_seconds": perf_counter()-start,
                                 "preparation_seconds": prep_time, "details": compact_stats(details)}
            failed = [f"{phase.removeprefix('last_')}: {value.get('return_status', 'unknown')}"
                      for phase, value in details.items() if phase.startswith("last_")
                      and isinstance(value, dict) and value.get("success") is False]
            print(f"{kind} failed at step {k}: {'; '.join(failed) if failed else str(exc)}", flush=True)
            break
        if (k+1) % 40 == 0:
            print(f"{kind}: {k+1}/{count} intervals", flush=True)
    return record


def metrics(record, raw, config):
    K = len(record["binary"])
    if K == 0:
        return {"completed_steps": 0, "failures": int(record["failure"] is not None)}
    binary = np.asarray(record["binary"])
    switches = np.diff(np.vstack([config.initial_switches(), binary]), axis=0)
    violations = [step["violations"] for step in record["steps"]]
    chiller = float(np.sum(record["chiller_power"])*config.dt/3600)
    pump = float(np.sum(record["pump_power"])*config.dt/3600)
    return {"completed_steps": K, "energy_kWh": chiller+pump, "chiller_kWh": chiller,
            "pump_kWh": pump, "tracking_RMSE_kW": float(np.sqrt(np.mean((np.sum(record["cooling"],axis=1)-raw[:K])**2))),
            "switches": int(np.sum(np.abs(switches))), "stage_sum_cost": float(sum(record["stage_cost"])),
            "temperature_violation_C": max(v["temperature_C"] for v in violations),
            "ramp_violation_kW": max(v["ramp_kW"] for v in violations),
            "capacity_violation_kW": max(v["capacity_kW"] for v in violations),
            "temperature_violation_intervals": sum(v["temperature_C"] > FEASIBILITY_TOL for v in violations),
            "ramp_violation_intervals": sum(v["ramp_kW"] > FEASIBILITY_TOL for v in violations),
            "max_actuator_prediction_difference_kW": max(s.get("actuator_prediction_difference_kW",0) for s in record["steps"]),
            "max_thermal_prediction_difference_C": max(s.get("thermal_prediction_difference_C",0) for s in record["steps"]),
            "failures": int(record["failure"] is not None),
            "mean_preparation_s": float(np.mean(record["preparation"])),
            "max_preparation_s": float(np.max(record["preparation"])),
            "mean_feedback_s": float(np.mean(record["feedback"])),
            "max_feedback_s": float(np.max(record["feedback"]))}


def json_value(value):
    if isinstance(value, (np.ndarray, ca.DM)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def print_summary(summary):
    headings = ["Algorithm", "Steps", "kWh", "RMSE[kW]", "Switches", "Prep[s]", "Feedback[s]", "Fail"]
    rows = [[label, str(m["completed_steps"]), f"{m.get('energy_kWh', float('nan')):.3f}",
             f"{m.get('tracking_RMSE_kW', float('nan')):.3f}", str(m.get("switches", 0)),
             f"{m.get('mean_preparation_s', 0):.4f}/{m.get('max_preparation_s', 0):.4f}",
             f"{m.get('mean_feedback_s', 0):.4f}/{m.get('max_feedback_s', 0):.4f}", str(m["failures"])]
            for label, m in summary.items()]
    print("Preparation/feedback: mean/max seconds.")
    print(ThreePhaseSolver._format_table(rows, headings, left_aligned=(0,)))


def save_records(directory, problem, records, raw, filtered, count, mode, *, save_results=False, print_table=True, camino_settings=None):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, record in records.items():
        name = label.lower().replace(" ", "_").replace("=", "_").replace("-", "_")
        arrays = {key: np.asarray(record[key]) for key in (
            "states", "continuous", "binary", "cooling", "chiller_power", "pump_power",
            "stage_cost", "preparation", "feedback")}
        if save_results:
            np.savez_compressed(directory/(name+".npz"), **arrays, raw_demand=raw,
                                filtered_demand=filtered)
        summary[label] = metrics(record, raw, problem.model.config)
        if save_results:
            (directory/(name+"_solver.json")).write_text(
                json.dumps(compact_stats({"steps": record["steps"], "failure": record["failure"]}),
                           default=json_value, indent=2), encoding="utf-8")
    metadata = {"config": asdict(problem.model.config), "N": problem.N, "requested_steps": count,
                "mode": mode, "seed": SEED,
                "thermal_state_dimension": problem.model.config.n_thermal,
                "shooting_state_dimension": problem.model.config.n_x, "mip_solver": MIP_SOLVER,
                "gurobi_seed": GUROBI_SEED, "single_thread": GUROBI_SINGLE_THREAD,
                "camino_settings": camino_settings if camino_settings is not None else {
                    "max_time": CAMINO_MAX_TIME, "max_iterations": CAMINO_MAX_ITERATIONS,
                    "mip_solver": CAMINO_MIP_SOLVER},
                "capacity_per_unit_kW": problem.model.config.capacity,
                "control_parameterization": "active_flow=a=v*m; evaporator_temperature",
                "pump_coefficient": problem.model.config.gamma,
                "flow_bounds_kg_s": [problem.model.config.flow_min, problem.model.config.flow_max],
                "active_flow_bounds_kg_s": [0.0, problem.model.config.flow_max],
                "supply_evaporator_bounds_C": [problem.model.config.temperature_min, problem.model.config.temperature_max],
                "return_bounds_C": [problem.model.config.temperature_min, problem.model.config.return_max],
                "COP_coefficients": COP_COEFFICIENTS,
                "solver_settings": {"gurobi": solver_options(problem).gurobi_options,
                                    "ipopt": solver_options(problem).ipopt_options},
                "partial_horizons": {label: {key: record.get(key) for key in ("M_f", "M_o")}
                                     for label, record in records.items()},
                "metrics": summary}
    if save_results:
        (directory/"summary.json").write_text(json.dumps(metadata, default=json_value, indent=2), encoding="utf-8")
    if print_table:
        print_summary(summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO if SOLVER_LOGS else logging.WARNING, format="%(message)s")
    count = SMOKE_STEPS if SMOKE_RUN else int(N_DAYS*86400/DT)
    config = ChillerConfig(M=M, dt=DT, with_ramp_limits=WITH_RAMP_LIMITS,
                           permanent_on_unit=PERMANENT_ON_UNIT, N_int_ocp=N_int_ocp,
                           N_int_sim=N_int_sim, use_active_flow=USE_ACTIVE_FLOW, lift_Q_expression=LIFT_Q_EXPRESSION)
    enabled = [("3P-MIQP", "three_phase", RUN_THREE_PHASE), (f"AS-MIQP Mo={M_O}", "mo", RUN_AS_MO),
               (f"AS-MIQP Mf={M_F}", "mf", RUN_AS_MF), ("RBC", "rbc", RUN_RBC),
               ("Bonmin", "bonmin", RUN_BONMIN), ("CAMINO SBMIQP", "camino", RUN_CAMINO)]
    if not any(flag for _, _, flag in enabled):
        raise ValueError("Enable at least one controller.")
    if RUN_AS_MO and importlib.util.find_spec("pycombina") is None:
        raise RuntimeError("The M_o controller requires PyCombina; install it in this interpreter.")
    if any(flag and kind not in ("rbc", "bonmin", "camino") for _, kind, flag in enabled) and not ca.has_conic(MIP_SOLVER):
        raise RuntimeError(f"CasADi {MIP_SOLVER} conic plugin is unavailable.")
    problem = build_problem(config, N)
    raw, filtered = demand_record(config, count+N)
    flow_name = "activeflow" if USE_ACTIVE_FLOW else "massflow"
    if LIFT_Q_EXPRESSION:
        flow_name += "_liftedQ"
    directory = OUTPUT_DIRECTORY/f"table1_{flow_name}_M{M}_{'ramp' if WITH_RAMP_LIMITS else 'smooth'}_DT{DT:g}s_{count}steps"
    records = {}
    for label, kind, flag in enabled:
        if flag:
            print(f"Running {label}: M={M}, N={N}, DT={DT:g}s, {count} intervals", flush=True)
            records[label] = run_controller(problem, raw, filtered, count, kind)
            summary = save_records(directory, problem, records, raw, filtered, count, "closed_loop", print_table=False, save_results=SAVE_RESULTS)
    print_summary(summary)
    from parallel_chillers_plotting import plot_records
    plot_records(problem, records, raw, directory, show=SHOW_PLOTS, plot_individual=PLOT_INDIVIDUAL_CONTROLLERS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
