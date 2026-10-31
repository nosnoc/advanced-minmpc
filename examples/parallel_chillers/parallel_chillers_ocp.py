"""One nonlinear chiller horizon, Eqs. (1)–(4), Table 1 physical settings.

J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and J. Drgona,
Data Center Chiller Plant Optimization via Mixed-Integer Nonlinear
Differentiable Predictive Control, 2026. https://arxiv.org/html/2603.28323v1
Source configuration: pnnl/MI-DPC commit 31ce8e6dd68c5d27bb4ac202d44fe82c3c7fbb54.
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from pathlib import Path

import casadi as ca
import numpy as np

from parallel_chillers_model import ChillerConfig, build_problem, demand_record
from parallel_chillers_mpc import (
    CaminoSolver, ThreePhaseSolver, applied_control, append_interval, check_solution, check_stats,
    new_record, save_records, solver_options,
)
from parallel_chillers_plotting import plot_records

M = 3
WITH_RAMP_LIMITS = True
USE_ACTIVE_FLOW = True  # True: a=v*m reformulation; False: paper's m controls (Eqs. (1), (3c), (4e)).
LIFT_Q_EXPRESSION = False  # Introduce auxiliary cooling at shooting nodes and RK4 stages, Eq. (1a).
PERMANENT_ON_UNIT = 0
DT = 180.0
N = 80
START_STEP = 0
RUN_BONMIN = False
RUN_CAMINO = True
CAMINO_MAX_TIME = 30.0
CAMINO_MAX_ITERATIONS = 50
CAMINO_MIP_SOLVER = "gurobi"
SHOW_PLOTS = True
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
OUTPUT_DIRECTORY = Path(__file__).with_name("results")


def main():
    c = ChillerConfig(M=M, dt=DT, with_ramp_limits=WITH_RAMP_LIMITS,
                      permanent_on_unit=PERMANENT_ON_UNIT, use_active_flow=USE_ACTIVE_FLOW,
                      lift_Q_expression=LIFT_Q_EXPRESSION)
    problem = build_problem(c, N)
    raw, filtered = demand_record(c, START_STEP+N)
    raw, filtered = raw[START_STEP:], filtered[START_STEP:]
    state, previous = c.initial_state(), c.initial_switches()
    parameter = problem.parameters(raw, filtered, previous)
    records = {}
    #("3P-MIQP", "three_phase"),
    #("3P-MIQP Mo=2", "mo2")
    for label, kind in [("3P-MIQP", "three_phase")  ]+(
            [("Bonmin", "bonmin")] if RUN_BONMIN else [])+(
            [("CAMINO SBMIQP", "camino")] if RUN_CAMINO else []):
        from time import perf_counter
        start = perf_counter()
        record = new_record(state)
        try:
            if kind == "bonmin":
                if not ca.has_nlpsol("bonmin"):
                    raise RuntimeError("Bonmin plugin is unavailable.")
                solver = ca.nlpsol("chiller_ocp_bonmin", "bonmin", problem.nlp,
                                  {"discrete": problem.discrete, "print_time": False,
                                   "bonmin.bb_log_level": 0, "bonmin.nlp_log_level": 0})
                solution = solver(x0=problem.x0, lbx=problem.lbx, ubx=problem.ubx,
                                  lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                stats, timings = {"bonmin": solver.stats()}, {}
            else:
                if kind == "camino":
                    solver = CaminoSolver("chiller_ocp_camino", problem.nlp, problem.discrete,
                                          mip_solver=CAMINO_MIP_SOLVER, max_time=CAMINO_MAX_TIME,
                                          max_iterations=CAMINO_MAX_ITERATIONS, hessian_not_psd=True)
                else:
                    solver = ThreePhaseSolver("chiller_ocp_"+kind, problem.nlp, problem.discrete,
                                               solver_options(problem))
                result = solver.solve(x0=problem.x0, lbx=problem.lbx, ubx=problem.ubx,
                                      lbg=problem.lbg, ubg=problem.ubg, p=parameter)
                solution, stats, timings = result.solution, result.stats, result.timings
                if kind == "camino":
                    stats = {"camino": stats}
            elapsed = perf_counter()-start
            check_stats(stats)
            check_solution(problem, solution["x"], parameter, state, kind, check_nlp=True)
            print(f"{label}: objective={float(solution['f']):.6f}, total={elapsed:.3f}s")
            current, preceding = state.copy(), previous.copy()
            values = np.asarray(solution["x"]).reshape(-1)
            for k in range(N):
                continuous = values[problem.continuous_control_indices[k]]
                switches = values[problem.binary_indices[k]]
                if kind != "mf4" or k < 4:
                    switches = np.rint(switches)
                    if c.use_active_flow:
                        continuous = continuous.copy()
                        continuous[:M] = np.where(switches == 0, 0.0, continuous[:M])
                current, _ = append_interval(record, problem.model, current, continuous, switches,
                    raw[k], filtered[k], preceding, 0, elapsed if k == 0 else 0,
                    {"stats": stats if k == 0 else {}, "timings": timings if k == 0 else {}})
                preceding = switches
            if kind == "mf4":
                label += " (continuous tail)"
        except (RuntimeError, ValueError) as exc:
            record["failure"] = {"message": str(exc), "elapsed_seconds": perf_counter()-start}
            print(f"{label}: {exc}")
        records[label] = record
    flow_name = "activeflow" if USE_ACTIVE_FLOW else "massflow"
    if LIFT_Q_EXPRESSION:
        flow_name += "_liftedQ"
    directory = OUTPUT_DIRECTORY/f"ocp_{flow_name}_M{M}_{'ramp' if WITH_RAMP_LIMITS else 'smooth'}"
    save_records(directory, problem, records, raw, filtered, N, "open_loop",
                 camino_settings={"max_time": CAMINO_MAX_TIME, "max_iterations": CAMINO_MAX_ITERATIONS,
                                  "mip_solver": CAMINO_MIP_SOLVER}, save_results=SAVE_RESULTS)
    plot_records(problem, records, raw, directory, show=SHOW_PLOTS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
