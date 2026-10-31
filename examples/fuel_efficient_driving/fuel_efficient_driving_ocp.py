"""Fuel-efficient OCP: position/speed tracking with torque, braking and six gears.

S. Mallick et al., Learning-Based MPC for Fuel Efficient Control of Autonomous
Vehicles with Discrete Gear Selection (2025), equations (1)–(12), Section VI.
https://arxiv.org/pdf/2503.11359v3
Source: https://github.com/SamuelMallick/mpcrl-vehicle-gears/tree/paper-2025
"""
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from fuel_efficient_driving_model import DrivingConfig, build_problem, driving_record
from fuel_efficient_driving_mpc import ThreePhaseSolver, append_interval, check_stats, make_bonmin, new_record, save_records, solver_options
from fuel_efficient_driving_plotting import plot_records

DT = 1.
N = 15
M_O = 2
M_F = 4
REFERENCE_DURATION = 100.  # Generate the same episode and preview as the MPC script.
SEED = 10
INITIAL_STATE = None
INITIAL_GEAR = None  # Highest rpm-feasible gear; overrides are numbered 1 through 6.
INITIAL_TORQUE = None  # Feasible steady-speed torque in the initial gear.
RUN_THREE_PHASE = True
RUN_MO = True
RUN_MF = True
RUN_BONMIN = False
# Numerical choices: requested RK4 replaces Euler; fuel/tracking costs remain sampled.
N_int_ocp = 2
N_int_sim = 4
MIP_SOLVER = "gurobi"
PYCOMBINA_BINARY_THRESHOLD = 1e-6
SOLVER_LOGS = False
SHOW_PLOTS = False
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
OUTPUT_DIRECTORY = Path(__file__).with_name("results")


def main():
    logging.basicConfig(level=logging.INFO if SOLVER_LOGS else logging.WARNING, format="%(message)s")
    c = DrivingConfig(dt=DT, seed=SEED, N_int_ocp=N_int_ocp, N_int_sim=N_int_sim)
    state, reference, changes, accelerations = driving_record(c, int(round(REFERENCE_DURATION/DT)), N, INITIAL_STATE)
    problem = build_problem(c, N, state, INITIAL_GEAR, INITIAL_TORQUE)
    parameter = problem.parameters(reference[:N+1], problem.initial_input[0], problem.initial_gear)
    lower, upper = problem.bounds(state)
    records = {}
    for label, kind, enabled, partial in (
        ("3P-MIQP", "three_phase", RUN_THREE_PHASE, {}),
        (f"3P-MIQP Mo={M_O}", "mo", RUN_MO, {"M_o": M_O}),
        (f"3P-MIQP Mf={M_F} (fractional tail)", "mf", RUN_MF, {"M_f": M_F}),
        ("Bonmin", "bonmin", RUN_BONMIN, {})):
        if not enabled:
            continue
        r, start = new_record(state, kind, **partial), perf_counter()
        r["pycombina_binary_threshold"] = PYCOMBINA_BINARY_THRESHOLD
        try:
            solver = make_bonmin(problem, "driving_ocp_bonmin") if kind == "bonmin" else ThreePhaseSolver(
                "driving_ocp_"+kind, problem.nlp, problem.discrete,
                solver_options(problem, mip_solver=MIP_SOLVER,
                    binary_threshold=PYCOMBINA_BINARY_THRESHOLD, **partial))
            r["setup_seconds"] = perf_counter()-start
            solve_start = perf_counter()
            args = dict(x0=problem.x0, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=parameter)
            result = solver(**args) if kind == "bonmin" else solver.solve(**args)
            elapsed = perf_counter()-solve_start
            check_stats({"bonmin": solver.stats()} if kind == "bonmin" else result.stats)
            solution = result if kind == "bonmin" else result.solution
            residual = problem.check(solution["x"], parameter, state, M_f=partial.get("M_f"))
            x, u, v = problem.trajectories(solution["x"])
            rollout = [state.copy()]
            for k in range(N):
                # Preserve all fractional weights in the relaxed tail and its model rollout.
                append_interval(r, problem.model, x[k], u[k], v[k], x[k+1], reference[k])
                rollout.append(problem.model.step(rollout[-1], u[k], v[k]))
            r.update(rollout_states=np.array(rollout), objectives=[float(solution["f"])],
                     preparation=[0.], feedback=[elapsed], prediction_violation=[residual])
        except (RuntimeError, ValueError, ImportError) as exc:
            r["failure"] = str(exc)
            print(f"{label} FAILED: {exc}", flush=True)
        r["runtime_seconds"] = perf_counter()-start
        records[label] = r
    if not records:
        raise ValueError("Enable at least one comparison.")
    directory = OUTPUT_DIRECTORY/f"ocp_seed{SEED}_N{N}"
    save_records(directory, problem, records, reference, N, "ocp", changes, accelerations, save_results=SAVE_RESULTS)
    plot_records(problem, records, reference, directory, "OCP", show=SHOW_PLOTS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
