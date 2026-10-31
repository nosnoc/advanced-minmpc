"""Open-loop OCP for a six-state needle with insertion/rotation and spherical obstacles.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)–(39).
https://arxiv.org/pdf/2002.09658v3
"""
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from steerable_needle_model import NeedleConfig, build_problem
from steerable_needle_mpc import CaminoSolver, ThreePhaseSolver, check_stats, make_bonmin, new_record, save_records, solver_options
from steerable_needle_plotting import plot_records

DT = 0.2
N = 25
ELL = 5  # Also try 4 or 8 samples.
M_O = 10
M_F = 12
X_INITIAL = (0., 0., 0., 0., 0., 0.)
X_TARGET = (-2., 3.5, 10., 0., 0., 0.)
INITIAL_MODE = 0  # Insertion; pre-simulation dwell is already fulfilled.
INITIAL_REMAINING_DWELL = 0
RUN_THREE_PHASE = True
RUN_MO = True
RUN_MF = True
RUN_BONMIN = False
RUN_CAMINO = False
# Numerical choices, independent of the paper parameters: RK4 substeps and solver settings.
N_int_ocp = 2
N_int_sim = 4
MIP_SOLVER = "gurobi"
CAMINO_MAX_TIME = 20.0  # Seconds per OCP solve.
CAMINO_MAX_ITERATIONS = 50
CAMINO_MIP_SOLVER = "gurobi"
SOLVER_LOGS = False
SHOW_PLOTS = True
SAVE_RESULTS = False  # Write trajectories, summaries, and figures when enabled.
OUTPUT_DIRECTORY = Path(__file__).with_name("results")


def main():
    logging.basicConfig(level=logging.INFO if SOLVER_LOGS else logging.WARNING, format="%(message)s")
    c = NeedleConfig(dt=DT, ell=ELL, x_initial=X_INITIAL, x_target=X_TARGET,
        initial_mode=INITIAL_MODE, initial_remaining_dwell=INITIAL_REMAINING_DWELL,
        N_int_ocp=N_int_ocp, N_int_sim=N_int_sim)
    problem = build_problem(c, N)
    history, records = problem.history(), {}
    lower, upper = problem.bounds(c.x_initial, history)
    for label, kind, enabled, partial in (
        ("3P-MIQP", "three_phase", RUN_THREE_PHASE, {}),
        (f"3P-MIQP Mo={M_O}", "mo", RUN_MO, {"M_o": M_O}),
        (f"3P-MIQP Mf={M_F} (fractional tail)", "mf", RUN_MF, {"M_f": M_F}),
        ("Bonmin", "bonmin", RUN_BONMIN, {}),
        ("CAMINO SBMIQP", "camino", RUN_CAMINO, {})):
        if not enabled:
            continue
        r = new_record(c.x_initial, kind, **partial)
        start = perf_counter()
        try:
            if kind == "bonmin":
                solver = make_bonmin(problem, "needle_ocp_bonmin")
            elif kind == "camino":
                r["camino_settings"] = {"max_time": CAMINO_MAX_TIME,
                    "max_iterations": CAMINO_MAX_ITERATIONS, "mip_solver": CAMINO_MIP_SOLVER}
                solver = CaminoSolver("needle_ocp_camino", problem.nlp, problem.discrete,
                    **r["camino_settings"], hessian_not_psd=True)
            else:
                solver = ThreePhaseSolver("needle_ocp_"+kind, problem.nlp, problem.discrete,
                    solver_options(problem, mip_solver=MIP_SOLVER, **partial))
            r["setup_seconds"] = perf_counter()-start
            solve_start = perf_counter()
            args = dict(x0=problem.x0, lbx=lower, ubx=upper, lbg=problem.lbg, ubg=problem.ubg, p=history.previous)
            result = solver(**args) if kind == "bonmin" else solver.solve(**args)
            elapsed = perf_counter()-solve_start
            stats = {"bonmin": solver.stats()} if kind == "bonmin" else result.stats
            check_stats({"camino": stats} if kind == "camino" else stats)
            solution = result if kind == "bonmin" else result.solution
            violation = problem.check(solution["x"], c.x_initial, history, M_f=partial.get("M_f"))
            x, u, v = problem.trajectories(solution["x"])
            # Retain relaxed-tail weights, including during the independent RK4 rollout.
            sim = [np.asarray(c.x_initial)]
            for continuous, mode in zip(u, v):
                sim.append(problem.model.step(sim[-1], continuous, mode))
            r.update(states=x, continuous=u, binary=v, simulation_states=np.array(sim),
                objectives=[float(solution["f"])], preparation=[0.], feedback=[elapsed],
                prediction_violation=[violation])
        except (RuntimeError, ValueError, ImportError) as exc:
            r["failure"] = str(exc)
            print(f"{label} FAILED: {exc}", flush=True)
        r["runtime_seconds"] = perf_counter()-start
        records[label] = r
    if not records:
        raise ValueError("Enable at least one comparison.")
    directory = OUTPUT_DIRECTORY/f"ocp_ell{ELL}_N{N}"
    save_records(directory, problem, records, N, "ocp", save_results=SAVE_RESULTS)
    plot_records(problem, records, directory, "OCP", show=SHOW_PLOTS, save_results=SAVE_RESULTS)
    return records


if __name__ == "__main__":
    main()
