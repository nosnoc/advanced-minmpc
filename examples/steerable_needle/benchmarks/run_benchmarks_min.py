"""Small needle MPC benchmark with RK4, obstacle avoidance and one-hot dwell modes.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)-(39).
https://arxiv.org/pdf/2002.09658v3
"""
from pathlib import Path
import sys

import numpy as np

import run_benchmarks as bench
import plot_results as reports
from steerable_needle_model import ModeHistory, NeedleConfig, NeedleModel

ELL_VALUES = (bench.mpc.ELL,)
N_VALUES = (20,)  # Prediction horizon, distinct from the number of MPC updates below.
N_MPC = 25  # Applied MPC moves; simulated duration = N_MPC*bench.DT seconds.
REPETITIONS = 1
MF_VALUES = (2, 4, 6)
MO_VALUES = (2, 4, 6)
SWEEP_CASE = (bench.mpc.ELL, 20)
SWEEP_VALUES = (2, 4, 6)
SELECTED_BATCHES = ("pareto", "sweep", "camino")
METHODS = ("ThreePhase", "CAMINO", "AS-MINMPC", "Mo", "Mf")
MIP_SOLVER = bench.MIP_SOLVER  # DAQP for these comparisons; CAMINO keeps its separate default.
HESSIAN_EPSILON = bench.HESSIAN_EPSILON
IPOPT_MAX_TIME = bench.IPOPT_MAX_TIME
GUROBI_TIME_LIMIT = bench.GUROBI_TIME_LIMIT
CAMINO_MAX_TIME = bench.CAMINO_MAX_TIME
CAMINO_MAX_ITERATIONS = bench.CAMINO_MAX_ITERATIONS
OVERWRITE = False
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "smoke"
TINY_STEPS = 3  # --tiny runs one setting per method before the minimal campaign.


def validate_runs(runs):
    errors = []
    for run in runs:
        if not run["valid"]:
            errors.append(f"Incomplete run: {run['path']}")
            continue
        k = run["job"]["steps"]
        c = NeedleConfig(**run["meta"]["effective"]["scenario"]["config"])
        x, u, v = run["states"], run["continuous"], run["binary"]
        checks = (x.shape == (k+1, 6), u.shape == (k, 2), v.shape == (k, 2),
            run["preparation"].shape == (k,), run["feedback"].shape == (k,),
            run["remaining_dwell"].shape == (k,), run["cumulative_cost"].shape == (k+1,),
            np.all(np.isin(v, (0, 1))), np.all(v.sum(axis=1) == 1),
            np.isclose(run["total_cost"], .01*np.sum(u*u)+10*np.sum((x[-1]-c.x_target)**2), atol=1e-8),
            np.isclose(run["cumulative_cost"][-1], run["total_cost"], atol=1e-8))
        if not all(checks):
            errors.append(f"Array/cost/control check failed: {run['path']}")
            continue
        model, history = NeedleModel(c), ModeHistory(c.initial_mode, c.initial_remaining_dwell)
        try:
            for i, (continuous, mode) in enumerate(zip(u, v)):
                history.advance(mode, c.ell)
                if history.remaining != run["remaining_dwell"][i]:
                    raise ValueError("Saved dwell history does not match applied modes.")
                if not np.allclose(model.step(x[i], continuous, mode), x[i+1], rtol=0, atol=1e-10):
                    raise ValueError("Saved states do not match nonlinear plant RK4.")
            if np.min(model.clearance(x)) < -run["job"]["options"]["feasibility_tol"]:
                raise ValueError("Obstacle constraint violated by the plant at a sampling node.")
        except (RuntimeError, ValueError) as error:
            errors.append(f"{run['path']}: {error}")
    return errors


def main(*, tiny=False):
    options = bench.settings()
    options.update(mip_solver=MIP_SOLVER, hessian_epsilon=HESSIAN_EPSILON, collect_gurobi_nodes=MIP_SOLVER == "gurobi",
                   ipopt_max_time=IPOPT_MAX_TIME, gurobi_time_limit=GUROBI_TIME_LIMIT,
                   camino_time=CAMINO_MAX_TIME, camino_iterations=CAMINO_MAX_ITERATIONS)
    bench.describe(options)
    steps = TINY_STEPS if tiny else N_MPC
    mf, mo = ((2,), (2,)) if tiny else (MF_VALUES, MO_VALUES)
    sweep = (2,) if tiny else SWEEP_VALUES
    directory = OUTPUT_DIRECTORY / "tiny" if tiny else OUTPUT_DIRECTORY
    cases = [(ell, n) for ell in ELL_VALUES for n in N_VALUES]
    jobs = bench.make_batch_jobs(cases, METHODS, mf, mo, REPETITIONS, steps, options,
        batches=SELECTED_BATCHES, sweep_case=SWEEP_CASE, sweep_values=sweep)
    print(f"{'Tiny smoke' if tiny else 'Minimal benchmark'}: {steps} MPC moves, "
          f"N={N_VALUES}, ell={ELL_VALUES}; no full campaign.", flush=True)
    counts = bench.run_campaign(jobs, directory / "results", overwrite=OVERWRITE)
    runs = reports.read_results(directory / "results", methods=METHODS, repetitions=None,
        paths=[bench.run_path(job, directory / "results") for job in jobs])
    errors = validate_runs(runs)
    controllers = (("ThreePhase", None), ("AS-MINMPC", None), ("Mo", mo[0]), ("Mf", mf[-1]), ("CAMINO", None))
    controllers = tuple(c for c in controllers if c[0] in METHODS)
    outputs = tuple(name for name in reports.OUTPUTS
        if ("sweep" in SELECTED_BATCHES if name == "sweep" else "pareto" in SELECTED_BATCHES))
    pareto_runs = reports.read_batch_results(directory / "results", "pareto", methods=METHODS,
        repetitions=None) if "pareto" in SELECTED_BATCHES else []
    sweep_runs = reports.read_batch_results(directory / "results", "sweep", methods=METHODS,
        repetitions=None) if "sweep" in SELECTED_BATCHES else []
    paths, report_errors = reports.generate_reports(pareto_runs, directory / "figures", directory / "tables",
        cases=cases, table_cases=cases[:1], table_horizons=sorted(set(mf) | set(mo)), sweep_case=SWEEP_CASE,
        sweep_runs=sweep_runs, trajectory_case=cases[-1], controllers=controllers, outputs=outputs)
    failed = sum(not r["valid"] for r in runs)
    print(f"{'Tiny smoke' if tiny else 'Minimal benchmark'} "
          f"{'PASS' if not (failed or errors or report_errors) else 'FAIL'}: "
          f"{len(runs)-failed}/{len(runs)} complete ({counts['skipped']} reused); {len(paths)} PDFs; "
          f"{len(errors)} validation and {len(report_errors)} report errors.", flush=True)
    print(f"Runs: {directory / 'results'}\nFigures: {directory / 'figures'}\nTables: {directory / 'tables'}")
    for error in errors:
        print(error)
    if failed or errors or report_errors:
        raise SystemExit(1)
    return runs, paths


if __name__ == "__main__":
    main(tiny="--tiny" in sys.argv[1:])
