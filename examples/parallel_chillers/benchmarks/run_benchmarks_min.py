"""Short closed-loop chiller benchmark with a synthetic load step.

Model: J. Boldocky et al., Data Center Chiller Plant Optimization via Mixed-Integer
Nonlinear Differentiable Predictive Control, 2026, Eqs. (1)-(4).
https://arxiv.org/pdf/2603.28323v1
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from pathlib import Path

import run_benchmarks as bench
import plot_results as reports

# Other physical and solver settings are inherited from run_benchmarks.py.
M_VALUES = (2, 3)  # Small chiller counts; all (m, N) pairs run in the pareto/camino batches.
N_VALUES = (10, 10)  # Short prediction horizons [control intervals].
N_MPC = 30  # Applied control moves per run; 30 moves are 90 minutes at DT=180 s.
LOAD_JUMP_STEP = 10  # Zero-based MPC step: keep the low load for the first 10 intervals, then hold the high load.
LOAD_LOW_PER_CHILLER_KW = 50.0  # Synthetic demand before the jump [kW/chiller]; total load scales with m.
LOAD_HIGH_PER_CHILLER_KW = 200.0  # Demand after the jump [kW/chiller], held through the remaining run and forecast.
REPETITIONS = 1  # Complete closed-loop runs per setting; 1 is one MPC simulation.
MF_VALUES = (1, 2, 4)  # Integer-prefix lengths with relaxed tails; each must be <= every case's N.
MO_VALUES = (1, 2, 4)  # Positive optimized integer-prefix lengths with PyCombina-fixed tails.
SWEEP_CASE = (3, 10)  # Fixed (chiller count, prediction horizon) for the feedback-time sweep.
SWEEP_VALUES = tuple(range(1, 10))  # Positive Mf/Mo sweep values; AS-MINMPC supplies the full-horizon baseline.
SELECTED_BATCHES = ("pareto","sweep", "camino")  # Same meanings as the main runner; add "camino" to include CAMINO.
#SELECTED_BATCHES = ("pareto", "sweep", "camino")  # Same meanings as the main runner; add "camino" to include CAMINO.
METHODS = ("ThreePhase", "CAMINO", "RBC", "AS-MINMPC", "Mo", "Mf")  # Available methods; CAMINO needs the camino batch.
# MIP_SOLVER = bench.MIP_SOLVER  # Gurobi; CAMINO keeps its separate default.
MIP_SOLVER = "daqp"
HESSIAN_EPSILON = bench.HESSIAN_EPSILON  # Minimum Hessian eigenvalue; inherits the main benchmark's 1e-4 default.
IPOPT_MAX_TIME = 4.0  # CPU-time limit [s] per ThreePhase/advanced-step NLP solve.
GUROBI_TIME_LIMIT = 2.0  # Time limit [s] per ThreePhase/advanced-step Gurobi solve.
GUROBI_OPTIONS = {  # ThreePhase/advanced-step only; CAMINO uses its own settings.
    "Heuristics": 0.01,  # Target 1% of runtime for feasibility heuristics (default: 5%).
    "Presolve": 1,  # Conservative presolve, still enabled.
    "PrePasses": 3,  # Limit presolve to three passes.
    "Cuts": 1,  # Moderate cut generation, still enabled.
}
CAMINO_MAX_TIME = 2.0  # Time budget [s] per CAMINO MPC solve.
CAMINO_MAX_ITERATIONS = 10  # Maximum CAMINO SBMIQP iterations per MPC solve.
OVERWRITE = False  # True reruns completed smoke runs; False reuses matching saved runs.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "smoke"  # Separate results/figures/tables for smoke data.


def main():
    options = bench.settings()
    options["gurobi_options"] = {**options["gurobi_options"], **GUROBI_OPTIONS}
    options.update(mip_solver=MIP_SOLVER, hessian_epsilon=HESSIAN_EPSILON,
                   start_step=0, ipopt_max_time=IPOPT_MAX_TIME, gurobi_time_limit=GUROBI_TIME_LIMIT,
                   camino_time=CAMINO_MAX_TIME, camino_iterations=CAMINO_MAX_ITERATIONS)
    if not 0 <= LOAD_JUMP_STEP < N_MPC:
        raise ValueError("LOAD_JUMP_STEP must lie within the N_MPC applied intervals.")
    bench.describe(options)
    print(f"Smoke check only: {N_MPC} steps, {REPETITIONS} repetitions; no full campaign.", flush=True)
    cases = [(m, n) for m in M_VALUES for n in N_VALUES]
    jobs = bench.make_batch_jobs(cases, METHODS, MF_VALUES, MO_VALUES, REPETITIONS, N_MPC, options,
                                batches=SELECTED_BATCHES, sweep_case=SWEEP_CASE, sweep_values=SWEEP_VALUES)
    for job in jobs:
        job["custom_demand"] = dict(jump_step=LOAD_JUMP_STEP, low_kw=LOAD_LOW_PER_CHILLER_KW*job["m"],
                                    high_kw=LOAD_HIGH_PER_CHILLER_KW*job["m"])
    counts = bench.run_campaign(jobs, OUTPUT_DIRECTORY / "results", overwrite=OVERWRITE)
    runs = reports.read_results(OUTPUT_DIRECTORY / "results", methods=METHODS, repetitions=None,
                               paths=[bench.run_path(job, OUTPUT_DIRECTORY / "results") for job in jobs])
    errors = []
    for run in runs:
        k, m = len(run["stage_cost"]), run["job"]["m"]
        if run["valid"]:
            n_x = m+1+(m if run["meta"]["effective"]["scenario"]["config"]["with_ramp_limits"] else 0)
            checks = (run["states"].shape == (k+1, n_x), run["continuous"].shape == (k, 2*m),
                      run["binary"].shape == (k, m), run["preparation"].shape == (k,),
                      run["feedback"].shape == (k,), len(run["cumulative_cost"]) == k+1,
                      abs(run["total_cost"]-sum(run["stage_cost"])) < 1e-7,
                      ((run["binary"] == 0) | (run["binary"] == 1)).all(),
                      (run["binary"].sum(axis=1) >= 1).all())
            if not all(checks):
                errors.append(f"Array/control check failed: {run['path']}")
    outputs = tuple(name for name in reports.OUTPUTS
                    if ("sweep" in SELECTED_BATCHES if name == "sweep" else "pareto" in SELECTED_BATCHES))
    pareto_runs = reports.read_batch_results(OUTPUT_DIRECTORY / "results", "pareto", methods=METHODS,
        repetitions=None) if "pareto" in SELECTED_BATCHES else []
    sweep_runs = reports.read_batch_results(OUTPUT_DIRECTORY / "results", "sweep", methods=METHODS,
        repetitions=None) if "sweep" in SELECTED_BATCHES else []
    paths, report_errors = reports.generate_reports(pareto_runs, OUTPUT_DIRECTORY / "figures", OUTPUT_DIRECTORY / "tables",
        cases=cases, table_cases=cases[:1], table_horizons=MO_VALUES, sweep_case=SWEEP_CASE,
        sweep_runs=sweep_runs, trajectory_case=cases[-1], controllers=(("RBC", None), ("ThreePhase", None), ("Mo", 2)), outputs=outputs)
    failed = sum(not run["valid"] for run in runs)
    print(f"Smoke {'PASS' if not (failed or errors or report_errors) else 'FAIL'}: "
          f"{len(runs)-failed}/{len(runs)} complete ({counts['skipped']} reused); {len(paths)} PDFs; "
          f"{len(errors)} validation and {len(report_errors)} report errors.")
    print(f"Runs: {OUTPUT_DIRECTORY / 'results'}\nFigures: {OUTPUT_DIRECTORY / 'figures'}\nTables: {OUTPUT_DIRECTORY / 'tables'}")
    for error in errors:
        print(error)
    if failed or errors or report_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
