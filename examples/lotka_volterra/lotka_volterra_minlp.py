"""Three-phase MINLP solution of the Lotka-Volterra fishing problem.

Model/transcription reference: CasADi, lotka_volterra_minlp.py.
https://github.com/casadi/casadi/blob/main/docs/examples/python/lotka_volterra_minlp.py
Benchmark: mintOC, Lotka Volterra fishing problem.
https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpc_implementation import CaminoSolver, ThreePhaseOptions, ThreePhaseSolver
from lotka_volterra_model import build_lotka_volterra_problem


VARIANT = "daqp"
RUN_BONMIN = False
RUN_CAMINO = False
CAMINO_MIP_SOLVER = "gurobi"
CAMINO_MAX_TIME = 25.0
CAMINO_MAX_ITERATIONS = 50
CAMINO_HESSIAN_NOT_PSD = True


def variant_options(problem, variant: str) -> ThreePhaseOptions:
    variants = {
        "gurobi",
        "daqp",
        "milp_highs",
        "pycombina",
        "pycombina_miqp",
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}")
    milp = variant.startswith("milp_")
    mip_solver = variant.removeprefix("milp_") if milp else (
        "gurobi" if variant == "gurobi" else "daqp"
    )
    return ThreePhaseOptions(
        mip_solver=mip_solver,
        enable_miqp=variant != "pycombina" and not milp,
        enable_milp=milp,
        enable_pycombina=variant in {"pycombina", "pycombina_miqp"},
        n_x=2,
        n_u=0,
        n_v=1,
        N=problem.N,
        binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=[],
        initial_state_indices=problem.state_indices[0],
    )


def solve_bonmin(problem):
    if not ca.has_nlpsol("bonmin"):
        raise RuntimeError("Bonmin is unavailable in this CasADi installation.")
    solver = ca.nlpsol(
        "lotka_bonmin", "bonmin", problem.nlp, {"discrete": problem.discrete}
    )
    return solver(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problem = build_lotka_volterra_problem()
    bonmin_solution = solve_bonmin(problem) if RUN_BONMIN else None
    solver = ThreePhaseSolver(
        "lotka_three_phase", problem.nlp, problem.discrete, variant_options(problem, VARIANT)
    )
    three_phase = solver.solve(
        x0=problem.x0,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
    )
    camino_solution = None
    if RUN_CAMINO:
        camino_solver = CaminoSolver(
            "lotka_camino",
            problem.nlp,
            problem.discrete,
            mip_solver=CAMINO_MIP_SOLVER,
            max_time=CAMINO_MAX_TIME,
            max_iterations=CAMINO_MAX_ITERATIONS,
            hessian_not_psd=CAMINO_HESSIAN_NOT_PSD,
        )
        camino_solution = camino_solver.solve(
            x0=problem.x0,
            lbx=problem.lbx,
            ubx=problem.ubx,
            lbg=problem.lbg,
            ubg=problem.ubg,
        )

    time = np.linspace(0.0, problem.T, problem.N + 1)
    solutions = [(f"Three phase: {VARIANT}", three_phase.solution)]
    if bonmin_solution is not None:
        solutions.insert(0, ("Bonmin", bonmin_solution))
    if camino_solution is not None:
        solutions.append(("CAMINO s-b-miqp", camino_solution.solution))
    fig, axes = plt.subplots(
        1, len(solutions), figsize=(5.5 * len(solutions), 4), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, (title, solution) in zip(axes, solutions):
        states, binary = problem.trajectories(solution["x"])
        axis.plot(time, states[0], "--", label="$x_0$")
        axis.plot(time, states[1], "-", label="$x_1$")
        axis.step(time[:-1], binary, where="post", linestyle="-.", label="$v$")
        axis.set(title=title, xlabel="time")
        axis.grid(True)
        axis.legend()
    axes[0].set_ylabel("state / control")
    fig.tight_layout()
    plt.show()
    print(three_phase.timings)
    if camino_solution is not None:
        print(camino_solution.timings)
        print(camino_solution.stats["return_status"])


if __name__ == "__main__":
    main()
