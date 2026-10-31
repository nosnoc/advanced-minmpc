"""Plot switch trajectories for full, fractional-tail, and fixed-tail MIQPs.

Model: J. Boldocky et al., Data Center Chiller Plant Optimization via
Mixed-Integer Nonlinear Differentiable Predictive Control (2026), Eqs. (1)-(4).
https://arxiv.org/abs/2603.28323v1
Model implementation and data: https://github.com/pnnl/MI-DPC

Run from anywhere with::

    python mpc_implementation/examples/parallel_chillers/plot_horizon_relaxation.py

The three variants solve the *same* quadratic approximation of one nonlinear
OCP, formed at one relaxed-NLP solution.  Thus only the treatment of the binary
controls changes.  ``M_f`` leaves the tail continuous; ``M_o`` fixes its tail
to a PyCombina approximation of that same relaxed solution.

Four vector PDFs, sized for a single column of a two-column paper, are written
to ``--output-dir`` (by default, this example's ``results`` directory): all
three switch controls or only v_2, each with the cases in rows or columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mpc_implementation import ThreePhaseOptions, ThreePhaseSolver

from parallel_chillers_model import ChillerConfig, build_problem, filtered_demand


# A demand pulse turns on two extra chillers.  Its fall makes the three tail
# treatments produce visibly different time trajectories.
N = 18
M_F = 8                 # A little longer than half of the horizon.
M_O = 3                 # A visibly shorter free-integer prefix.
DEMAND_RISE = 4
DEMAND_FALL = 10
RAW_DEMAND = np.r_[np.full(DEMAND_RISE, 160.0),
                   np.full(DEMAND_FALL - DEMAND_RISE, 950.0),
                   np.full(N - DEMAND_FALL, 160.0)]

COLUMN_WIDTH_IN = 3.45


def options(problem, *, m_f=None, m_o=None):
    c = problem.model.config
    return ThreePhaseOptions(
        mip_solver="gurobi", enable_miqp=True,
        enable_pycombina=m_o is not None, binary_controls_one_hot=False,
        n_x=c.n_x, n_u=2 * c.M, n_v=c.M, N=problem.N,
        M_f=m_f, M_o=m_o, binary_intervals=problem.binary_intervals,
        state_indices=problem.state_indices,
        continuous_control_indices=problem.continuous_control_indices,
        initial_state_indices=problem.state_indices[0],
        ipopt_options={"ipopt.print_level": 0, "ipopt.max_iter": 1000,
                       "ipopt.tol": 1e-7, "ipopt.bound_relax_factor": 1e-8,
                       "print_time": False},
        gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0,
                                     "Threads": 1, "Seed": 210,
                                     "MIPGap": 1e-5}},
        pycombina_options={"time_grid": np.arange(problem.N + 1) * c.dt,
                           "b_bin_pre": c.initial_switches()},
    )


def solve_switches():
    if not ca.has_conic("gurobi"):
        raise RuntimeError("This illustration requires the CasADi Gurobi plugin.")
    c = ChillerConfig(M=3, dt=180.0, permanent_on_unit=0,
                      with_ramp_limits=False, N_int_ocp=1)
    problem = build_problem(c, N)
    parameter = problem.parameters(RAW_DEMAND, filtered_demand(RAW_DEMAND),
                                   c.initial_switches())
    lbx, ubx = problem.bounds(c.initial_state(), cooling_tolerance=1e-2)
    base = ThreePhaseSolver("horizon_illustration_full", problem.nlp,
                            problem.discrete, options(problem))
    relaxed, _ = base._solve_rnlp(problem.x0, lbx, ubx, problem.lbg,
                                  problem.ubg, parameter, log_call=False)
    status = base._nlp_solver.stats()
    if not status.get("success", False):
        raise RuntimeError(f"Relaxed NLP failed: {status.get('return_status')}")
    quadratic, _, _ = base.prepare_quadratic_model(relaxed, parameter)

    schedules = {}
    pycombina_binary = None
    for name, m_f, m_o in (("full", None, None),
                           ("mf", M_F, None), ("mo", None, M_O)):
        solver = base if name == "full" else ThreePhaseSolver(
            f"horizon_illustration_{name}", problem.nlp, problem.discrete,
            options(problem, m_f=m_f, m_o=m_o))
        if name == "mo":
            pycombina_binary, _ = solver.solve_pycombina(relaxed["x"],
                                                           log_call=False)
        result, _, elapsed = solver.solve_mip(
            quadratic, lbx, ubx, problem.lbg, problem.ubg,
            pycombina_binary, relaxed["x"], log_call=False)
        stats = solver._mip_solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(f"{name} MIQP failed: {stats.get('return_status')}")
        values = np.asarray(result["x"], dtype=float).reshape(-1)
        binary = values[np.asarray(problem.binary_indices)].reshape(N, c.M)
        integer_prefix = N if m_f is None else m_f
        if np.max(np.abs(binary[:integer_prefix] -
                         np.rint(binary[:integer_prefix]))) > 1e-5:
            raise RuntimeError(f"{name}: nonintegral value in integer prefix")
        if name == "mo" and not np.allclose(binary[M_O:],
                                            pycombina_binary.reshape(N, c.M)[M_O:],
                                            atol=1e-5, rtol=0):
            raise RuntimeError("M_o fixed tail differs from PyCombina schedule")
        schedules[name] = np.clip(binary, 0.0, 1.0)
        print(f"{name:4s}: {elapsed:.2f} s, status={stats.get('return_status')}, "
              f"switches=\n{np.array2string(schedules[name].T, precision=2)}")

    if not np.any(schedules["full"][DEMAND_RISE:, 1:] > 0.5):
        raise RuntimeError("Demand did not turn on another chiller; retune RAW_DEMAND.")
    if not np.any((schedules["mf"][M_F:] > 1e-3) &
                  (schedules["mf"][M_F:] < 1 - 1e-3)):
        raise RuntimeError("M_f tail is integral; retune RAW_DEMAND for illustration.")
    return schedules


def draw_panel(ax, values, name, *, compact, component):
    dt_minutes = 180.0 / 60.0
    times = np.arange(N + 1) * dt_minutes
    end = times[-1]
    if name != "full":
        boundary = (M_F if name == "mf" else M_O) * dt_minutes
        color = "#f6e4c9" if name == "mf" else "#e5efe6"
        ax.axvspan(boundary, end, facecolor=color, edgecolor="none", zorder=0)
        ax.axvline(boundary, color="#bd7a1a" if name == "mf" else "#668468",
                   linewidth=0.85, zorder=2)
    for step in (DEMAND_RISE, DEMAND_FALL):
        ax.axvline(step * dt_minutes, color="#777777", linewidth=0.65,
                   linestyle=":", zorder=1)

    controls = (1,) if component == "single" else (0, 1, 2)
    styles = (("#164a76", "-", 1.05), ("#c45724", "-", 1.55),
              ("#16878b", (0, (3, 2)), 1.25))
    for i in controls:
        color, line_style, width = styles[i]
        ax.step(times, np.r_[values[:, i], values[-1, i]], where="post",
                color=color, linestyle=line_style, linewidth=width,
                label=rf"$v_{i+1}$", zorder=3 + i)
    ax.set_xlim(0, end)
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks((0, 0.5, 1), (r"$0$", r"$\frac{1}{2}$", r"$1$"))
    ax.set_xticks((0, 12, 24, 30, 42) if not compact else (0, 21, 42))
    ax.set_xlabel(r"$t\,[\mathrm{min}]$", labelpad=1)
    ax.set_ylabel(r"$v_2(t)$" if component == "single" else r"$v_i(t)$",
                  labelpad=1)
    ax.set_title({"full": "Full MIQP", "mf": rf"$M_f={M_F}$: relaxed tail",
                  "mo": rf"$M_o={M_O}$: fixed tail"}[name], pad=3)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.4)
    ax.tick_params(length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def save_layout(schedules, directory, layout, component):
    compact = layout == "columns"
    shape = (1, 3) if compact else (3, 1)
    size = (COLUMN_WIDTH_IN, 1.40) if compact else (COLUMN_WIDTH_IN, 3.55)
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False,
                             constrained_layout=True)
    for ax, name in zip(axes.flat, ("full", "mf", "mo")):
        draw_panel(ax, schedules[name], name, compact=compact,
                   component=component)
    if component == "all":
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.11 if compact else -0.085),
                   frameon=False, fontsize=6, handlelength=2.0)
    path = directory / f"horizon_relaxation_{component}_{layout}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    if not 0 < M_O < M_F < N:
        raise ValueError("Require 0 < M_O < M_F < N.")
    matplotlib.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                                "font.size": 7, "axes.titlesize": 7,
                                "axes.labelsize": 7, "xtick.labelsize": 6,
                                "ytick.labelsize": 7, "pdf.fonttype": 42})
    schedules = solve_switches()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for component in ("all", "single"):
        for layout in ("rows", "columns"):
            save_layout(schedules, args.output_dir, layout, component)


if __name__ == "__main__":
    main()
