"""Paper figures from saved parallel-chiller runs; no simulations are started.

Model: J. Boldocky et al., Data Center Chiller Plant Optimization via Mixed-Integer
Nonlinear Differentiable Predictive Control, 2026, Eqs. (1)-(4).
https://arxiv.org/pdf/2603.28323v1
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import plot_results as reports

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIRECTORY = HERE / "results"
FIGURES_DIRECTORY = HERE / "figures" / "paper"  # Keeps the existing report figures intact.
OUTPUTS = ("pareto", "trajectories", "sweep")  # Remove a name to skip that figure type.

SCENARIOS = ((3, 15), (3, 20))  # Two panels: (number of binary controls n_v, prediction horizon N).
METHODS = ("ThreePhase", "CAMINO", "RBC", "AS-MINMPC", "Mo", "Mf")
MIP_SOLVER = "daqp"  # Select the backend used for the saved MIQP runs.
REPETITIONS = None  # None loads all saved repetitions; (0,) selects only the first.
MATCH_SETTINGS = {"effective.scenario.schema": 4, "job.options.fresh_process": True}

PARETO_HORIZONS = None  # None displays all loaded Mo/Mf values; e.g. (2, 5, 8) selects fewer.
PARETO_LEGEND_COLUMNS = 5
PARETO_HEIGHT = 2.2  # Total figure height [inches], including the legend.
PARETO_MARKER_ALPHA = .6  # Translucent fills retain thin black outlines.
PARETO_ZOOM_COST_PERCENT = 1.0  # Linear inset within this percentage of the lowest optimizer cost; None disables it.
PARETO_BREAK_RBC_AXIS = True  # Skip the RBC-to-optimizer gap on linear time axes; log axes stay continuous.
PARETO_COST_LOG_SCALE = False
PRINT_PARETO_SUMMARY = True  # Print timing speedups and realized-cost comparisons once per scenario and statistic.
PARETO_COST_ALLOWANCE_PERCENT = 1.0  # Cost margin above the lowest measured optimizer cost for the near-best-cost comparison.

TRAJECTORY_SCENARIO = (3, 15)
TRAJECTORY_REPETITION = 0
TRAJECTORY_CONTROLLERS = (("RBC", None), ("ThreePhase", None), ("Mo", 2))  # Baselines use None; Mo/Mf use the desired horizon.
TRAJECTORY_LEGEND_COLUMNS = 4  # Legend sits between the two trajectory rows.
TRAJECTORY_HEIGHT = 2.3  # Total figure height [inches], including the middle legend.

SWEEP_SCENARIO = (3, 30)
SWEEP_VALUES = None  # None displays every saved sweep horizon; otherwise supply a tuple.
SWEEP_SHOW_95_PERCENT_BAND = False  # Shade the 2.5th-97.5th percentiles of per-step feedback times, not a confidence interval.

FIGURE_WIDTH = 3.4  # Inches; one IEEE column. Increase for a double-column figure.


def load_runs(batch, cases):
    manifest = RESULTS_DIRECTORY / f"{batch}_runs.json"
    paths = None
    if manifest.exists():
        names = json.loads(manifest.read_text(encoding="utf-8"))["paths"]
        camino = RESULTS_DIRECTORY / "camino_runs.json"
        if batch == "pareto" and "CAMINO" in METHODS and camino.exists():
            names += json.loads(camino.read_text(encoding="utf-8"))["paths"]
        names = list(dict.fromkeys(names))
        paths = [RESULTS_DIRECTORY / name for name in names if (RESULTS_DIRECTORY / name).exists()]
        if len(paths) < len(names):
            print(f"{batch}: {len(names)-len(paths)} unsaved runs skipped.", flush=True)
        if not paths:
            return []
    elif not any(RESULTS_DIRECTORY.glob("*.npz")):
        return []
    return reports.read_results(RESULTS_DIRECTORY, paths=paths, scenarios=cases, methods=METHODS,
        repetitions=REPETITIONS, match=MATCH_SETTINGS, mip_solver=MIP_SOLVER)


def finish(fig, handles, columns, name, height=None):
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 0),
        ncol=columns, frameon=False, fontsize=6, labelspacing=.35, columnspacing=.8,
        handlelength=1.3, handletextpad=.4, borderpad=.1)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    fig.set_size_inches(FIGURE_WIDTH, height if height is not None else fig.get_figheight()+legend_height+.1)
    grids = {ax.get_subplotspec().get_topmost_subplotspec().get_gridspec() for ax in fig.axes}
    for grid in grids:
        grid.update(bottom=(legend_height+.45)/fig.get_figheight())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)-legend.get_window_extent(renderer).y1
    if gap < 4:
        for grid in grids:
            grid.update(bottom=grid.get_subplot_params(fig).bottom+(4-gap)/fig.bbox.height)
    return reports.save_figure(fig, FIGURES_DIRECTORY, name)


def print_pareto_summary(runs, cases):
    def name(key):
        return reports.controller_label(*key).replace("$", "")

    print("\nPareto facts: completed runs only; costs and mean times average across runs; max time is the largest recorded step.", flush=True)
    for case in cases:
        groups = {key: g for key, g in reports.summarize(runs, case).items()
            if key[0] != "RBC" and (key[1] is None or PARETO_HORIZONS is None or key[1] in PARETO_HORIZONS)}
        excluded = sum(g["failed"] for g in groups.values())
        for statistic, field in (("mean", "feedback"), ("max", "max_feedback")):
            valid = {key: g for key, g in groups.items() if g["count"] and np.isfinite(g[field]) and g[field] > 0}
            print(f"\nn_v={case[0]}, N={case[1]} -- {statistic} feedback" +
                  (f" ({excluded} failed/incomplete runs excluded)" if excluded else ""), flush=True)
            if not valid:
                print("  No completed optimization runs.", flush=True)
                continue
            fastest = min(valid, key=lambda k: valid[k][field])
            best = min(valid, key=lambda k: valid[k]["cost"])
            fast, low = valid[fastest], valid[best]
            print(f"  Fastest optimizer: {name(fastest)}, {fast[field]:.4f} s; "
                  f"cost {fast['cost']:.6g} (normalized {fast['normalized']:.4f}).", flush=True)
            references = [key for key in (("ThreePhase", None), ("CAMINO", None), ("AS-MINMPC", None)) if key in valid]
            slowest = max(valid, key=lambda k: valid[k][field])
            if slowest not in references:
                references.append(slowest)
            for key in references:
                if key != fastest:
                    g = valid[key]
                    print(f"  Versus {name(key)}: {g[field]/fast[field]:.2f}x faster; "
                          f"realized cost {100*(fast['cost']/g['cost']-1):+.2f}%.", flush=True)
            print(f"  Lowest measured optimizer cost: {name(best)}, {low['cost']:.6g}; "
                  f"fastest costs {100*(fast['cost']/low['cost']-1):+.2f}% and is {low[field]/fast[field]:.2f}x faster.", flush=True)
            near = {key: g for key, g in valid.items()
                    if g["cost"] <= low["cost"]*(1+PARETO_COST_ALLOWANCE_PERCENT/100)}
            balanced = min(near, key=lambda k: near[k][field])
            g = valid[balanced]
            print(f"  Fastest within {PARETO_COST_ALLOWANCE_PERCENT:g}% of lowest measured optimizer cost: "
                  f"{name(balanced)}, {g[field]:.4f} s, cost {100*(g['cost']/low['cost']-1):+.2f}%.", flush=True)
            print(f"  {name(fastest)} peak/mean feedback: {fast['max_feedback']/fast['feedback']:.2f}x "
                  f"({fast['max_feedback']:.4f} / {fast['feedback']:.4f} s).", flush=True)


def pareto(runs, cases, statistic, logarithmic):
    if len(cases) != 2:
        raise ValueError("SCENARIOS must contain exactly two distinct (n_v, N) cases.")
    fig = plt.figure(figsize=(FIGURE_WIDTH, PARETO_HEIGHT))
    grid = fig.add_gridspec(1, 2, left=.17, right=.97, top=.88, wspace=.22)
    horizons = sorted({r["job"]["horizon"] for r in runs if r["job"]["horizon"] is not None})
    field = "feedback" if statistic == "mean" else "max_feedback"
    first, handles = None, {}
    for i, case in enumerate(cases):
        groups = [(key, g) for key, g in reports.summarize(runs, case).items() if g["count"]
            and (key[1] is None or PARETO_HORIZONS is None or key[1] in PARETO_HORIZONS)]
        if not groups:
            plt.close(fig)
            raise ValueError(f"No completed Pareto runs for {case}.")
        rbc = [g[field] for (method, _), g in groups if method == "RBC"]
        optimization = [g[field] for (method, _), g in groups if method != "RBC"]
        broken = (PARETO_BREAK_RBC_AXIS and not logarithmic and rbc and optimization
                  and min(optimization) > 10*max(rbc))
        if broken:
            subgrid = grid[i].subgridspec(1, 2, width_ratios=(1, 7), wspace=.15)
            low = fig.add_subplot(subgrid[0], sharey=first)
            first = first if first is not None else low
            high = fig.add_subplot(subgrid[1], sharey=first)
            axes = (low, high)
            end = max(1.5*max(rbc), .05*min(optimization))
            low.set_xlim(-.25*end, end)
            low.set_xticks([0])
            high.set_xlim(.85*min(optimization), 1.08*max(optimization))
            high.spines["left"].set_visible(False)
            high.tick_params(axis="y", left=False, labelleft=False)
            for ax, x in ((low, 1), (high, 0)):
                ax.plot([x], [0], transform=ax.transAxes, marker=[(-1, -.6), (1, .6)],
                        markersize=4, color="black", linestyle="none", clip_on=False)
        else:
            ax = fig.add_subplot(grid[i], sharey=first)
            first = first if first is not None else ax
            axes = (ax,)
            if logarithmic:
                if any(g[field] <= 0 for _, g in groups):
                    plt.close(fig)
                    raise ValueError("Logarithmic time axes require positive feedback times.")
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(numticks=6))
        for ax in axes:
            for (method, h), group in groups:
                color, marker, _ = reports.STYLE[method]
                if h is not None:
                    marker = reports.horizon_marker(h, horizons)
                ax.scatter(group[field], group["normalized"], facecolors=to_rgba(color, PARETO_MARKER_ALPHA),
                           edgecolors="black", marker=marker, s=28 if h is None else 18, linewidths=.35)
                handles[(method, h)] = Line2D([], [], color=color, marker=marker, linestyle="none",
                    markerfacecolor=to_rgba(color, PARETO_MARKER_ALPHA), markeredgecolor="black",
                    markeredgewidth=.35, markersize=3.8, label=reports.controller_label(method, h))
            if not logarithmic and ax is axes[-1]:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            ax.grid(alpha=.2)
            if i:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
        axes[-1].set_title(rf"$n_v={case[0]},\ N={case[1]}$", pad=5)
        axes[-1].set_xlabel(f"{'Mean' if statistic == 'mean' else 'Max'} feedback [s]", labelpad=3)
        if PARETO_ZOOM_COST_PERCENT is not None and optimization:
            best = min(g["normalized"] for (method, _), g in groups if method != "RBC")
            near = [(key, g) for key, g in groups if key[0] != "RBC"
                    and g["normalized"] <= best*(1+PARETO_ZOOM_COST_PERCENT/100)]
            if len(near) >= 3:
                zoom = axes[-1].inset_axes((.43, .64, .53, .3))
                for (method, h), g in near:
                    color, marker, _ = reports.STYLE[method]
                    marker = reports.horizon_marker(h, horizons) if h is not None else marker
                    zoom.scatter(g[field], g["normalized"], facecolors=to_rgba(color, PARETO_MARKER_ALPHA),
                                 edgecolors="black", marker=marker, s=22 if h is None else 15, linewidths=.35)
                zoom.xaxis.set_major_locator(MaxNLocator(nbins=2))
                zoom.xaxis.set_major_formatter(FormatStrFormatter("%g"))
                zoom.yaxis.set_major_locator(MaxNLocator(nbins=2))
                zoom.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                zoom.tick_params(labelsize=5, length=2, pad=1)
                zoom.margins(x=.2, y=.3)
                zoom.set_title(f"Within {PARETO_ZOOM_COST_PERCENT:g}%"+ (" (lin.)" if logarithmic else " cost"), fontsize=5, pad=2)
                zoom.grid(alpha=.15)
                for spine in zoom.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(.4)
    first.set_ylabel("Normalized cost", labelpad=4)
    if PARETO_COST_LOG_SCALE:
        first.set_yscale("log")
    else:
        first.yaxis.set_major_locator(MaxNLocator(nbins=3))
        first.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ordered = sorted(handles, key=lambda k: (list(reports.STYLE).index(k[0]), k[1] or 0))
    suffix = "_".join(f"nv{c[0]}_N{c[1]}" for c in cases)
    return finish(fig, [handles[k] for k in ordered], PARETO_LEGEND_COLUMNS,
                  f"pareto_{statistic}_{'log' if logarithmic else 'linear'}_{suffix}", height=PARETO_HEIGHT)


def trajectory(runs):
    selected = []
    for method, h in TRAJECTORY_CONTROLLERS:
        matches = [r for r in runs if r["case"] == TRAJECTORY_SCENARIO and r["valid"]
            and r["job"]["repetition"] == TRAJECTORY_REPETITION
            and (r["job"]["method"], r["job"]["horizon"]) == (method, h)]
        if len(matches) != 1:
            raise ValueError(f"Select one completed trajectory: {method} {h}, {TRAJECTORY_SCENARIO}, repetition {TRAJECTORY_REPETITION}.")
        selected.append(matches[0])
    if not selected:
        raise ValueError("TRAJECTORY_CONTROLLERS must not be empty.")
    reference = selected[0]
    k = len(reference["stage_cost"])
    dt = reference["meta"]["effective"]["scenario"]["config"]["dt"]
    t = np.arange(k+1)*dt/3600
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIGURE_WIDTH, TRAJECTORY_HEIGHT))
    fig.subplots_adjust(left=.18, right=.98, top=.97, hspace=.17)
    handles, seen = [], Counter()
    for r in selected:
        if (len(r["stage_cost"]) != k or r["meta"]["effective"]["scenario"]["config"]["dt"] != dt
                or not np.array_equal(r["raw_demand"][:k], reference["raw_demand"][:k])):
            plt.close(fig)
            raise ValueError("Selected trajectories must use identical demand, duration, and sampling time.")
        method, h = r["job"]["method"], r["job"]["horizon"]
        color = reports.STYLE[method][0]
        style = ("-", "--", "-.", ":")[seen[method] % 4]
        seen[method] += 1
        cooling, active = r["cooling"].sum(axis=1), r["binary"].sum(axis=1)
        line, = axes[0].step(t, np.r_[cooling, cooling[-1]], where="post", color=color,
                            linestyle=style, label=reports.controller_label(method, h))
        handles.append(line)
        axes[1].step(t, np.r_[active, active[-1]], where="post", color=color, linestyle=style)
    demand = reference["raw_demand"][:k]
    line, = axes[0].step(t, np.r_[demand, demand[-1]], where="post", color="black", linestyle="--", label="Demand", linewidth=.8)
    handles.append(line)
    axes[0].set_ylabel("Cooling [kW]")
    axes[1].set(ylabel="Active units", xlabel="Time [h]", yticks=range(1, TRAJECTORY_SCENARIO[0]+1),
                ylim=(.8, TRAJECTORY_SCENARIO[0]+.2))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.margins(x=0)
    legend = fig.legend(handles=handles, loc="center", ncol=TRAJECTORY_LEGEND_COLUMNS,
                        fontsize=6, frameon=False, columnspacing=.8, handletextpad=.4)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height/fig.bbox.height
    gap = legend_height+.04
    row_height = (.97-.18-gap)/2
    axes[0].set_position((.18, .18+row_height+gap, .8, row_height))
    axes[1].set_position((.18, .18, .8, row_height))
    legend.set_bbox_to_anchor((.58, .18+row_height+gap/2))
    return reports.save_figure(fig, FIGURES_DIRECTORY,
                  f"trajectory_nv{TRAJECTORY_SCENARIO[0]}_N{TRAJECTORY_SCENARIO[1]}_r{TRAJECTORY_REPETITION}")


def sweep(runs, logarithmic):
    summary = reports.summarize(runs, SWEEP_SCENARIO)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 1.6))
    fig.subplots_adjust(left=.18, right=.98, top=.96)
    handles = []
    for method in ("Mo", "Mf"):
        points = sorted((h, g) for (m, h), g in summary.items() if m == method and g["count"]
                        and (SWEEP_VALUES is None or h in SWEEP_VALUES))
        if not points:
            print(f"Sweep: no completed {method} runs for {SWEEP_SCENARIO}.", flush=True)
            continue
        x, mean, lower, upper = [], [], [], []
        for h, group in points:
            samples = [r["feedback"] for r in group["runs"]]
            if any(not len(s) or not np.all(np.isfinite(s)) for s in samples):
                raise ValueError(f"Invalid feedback timings for {method}={h}.")
            x.append(h)
            mean.append(np.mean([np.mean(s) for s in samples]))
            if SWEEP_SHOW_95_PERCENT_BAND:
                lo, hi = np.percentile(np.concatenate(samples), (2.5, 97.5))
                lower.append(lo)
                upper.append(hi)
        color, marker, label = reports.STYLE[method]
        line, = ax.plot(x, mean, color=color, marker=marker, markersize=3.5, label=label)
        handles.append(line)
        if SWEEP_SHOW_95_PERCENT_BAND:
            ax.fill_between(x, lower, upper, color=color, alpha=.18, linewidth=0)
    if not handles:
        plt.close(fig)
        raise ValueError(f"No completed sweep runs for {SWEEP_SCENARIO}.")
    if SWEEP_SHOW_95_PERCENT_BAND:
        handles.append(Patch(facecolor="gray", alpha=.18, label="95% of step times"))
    ax.set(xlabel=rf"Integer horizon $M_o$ / $M_f$ ($n_v={SWEEP_SCENARIO[0]},\ N={SWEEP_SCENARIO[1]}$)",
           ylabel="Feedback [s]", yscale="log" if logarithmic else "linear")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(alpha=.2)
    return finish(fig, handles, 3, f"feedback_sweep_{'log' if logarithmic else 'linear'}_nv{SWEEP_SCENARIO[0]}_N{SWEEP_SCENARIO[1]}")


def main():
    if not set(OUTPUTS) <= {"pareto", "trajectories", "sweep"}:
        raise ValueError("OUTPUTS supports pareto, trajectories, and sweep.")
    if PARETO_LEGEND_COLUMNS < 1 or TRAJECTORY_LEGEND_COLUMNS < 1:
        raise ValueError("Legend column counts must be positive.")
    if not np.isfinite(PARETO_COST_ALLOWANCE_PERCENT) or PARETO_COST_ALLOWANCE_PERCENT < 0:
        raise ValueError("PARETO_COST_ALLOWANCE_PERCENT must be finite and nonnegative.")
    reports.plot_style()
    paths, errors = [], []
    cases = tuple(dict.fromkeys((*SCENARIOS, TRAJECTORY_SCENARIO)))
    runs = load_runs("pareto", cases) if set(OUTPUTS)-{"sweep"} else []
    sweep_runs = load_runs("sweep", (SWEEP_SCENARIO,)) if "sweep" in OUTPUTS else []
    tasks = []
    if "pareto" in OUTPUTS and runs:
        if PRINT_PARETO_SUMMARY:
            print_pareto_summary(runs, SCENARIOS)
        for statistic in ("mean", "max"):
            for logarithmic in (False, True):
                tasks.append((f"Pareto {statistic} {'log' if logarithmic else 'linear'}",
                              lambda s=statistic, l=logarithmic: pareto(runs, SCENARIOS, s, l)))
    if "trajectories" in OUTPUTS and runs:
        tasks.append(("Trajectories", lambda: trajectory(runs)))
    if "sweep" in OUTPUTS and sweep_runs:
        for logarithmic in (False, True):
            tasks.append((f"Sweep {'log' if logarithmic else 'linear'}", lambda l=logarithmic: sweep(sweep_runs, l)))
    for name, data in (("Pareto/trajectory", runs), ("Sweep", sweep_runs)):
        if not data and ((name == "Sweep" and "sweep" in OUTPUTS) or (name != "Sweep" and set(OUTPUTS)-{"sweep"})):
            print(f"{name}: no saved runs; skipped.", flush=True)
    for label, task in tasks:
        try:
            paths.append(task())
            print(f"OK: {label}", flush=True)
        except Exception as error:
            errors.append(label)
            print(f"FAILED: {label}: {error}", flush=True)
    print(f"{len(paths)} PDFs in {FIGURES_DIRECTORY}; {len(errors)} plot errors.", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
