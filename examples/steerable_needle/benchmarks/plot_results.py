"""Reports for needle position/orientation, insertion/rotation and dwell constraints.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)-(39).
https://arxiv.org/pdf/2002.09658v3
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

# Reads saved runs only; the minimal runner supplies separate smoke folders.
RESULTS_DIRECTORY = HERE / "results"
FIGURES_DIRECTORY = HERE / "figures"
TABLES_DIRECTORY = HERE / "tables"
SCENARIOS = ((5, 20), (5, 40))  # Legacy cases (ell, N); saved batch manifests take precedence.
PARETO_MO_VALUES = (1, 2, 5, 8, 10)  # Legacy Pareto horizons; match the experiment's MO_VALUES/MF_VALUES.
PARETO_MF_VALUES = (1, 2, 5, 8, 10)
MIP_SOLVER = None  # Legacy: infer a single saved backend; set "daqp"/"gurobi" if both exist. Manifests select exact runs.
METHODS = ("ThreePhase", "CAMINO", "AS-MINMPC", "Mo", "Mf")
REPETITIONS = None
MATCH_SETTINGS = {"effective.scenario.schema": 1, "job.options.fresh_process": True}
TABLE_HORIZONS = (2, 5, 8)
PARETO_HORIZONS = None
SWEEP_SCENARIO = (5, 40)
SWEEP_VALUES = tuple(range(1, 41))  # Legacy sweep horizons; saved manifests record the actual values.
SWEEP_SHOW_BANDS = False  # True adds the percentile shading and its legend entry.
SWEEP_PERCENTILES = (5., 95.)
SWEEP_PERCENTILE_SAMPLES = "steps"
SWEEP_GUROBI_NODES = False  # DAQP comparisons do not produce Gurobi node-count files.
TRAJECTORY_SCENARIO = (5, 20)
TRAJECTORY_REPETITION = 0
TRAJECTORY_CONTROLLERS = (("ThreePhase", None), ("AS-MINMPC", None), ("Mo", 2), ("Mf", 5), ("CAMINO", None))
OUTPUTS = ("tables", "pareto", "sweep", "trajectories")
PARETO_TIME_LOG_SCALE = False  # Single-case time axis; two-case Pareto figures always save linear and logarithmic versions.
PARETO_COST_LOG_SCALE = False
TRAJECTORY_LEGEND_INSIDE = True
# Every two/three-row figure includes target-distance and the integral applied mode.
THIRD_ROWS = (None, "positions", "orientation", "controls", "clearance", "cost")
COLUMN_WIDTH = 3.4
STYLE = {"ThreePhase": ("#1f77b4", "s", "Three-phase"), "CAMINO": ("#d62728", "D", "CAMINO"),
         "AS-MINMPC": ("#9467bd", "X", "AS-MINMPC"),
         "Mo": ("#ff7f0e", "o", r"$M_o>0$"), "Mf": ("#2ca02c", "^", r"$M_f$")}


def nested(value, key):
    for part in key.split("."):
        value = value[part]
    return value


def read_results(directory, *, scenarios=None, methods=METHODS, repetitions=REPETITIONS, match=None, paths=None,
                 horizons=None, mip_solver=None):
    runs, identities = [], {}
    files = Path(directory).glob("*.npz") if paths is None else {Path(p) for p in paths}
    for path in sorted(files):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            job = meta["job"]
            case = (job["ell"], job["N"])
            if scenarios is not None and case not in scenarios or job["method"] not in methods:
                continue
            if repetitions is not None and job["repetition"] not in repetitions:
                continue
            if horizons is not None and job["method"] in horizons and job["horizon"] not in horizons[job["method"]]:
                continue
            if mip_solver is not None and job["method"] not in ("CAMINO", "RBC") and job["options"]["mip_solver"] != mip_solver:
                continue
            try:
                if any(nested(meta, key) != value for key, value in (match or {}).items()):
                    continue
            except KeyError:
                continue
            key = (*case, job["method"], job["horizon"], job["repetition"])
            if key in identities:
                raise ValueError(f"Ambiguous settings for {key}; select MATCH_SETTINGS.\n{identities[key]}\n{path}")
            identities[key] = path
            arrays = {name: data[name].copy() for name in data.files if name not in ("metadata", "step_details")}
            valid = bool(arrays["completed"]) and not bool(arrays["fallback_used"])
            valid &= bool(np.isfinite(arrays["total_cost"]))
            runs.append(dict(path=path, meta=meta, job=job, case=case, valid=valid, **arrays))
    if not runs:
        raise ValueError(f"No matching NPZ runs in {directory}")
    for case in {r["case"] for r in runs}:
        if len({r["meta"]["scenario_key"] for r in runs if r["case"] == case}) > 1:
            raise ValueError(f"Mixed physical/simulation settings for {case}; select MATCH_SETTINGS.")
        backends = {r["job"]["options"]["mip_solver"] for r in runs
                    if r["case"] == case and r["job"]["method"] not in ("CAMINO", "RBC")}
        if len(backends) > 1:
            raise ValueError(f"Mixed MIQP backends for {case}; select MIP_SOLVER or a batch manifest.")
    return runs


def read_batch_results(directory, batch, *, methods=METHODS, repetitions=REPETITIONS, match=None):
    """Load one recorded experiment; never borrow other horizons for normalization."""
    if batch not in ("pareto", "sweep"):
        raise ValueError("Report batch must be 'pareto' or 'sweep'.")
    directory = Path(directory)
    manifest_path = directory / f"{batch}_runs.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenarios = [tuple(case) for case in manifest["cases"]]
        paths = [directory / name for name in manifest["paths"]]
        camino_path = directory / "camino_runs.json"
        if batch == "pareto" and "CAMINO" in methods and camino_path.exists():
            camino = json.loads(camino_path.read_text(encoding="utf-8"))
            paths += [directory / name for name in camino["paths"]]
        return read_results(directory, scenarios=scenarios, methods=methods, repetitions=repetitions,
                            match=match, paths=paths)
    # Older campaigns have no manifests. Select each family's configured horizons
    # before duplicate detection and cost normalization, including failed attempts.
    scenarios = SCENARIOS if batch == "pareto" else (SWEEP_SCENARIO,)
    horizons = {"Mo": PARETO_MO_VALUES, "Mf": PARETO_MF_VALUES} if batch == "pareto" else {
        "Mo": SWEEP_VALUES, "Mf": SWEEP_VALUES}
    if batch == "sweep":
        methods = tuple(m for m in methods if m in ("AS-MINMPC", "Mo", "Mf"))
    print(f"No {manifest_path.name}; using explicit legacy {batch} cases/horizons.", flush=True)
    return read_results(directory, scenarios=scenarios, methods=methods, repetitions=repetitions,
                        match=match, horizons=horizons, mip_solver=MIP_SOLVER)


def summarize(runs, case):
    result = {}
    for r in runs:
        if r["case"] != case:
            continue
        key = (r["job"]["method"], r["job"]["horizon"])
        result.setdefault(key, []).append(r)
    output = {}
    for key, group in result.items():
        valid = [r for r in group if r["valid"]]
        output[key] = dict(runs=valid, count=len(valid), attempts=len(group),
            failed=sum(not r["valid"] for r in group), cost=np.mean([r["total_cost"] for r in valid]) if valid else np.nan,
            prep=np.mean([np.mean(r["preparation"]) for r in valid]) if valid else np.nan,
            feedback=np.mean([np.mean(r["feedback"]) for r in valid]) if valid else np.nan,
            max_prep=max((float(np.max(r["preparation"])) for r in valid), default=np.nan),
            max_feedback=max((float(np.max(r["feedback"])) for r in valid), default=np.nan))
    costs = [g["cost"] for g in output.values() if np.isfinite(g["cost"])]
    best = min(costs) if costs else np.nan
    if costs and best <= 0:
        raise ValueError("Cost normalization requires a positive best cost.")
    for group in output.values():
        group["normalized"] = group["cost"] / best
    return output


def plot_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0})


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".pdf")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return path


def horizon_marker(h, values):
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "8", "*")
    index = values.index(h)
    return markers[index] if index < len(markers) else (3+(index-len(markers)) % 8, 0, 15*((index-len(markers))//8+1))


def pareto(runs, cases, directory, statistic="mean", horizons=None, log_scale=None):
    if not 1 <= len(cases) <= 2 or statistic not in ("mean", "max"):
        raise ValueError("Pareto requires one or two cases and statistic='mean' or 'max'.")
    if log_scale is None:
        log_scale = PARETO_TIME_LOG_SCALE
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(COLUMN_WIDTH, 2.4))
    values = sorted({r["job"]["horizon"] for r in runs if r["job"]["horizon"] is not None})
    handles = {}
    for case_index, case in enumerate(cases):
        for (method, h), group in summarize(runs, case).items():
            if not group["count"] or h is not None and horizons is not None and h not in horizons:
                continue
            color, marker, _ = STYLE[method]
            if h is not None:
                marker = horizon_marker(h, values)
            handles[(method, h)] = Line2D([], [], color=color, marker=marker, linestyle="none",
                markersize=4, label=controller_label(method, h))
            panels = zip(("feedback", "prep"), axes) if len(cases) == 1 else (("feedback", axes[case_index]),)
            for phase, ax in panels:
                if phase == "prep" and method in ("ThreePhase", "CAMINO"):
                    continue
                key = phase if statistic == "mean" else "max_"+phase
                ax.scatter(group[key], group["normalized"], marker=marker, s=23,
                           facecolors=color, edgecolors=color, linewidths=.8)
    phases = ("feedback", "preparation") if len(cases) == 1 else ("feedback", "feedback")
    for i, (phase, ax) in enumerate(zip(phases, axes)):
        ax.set_xlabel(f"{'Mean' if statistic == 'mean' else 'Max'} {phase} [s]")
        if len(cases) == 2:
            ax.set_title(rf"$\ell={cases[i][0]},\ N={cases[i][1]}$")
        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.locator_params(axis="x", nbins=3)
        ax.grid(alpha=.2)
        ax.margins(.12)
    axes[0].set_ylabel("Normalized cost")
    if PARETO_COST_LOG_SCALE:
        axes[0].set_yscale("log")
    else:
        lower, upper = axes[0].get_ylim()
        if upper-lower < .001:
            center = (upper+lower)/2
            axes[0].set_ylim(center-.005, center+.005)
        axes[0].yaxis.set_major_locator(MaxNLocator(3))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ordered = sorted(handles, key=lambda key: (list(STYLE).index(key[0]), key[1] or 0))
    legend_items = [handles[key] for key in ordered]
    if len(cases) == 1:
        fig.suptitle(rf"$\ell={cases[0][0]},\ N={cases[0][1]}$", fontsize=8)
    legend = fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3,
                        fontsize=6.5, frameon=False, labelspacing=.3, columnspacing=.9, handletextpad=.4, borderpad=.1)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height/fig.dpi
    height = 1.75+legend_height
    fig.set_size_inches(COLUMN_WIDTH, height)
    fig.tight_layout(rect=(0, (legend_height+.015)/height, 1, .95 if len(cases) == 1 else 1), pad=.45, w_pad=.35)
    suffix = "_".join(f"ell{c[0]}_N{c[1]}" for c in cases)
    return save_figure(fig, directory, f"pareto_{statistic}{'_log' if log_scale else ''}_{suffix}")


def feedback_sweep(runs, case, directory, log_scale=False, metric="time"):
    if metric not in ("time", "nodes"):
        raise ValueError("Sweep metric must be 'time' or 'nodes'.")
    field = "feedback" if metric == "time" else "feedback_miqp_nodes"
    summary = summarize(runs, case)
    if len(SWEEP_PERCENTILES) != 2 or not 0 <= SWEEP_PERCENTILES[0] <= SWEEP_PERCENTILES[1] <= 100:
        raise ValueError("SWEEP_PERCENTILES must contain two ordered percentiles in [0, 100].")
    if SWEEP_PERCENTILE_SAMPLES not in ("steps", "run_means"):
        raise ValueError("SWEEP_PERCENTILE_SAMPLES must be 'steps' or 'run_means'.")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.85))
    for method in ("Mo", "Mf"):
        points = sorted((h, g) for (m, h), g in summary.items() if m == method and g["count"])
        if not points:
            raise ValueError(f"No completed {method} sweep for {case}.")
        x, mean, low, high = [], [], [], []
        for h, group in points:
            if any(field not in r or not np.all(np.isfinite(r[field])) for r in group["runs"]):
                raise ValueError(f"Missing {metric} data for {method}={h}, {case}; rerun with COLLECT_GUROBI_NODES=True for node counts.")
            samples = [float(np.mean(r[field])) for r in group["runs"]]
            x.append(h)
            mean.append(np.mean(samples))
            if SWEEP_SHOW_BANDS:
                band_samples = np.concatenate([r[field] for r in group["runs"]]) if SWEEP_PERCENTILE_SAMPLES == "steps" else samples
                q = np.percentile(band_samples, SWEEP_PERCENTILES)
                low.append(q[0]); high.append(q[1])
        color, marker, label = STYLE[method]
        ax.plot(x, mean, color=color, marker=marker, markersize=3.5, label=label)
        if SWEEP_SHOW_BANDS:
            ax.fill_between(x, low, high, color=color, alpha=.18, linewidth=0)
    baseline = summary.get(("AS-MINMPC", None))
    if baseline and baseline["count"]:
        samples = [r.get(field, np.array([np.nan])) for r in baseline["runs"]]
        if all(np.all(np.isfinite(s)) for s in samples):
            ax.axhline(np.mean([np.mean(s) for s in samples]), color=STYLE["AS-MINMPC"][0],
                       linestyle="--", label="AS-MINMPC", linewidth=.8)
    ax.set(xlabel=rf"Integer horizon $M_o$ / $M_f$ ($\ell={case[0]},N={case[1]}$)",
           ylabel="Feedback [s]" if metric == "time" else "MIQP nodes / step")
    ax.set_yscale("log" if log_scale else "linear")
    ax.grid(alpha=.2)
    handles, _ = ax.get_legend_handles_labels()
    if SWEEP_SHOW_BANDS:
        handles.append(Patch(facecolor="gray", alpha=.18,
            label=f"{SWEEP_PERCENTILES[0]:g}-{SWEEP_PERCENTILES[1]:g}% {'steps' if SWEEP_PERCENTILE_SAMPLES == 'steps' else 'run means'}"))
    ax.legend(handles=handles, ncol=2, framealpha=.75, edgecolor="none")
    horizons = sorted({h for method, h in summary if method in ("Mo", "Mf")})
    if len(horizons) > 8:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    else:
        ax.set_xticks(horizons)
    fig.tight_layout(pad=.55)
    name = f"feedback_sweep_{'log' if log_scale else 'linear'}" if metric == "time" else "miqp_nodes_sweep"
    return save_figure(fig, directory, f"{name}_ell{case[0]}_N{case[1]}")


def controller_label(method, h):
    return rf"$M_{'o' if method == 'Mo' else 'f'}={h}$" if h is not None else STYLE[method][2]


def selected_trajectories(runs, case, repetition, controllers):
    selected = []
    for method, h in controllers:
        matches = [r for r in runs if r["case"] == case and r["job"]["repetition"] == repetition
                   and (r["job"]["method"], r["job"]["horizon"]) == (method, h) and r["valid"]]
        if len(matches) != 1:
            raise ValueError(f"Trajectory requires one completed {method} {h}, {case}, repetition {repetition}.")
        selected.append(matches[0])
    if not selected:
        raise ValueError("Select at least one trajectory controller.")
    return selected


def trajectory(runs, case, repetition, controllers, directory, third=None):
    if third not in THIRD_ROWS:
        raise ValueError(f"Unknown needle third row: {third}")
    selected = selected_trajectories(runs, case, repetition, controllers)
    fig, axes = plt.subplots(2 if third is None else 3, 1, sharex=True,
                             figsize=(COLUMN_WIDTH, 2.7 if third is None else 3.65))
    components = []
    for run in selected:
        method, h = run["job"]["method"], run["job"]["horizon"]
        c = run["meta"]["effective"]["scenario"]["config"]
        x, u, v = run["states"], run["continuous"], run["binary"]
        t = np.arange(len(x))*c["dt"]
        color = STYLE[method][0]
        label = controller_label(method, h)
        axes[0].plot(t, np.linalg.norm(x[:, :3]-np.array(c["x_target"][:3]), axis=1), color=color, label=label)
        # The sum of one-hot binaries is always one; plot the selected mode instead.
        mode = 1+v[:, 1]
        axes[1].step(t, np.r_[mode, mode[-1]], where="post", color=color)
        if third in ("positions", "orientation"):
            offset = 0 if third == "positions" else 3
            names = (r"$p_1$", r"$p_2$", r"$p_3$") if offset == 0 else (r"$x_4$", r"$x_5$", r"$x_6$")
            for j, style in enumerate(("-", "--", "-.")):
                axes[2].plot(t, x[:, j+offset], color=color, linestyle=style)
                if run is selected[0]:
                    components.append(Line2D([], [], color="black", linestyle=style, label=names[j]))
                    if offset == 0:
                        axes[2].axhline(c["x_target"][j], color="black", linestyle=style, alpha=.3, linewidth=.6)
        elif third == "controls":
            for j, style in enumerate(("-", "--")):
                axes[2].step(t, np.r_[u[:, j], u[-1, j]], where="post", color=color, linestyle=style)
                if run is selected[0]:
                    components.append(Line2D([], [], color="black", linestyle=style,
                        label=(r"$u_1$", r"$u_2$ [rad/s]")[j]))
        elif third == "clearance":
            clearance = np.linalg.norm(x[:, None, :3]-np.array(c["centers"])[None, :, :], axis=2).min(axis=1)-c["radius"]
            axes[2].plot(t, clearance, color=color)
        elif third == "cost":
            axes[2].plot(t, run["cumulative_input_cost"], color=color)
    axes[0].set_ylabel("Target distance")
    axes[0].axhline(0, color="black", linestyle=":", linewidth=.6)
    axes[1].set(ylabel="Mode", yticks=[1, 2], yticklabels=["Insert", "Rotate"], ylim=(.85, 2.15))
    if third is not None:
        axes[2].set_ylabel({"positions": "Position", "orientation": "Angles [rad]",
            "controls": "Speed inputs", "clearance": "Obstacle clearance", "cost": "Accum. input cost"}[third])
        if third == "clearance":
            axes[2].axhline(0, color="black", linestyle=":", linewidth=.6)
        if components:
            axes[2].legend(handles=components, loc="best", ncol=3, framealpha=.75, edgecolor="none")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.margins(x=0)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(rf"$\ell={case[0]},\ N={case[1]}$")
    handles, _ = axes[0].get_legend_handles_labels()
    if TRAJECTORY_LEGEND_INSIDE:
        axes[0].legend(handles=handles, loc="best", ncol=2, framealpha=.75, edgecolor="none")
        bottom = 0
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        bottom = .18 if third is None else .13
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=.7, h_pad=.65)
    return save_figure(fig, directory, f"trajectory_{third or 'two_rows'}_ell{case[0]}_N{case[1]}_r{repetition}")


def trajectory_3d(runs, case, repetition, controllers, directory):
    selected = selected_trajectories(runs, case, repetition, controllers)
    c = selected[0]["meta"]["effective"]["scenario"]["config"]
    fig = plt.figure(figsize=(COLUMN_WIDTH, 3.25))
    ax = fig.add_subplot(111, projection="3d")
    positions = [np.array(c["x_target"][:3], ndmin=2)]
    for run in selected:
        method, h = run["job"]["method"], run["job"]["horizon"]
        x = run["states"][:, :3]
        ax.plot(*x.T, color=STYLE[method][0], label=controller_label(method, h))
        positions.append(x)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 36), np.linspace(0, np.pi, 20))
    sphere = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    for center in np.asarray(c["centers"]):
        ax.plot_surface(*(center[:, None, None]+c["radius"]*sphere), color="gray", alpha=.18, linewidth=0)
        positions.append(np.array([center-c["radius"], center+c["radius"]]))
    ax.scatter(*c["x_target"][:3], color="black", s=20, depthshade=False, label="Target")
    points = np.vstack(positions)
    middle = (points.min(axis=0)+points.max(axis=0))/2
    half = max(np.ptp(points, axis=0).max()/2, 1)*1.05
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
        setter(value-half, value+half)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel=r"$p_1$", ylabel=r"$p_2$", zlabel=r"$p_3$",
           title=rf"$\ell={case[0]},\ N={case[1]}$")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6)
    fig.subplots_adjust(left=0, right=.91, bottom=.17, top=.92)
    return save_figure(fig, directory, f"trajectory_3d_ell{case[0]}_N{case[1]}_r{repetition}")


def compile_preview(path):
    path = Path(path).resolve()
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is unavailable; install a LaTeX distribution to compile table previews.")
    auxiliary = ROOT / "tmp" / "needle_benchmarks" / "latex"
    auxiliary.mkdir(parents=True, exist_ok=True)
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={auxiliary}", str(path)]
    environment = dict(os.environ)
    environment["TEXINPUTS"] = os.pathsep.join((str(ROOT), str(path.parent), environment.get("TEXINPUTS", "")))
    result = subprocess.run(command, cwd=path.parent, env=environment, capture_output=True, text=True)
    compiled = auxiliary / (path.stem + ".pdf")
    if result.returncode or not compiled.exists():
        raise RuntimeError(f"Table compilation failed: {path}\n{result.stdout[-2500:]}\n{result.stderr[-1500:]}")
    log = auxiliary / (path.stem + ".log")
    if log.exists() and "Overfull \\hbox" in log.read_text(errors="replace"):
        raise RuntimeError(f"Table exceeds its column width: {log}")
    shutil.copyfile(compiled, path.with_suffix(".pdf"))
    return path.with_suffix(".pdf")


def latex_tables(runs, case, horizons, directory):
    summary = summarize(runs, case)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def cell(key, field):
        if field == "prep" and key[0] in ("ThreePhase", "CAMINO"):
            return r"\textemdash"
        group = summary.get(key)
        if group is None:
            return r"\textit{missing}"
        if not group["count"]:
            return r"\textit{failed}"
        value = f"{group[field]:.2f}" if field == "normalized" else f"{group[field]:.4f}"
        return value + (r"$^{\dagger}$" if field == "normalized" and group["failed"] else "")
    rows = []
    for method in ("AS-MINMPC", "ThreePhase", "CAMINO"):
        name = {"ThreePhase": "Three-phase"}.get(method, method)
        rows.append(name + " & " + " & ".join(cell((method, None), f) for f in ("normalized", "prep", "feedback")) + r" \\")
    rows.append(r"\midrule")
    for h in horizons:
        pairs = [cell(("Mo", h), f) + " / " + cell(("Mf", h), f) for f in ("normalized", "prep", "feedback")]
        rows.append(rf"$M_o=M_f={h}$ & " + " & ".join(pairs) + r" \\")
    failed = sum(g["failed"] for g in summary.values())
    counts = sorted({g["count"] for g in summary.values()})
    repetitions = ",".join(map(str, counts))
    comparison = ", ".join(STYLE[m][2] for m in STYLE if any(k[0] == m for k in summary))
    note = (rf"Paired entries: $M_o$ / $M_f$. $J/\min\bar J$: mean input-sum plus terminal target cost, "
            rf"normalized over {comparison}, including all loaded horizons in this scenario. "
            rf"Completed, finite, no-fallback runs only. Times: mean of run means, all steps. "
            rf"Completed runs/setting: {repetitions}; failed: {failed}. "
            r"\textemdash: no preparation phase.")
    if failed:
        note += r" $\dagger$: some repetitions failed."
    outputs = []
    for layout in ("single", "double"):
        width = r"\columnwidth" if layout == "single" else r"\textwidth"
        environment = "table" if layout == "single" else "table*"
        content = (rf"\begin{{{environment}}}[t]" + "\n\\centering\n\\footnotesize\n"
            + rf"\caption{{Needle benchmark: $\ell={case[0]}$, $N={case[1]}$.}}" + "\n"
            + r"\setlength{\tabcolsep}{3pt}" + "\n"
            + r"\begin{tabular}{@{}lrrr@{}}\toprule" + "\n"
            + r"Method & $J/\min\bar J$ & Prep. [s] & Feedback [s] \\ \midrule" + "\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            + rf"\begin{{minipage}}{{{width}}}\scriptsize {note}\end{{minipage}}" + "\n"
            + rf"\end{{{environment}}}" + "\n")
        name = f"table_ell{case[0]}_N{case[1]}_{layout}"
        fragment = directory / (name + ".tex")
        fragment.write_text(content, encoding="utf-8")
        preview = directory / (name + "_preview.tex")
        preview.write_text(r"\documentclass[letterpaper,10pt,conference]{ieeeconf}" + "\n"
            + r"\IEEEoverridecommandlockouts\overrideIEEEmargins" + "\n"
            + r"\usepackage{booktabs}" + "\n\\pagestyle{empty}\n\\begin{document}\n"
            + ("\\onecolumn\n" if layout == "double" else "")
            + rf"\input{{{fragment.name}}}" + "\n\\null\\clearpage\n\\end{document}\n", encoding="utf-8")
        outputs.append(compile_preview(preview))
    return outputs


def generate_reports(runs, figures, tables, *, cases=SCENARIOS, table_cases=None,
                     table_horizons=TABLE_HORIZONS, pareto_horizons=PARETO_HORIZONS,
                     sweep_case=SWEEP_SCENARIO, sweep_runs=None, trajectory_case=TRAJECTORY_SCENARIO,
                     repetition=TRAJECTORY_REPETITION, controllers=TRAJECTORY_CONTROLLERS, outputs=OUTPUTS):
    plot_style()
    sweep_runs = runs if sweep_runs is None else sweep_runs
    cases = list(dict.fromkeys(cases))
    paths, errors = [], []
    tasks = []
    if "tables" in outputs:
        for case in table_cases if table_cases is not None else cases:
            tasks.append((f"tables {case}", lambda c=case: latex_tables(runs, c, table_horizons, tables)))
    if "pareto" in outputs:
        comparisons = [(case,) for case in cases] + [tuple(cases[i:i+2]) for i in range(0, len(cases)-1, 2)]
        for comparison in comparisons:
            for statistic in ("mean", "max"):
                for logarithmic in (False, True) if len(comparison) == 2 else (PARETO_TIME_LOG_SCALE,):
                    tasks.append((f"Pareto {statistic} {'log' if logarithmic else 'linear'} {comparison}",
                        lambda c=comparison, s=statistic, l=logarithmic:
                        [pareto(runs, c, figures, s, pareto_horizons, log_scale=l)]))
    if "sweep" in outputs:
        for logarithmic in (False, True):
            tasks.append((f"feedback sweep {'log' if logarithmic else 'linear'}", lambda l=logarithmic:
                          [feedback_sweep(sweep_runs, sweep_case, figures, l)]))
        if SWEEP_GUROBI_NODES:
            tasks.append(("MIQP node sweep", lambda: [feedback_sweep(sweep_runs, sweep_case, figures, metric="nodes")]))
    if "trajectories" in outputs:
        for third in THIRD_ROWS:
            tasks.append((f"trajectory {third}", lambda t=third: [trajectory(runs, trajectory_case, repetition, controllers, figures, t)]))
        tasks.append(("3D needle path", lambda: [trajectory_3d(runs, trajectory_case, repetition, controllers, figures)]))
    for label, task in tasks:
        try:
            generated = task()
            paths.extend(generated)
            print(f"Report OK: {label}", flush=True)
        except Exception as error:
            errors.append(f"{label}: {error}")
            print(f"Report FAILED: {label}: {error}", flush=True)
    return paths, errors


def main():
    runs = read_batch_results(RESULTS_DIRECTORY, "pareto", match=MATCH_SETTINGS) if set(OUTPUTS) - {"sweep"} else []
    sweep_runs = read_batch_results(RESULTS_DIRECTORY, "sweep", match=MATCH_SETTINGS) if "sweep" in OUTPUTS else []
    cases = list(dict.fromkeys(r["case"] for r in runs))
    sweep_cases = list(dict.fromkeys(r["case"] for r in sweep_runs))
    if len(sweep_cases) > 1:
        raise ValueError("A sweep experiment must contain exactly one case.")
    paths, errors = generate_reports(runs, FIGURES_DIRECTORY, TABLES_DIRECTORY, cases=cases,
        sweep_runs=sweep_runs, sweep_case=sweep_cases[0] if sweep_cases else SWEEP_SCENARIO)
    selected = {r["path"]: r for r in runs + sweep_runs}
    failures = sum(not r["valid"] for r in selected.values())
    print(f"{len(paths)} PDFs; {failures} failed/incomplete runs; {len(errors)} report errors.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
