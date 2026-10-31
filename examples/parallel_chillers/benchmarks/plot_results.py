"""Reports from saved parallel-chiller runs; importing this file runs no experiment.

Model: J. Boldocky et al., Data Center Chiller Plant Optimization via Mixed-Integer
Nonlinear Differentiable Predictive Control, 2026, Eqs. (1)-(4).
https://arxiv.org/pdf/2603.28323v1
Model implementation and data: https://github.com/pnnl/MI-DPC
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
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

# Input/output folders; this script only reads saved runs and never starts a simulation.
RESULTS_DIRECTORY = HERE / "results"  # For smoke data, use HERE / "smoke" / "results".
FIGURES_DIRECTORY = HERE / "figures"  # Destination for vector PDF plots.
TABLES_DIRECTORY = HERE / "tables"  # Destination for LaTeX tables and compiled previews.
# The smoke runner supplies its own folders, cases, and controller selection; style flags still apply.

# Select saved data: (m, N) means (number of chillers, prediction horizon in control intervals).
SCENARIOS = ((3, 15), (3, 20))  # Legacy Pareto/table cases; saved batch manifests take precedence.
PARETO_MO_VALUES = (1, 2, 5, 8, 10)  # Legacy Pareto horizons; match the experiment's MO_VALUES/MF_VALUES.
PARETO_MF_VALUES = (1, 2, 5, 8, 10)
MIP_SOLVER = "gurobi"  # Legacy result selection; CAMINO keeps its own backend.
# AS-MINMPC is the full MIQP without PyCombina; Mo/Mf are partial-horizon advanced-step families.
METHODS = ("ThreePhase", "CAMINO", "RBC", "AS-MINMPC", "Mo", "Mf")  # Saved methods to load; adding CAMINO never runs it.
REPETITIONS = None  # None=all saved repetitions; (0,)=first only; (0, 1, 2)=first three.
MATCH_SETTINGS = {"effective.scenario.schema": 4, "job.options.fresh_process": True}  # Current timing/guard/node-recording policy; match the runner's process mode.
# Normalized cost = method's mean completed-run cost / best such mean, over all loaded methods/horizons in each case.

# Select displayed horizons; these display filters do not change the normalization comparison set.
TABLE_HORIZONS = (2, 5, 8)  # Mo/Mf values paired in table rows; these are integer-prefix lengths, not N.
PARETO_HORIZONS = None  # None=all loaded Mo/Mf values; e.g. (2, 4, 8) displays only those values.
SWEEP_SCENARIO = (3, 30)  # Choose one loaded (m, N) case for feedback time versus Mo/Mf.
SWEEP_VALUES = tuple(range(1, 31, 2))  # Legacy sweep horizons; saved manifests record the actual values.
SWEEP_SHOW_BANDS = False  # True adds the percentile shading and its legend entry.
SWEEP_PERCENTILES = (5.0, 95.0)  # Lower/upper timing percentiles shown as shaded bands.
SWEEP_PERCENTILE_SAMPLES = "steps"  # Per-step variability within completed MPC runs; "run_means" instead compares repetitions.
SWEEP_GUROBI_NODES = True  # Also plot mean feedback-MIQP node counts; requires runs saved with node collection enabled.

# Trajectories show one saved repetition, rather than averages across repetitions.
TRAJECTORY_SCENARIO = (3, 15)  # Saved (m, N) case for state/control plots; it may differ from the Pareto cases.
TRAJECTORY_REPETITION = 0  # Saved run index within that case; 0 is its first repetition.
# Entries are (method, integer horizon): ("Mo", 2) means Mo=2; baselines use None.
TRAJECTORY_CONTROLLERS = (("RBC", None), ("ThreePhase", None), ("Mo", 2))  # Every chosen method must also be included in METHODS.

# Single-case Pareto: feedback/preparation; adjacent case pairs: feedback for one case in each panel.
OUTPUTS = ("tables", "pareto", "sweep", "trajectories")  # Remove a name to skip it; Pareto saves mean/max, sweep linear/log.
PARETO_TIME_LOG_SCALE = False  # Single-case time axis; two-case Pareto figures always save linear and logarithmic versions.
PARETO_COST_LOG_SCALE = False  # True=logarithmic cost axis; costs remain normalized with either choice.
INCLUDE_RBC_IN_TABLE = True  # False hides its table row; remove RBC from METHODS to also exclude it from normalization.
TRAJECTORY_LEGEND_INSIDE = True  # True=translucent legends inside plots; False=legends below the figure.
# Third-row choices: cop=plant COP; temperatures=supply+return; return=return only; cost=cumulative realized cost.
THIRD_ROWS = (None, "cop", "temperatures", "return", "cost")  # Each entry creates one figure; None creates the two-row version.
COLUMN_WIDTH = 3.4  # PDF figure width [inches]; 3.4 matches one IEEE column.
# Presentation only: each method maps to (color, Pareto marker, displayed label).
STYLE = {"ThreePhase": ("#439b72", "s", "Three-phase"), "CAMINO": ("#b24583", "D", "CAMINO"),
         "RBC": ("#757575", "P", "RBC"), "AS-MINMPC": ("#5b4897", "X", "AS-MINMPC"),
         "Mo": ("#dd851d", "o", r"$M_o>0$"), "Mf": ("#2878b5", "^", r"$M_f$")}


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
            case = (job["m"], job["N"])
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
                if phase == "prep" and method in ("ThreePhase", "CAMINO", "RBC"):
                    continue
                key = phase if statistic == "mean" else "max_"+phase
                ax.scatter(group[key], group["normalized"], marker=marker, s=23,
                           facecolors=color, edgecolors=color, linewidths=.8)
    phases = ("feedback", "preparation") if len(cases) == 1 else ("feedback", "feedback")
    for i, (phase, ax) in enumerate(zip(phases, axes)):
        ax.set_xlabel(f"{'Mean' if statistic == 'mean' else 'Max'} {phase} [s]")
        if len(cases) == 2:
            ax.set_title(rf"$m={cases[i][0]},\ N={cases[i][1]}$")
        if log_scale:
            ax.set_xscale("log")
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
        fig.suptitle(rf"$m={cases[0][0]},\ N={cases[0][1]}$", fontsize=8)
    legend = fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3,
                        fontsize=6.5, frameon=False, labelspacing=.3, columnspacing=.9, handletextpad=.4, borderpad=.1)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height/fig.dpi
    height = 1.75+legend_height
    fig.set_size_inches(COLUMN_WIDTH, height)
    fig.tight_layout(rect=(0, (legend_height+.015)/height, 1, .95 if len(cases) == 1 else 1), pad=.45, w_pad=.35)
    suffix = "_".join(f"m{c[0]}_N{c[1]}" for c in cases)
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
    ax.set(xlabel=rf"Integer horizon $M_o$ / $M_f$ ($m={case[0]},N={case[1]}$)",
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
    return save_figure(fig, directory, f"{name}_m{case[0]}_N{case[1]}")


def controller_label(method, h):
    return rf"$M_{'o' if method == 'Mo' else 'f'}={h}$" if h is not None else STYLE[method][2]


def trajectory(runs, case, repetition, controllers, directory, third=None):
    selected = []
    for method, h in controllers:
        matches = [r for r in runs if r["case"] == case and r["job"]["repetition"] == repetition
                   and (r["job"]["method"], r["job"]["horizon"]) == (method, h) and r["valid"]]
        if len(matches) != 1:
            raise ValueError(f"Trajectory requires one completed {method} {h}, {case}, repetition {repetition}.")
        selected.append(matches[0])
    height = (2.7 if third is None else 3.65) - (.35 if TRAJECTORY_LEGEND_INSIDE else 0)
    fig, axes = plt.subplots(2 if third is None else 3, 1, sharex=True, figsize=(COLUMN_WIDTH, height))
    handles = []
    temperature_styles = ["-", "--", "-."] + [(0, (4+i, 1, 1, 1, 1, 1)) for i in range(max(0, case[0]-2))]
    for r in selected:
        method, h = r["job"]["method"], r["job"]["horizon"]
        c = r["meta"]["effective"]["scenario"]["config"]
        k, dt = len(r["stage_cost"]), c["dt"]
        t = np.arange(k+1)*dt/3600
        color = STYLE[method][0]
        label = controller_label(method, h)
        handle, = axes[0].step(t, np.r_[r["cooling"].sum(axis=1), r["cooling"][-1].sum()],
                              where="post", color=color, label=label)
        handles.append(handle)
        axes[1].step(t, np.r_[r["binary"].sum(axis=1), r["binary"][-1].sum()], where="post", color=color)
        if third == "cost":
            axes[2].plot(t, r["cumulative_cost"], color=color)
        elif third == "cop":
            power = r["chiller_power"].sum(axis=1)
            cop = np.divide(r["cooling"].sum(axis=1), power, out=np.zeros(k), where=power > 0)
            axes[2].step(t, np.r_[cop, cop[-1]], where="post", color=color)
        elif third in ("return", "temperatures"):
            axes[2].plot(t, r["states"][:, 0], color=color)
            if third == "temperatures":
                for i in range(case[0]):
                    axes[2].plot(t, r["states"][:, i+1], color=color, linestyle=temperature_styles[i+1], linewidth=.75)
    r = selected[0]
    demand, = axes[0].step(t, np.r_[r["raw_demand"][:k], r["raw_demand"][k-1]],
                           where="post", color="black", linestyle="--", label="Demand", linewidth=.8)
    handles.append(demand)
    axes[0].set_ylabel("Cooling [kW]")
    axes[1].set(ylabel="Active units", yticks=range(1, case[0]+1), ylim=(.8, case[0]+.2))
    if third is not None:
        axes[2].set_ylabel({"cost": "Cumulative cost", "cop": "Plant COP", "temperatures": r"$T_r,\ T_s^{(i)}$ [$^\circ$C]",
                            "return": r"$T_r$ [$^\circ$C]"}[third])
        if third in ("temperatures", "return"):
            axes[2].axhline(c["temperature_min"], color="black", linestyle=":", linewidth=.6)
            axes[2].axhline(c["return_max"], color="black", linestyle=":", linewidth=.6)
            if third == "temperatures":
                axes[2].axhline(c["temperature_max"], color="black", linestyle=":", linewidth=.6)
    temperature_handles = [Line2D([], [], color="black", linestyle=temperature_styles[i], linewidth=.8,
        label=r"$T_r$" if i == 0 else rf"$i={i}$") for i in range(case[0]+1)] if third == "temperatures" else []
    for ax in axes:
        ax.grid(alpha=.2)
        ax.margins(x=0)
    axes[-1].set_xlabel("Time [h]")
    if TRAJECTORY_LEGEND_INSIDE:
        axes[0].legend(handles=handles, loc="best", ncol=2, framealpha=.75, edgecolor="none")
        if temperature_handles:
            axes[2].legend(handles=temperature_handles, loc="upper left", ncol=min(4, case[0]+1),
                           framealpha=.75, edgecolor="none", columnspacing=.9, handlelength=2.2)
        bottom = 0
    else:
        fig.legend(handles=handles+temperature_handles, loc="lower center", bbox_to_anchor=(.5, 0),
                   ncol=3 if temperature_handles else 2, frameon=False)
        bottom = .19 if temperature_handles else .12 if third else .17
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=.7, h_pad=.65)
    return save_figure(fig, directory, f"trajectory_{third or 'two_rows'}_m{case[0]}_N{case[1]}_r{repetition}")


def compile_preview(path):
    path = Path(path).resolve()
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is unavailable; install a LaTeX distribution to compile table previews.")
    auxiliary = ROOT / "tmp" / "chiller_benchmarks" / "latex"
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


def latex_tables(runs, case, horizons, directory, *, include_rbc=True):
    summary = summarize(runs, case)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def cell(key, field):
        if field == "prep" and key[0] in ("ThreePhase", "CAMINO", "RBC"):
            return r"\textemdash"
        group = summary.get(key)
        if group is None:
            return r"\textit{missing}"
        if not group["count"]:
            return r"\textit{failed}"
        value = f"{group[field]:.2f}" if field == "normalized" else f"{group[field]:.4f}"
        return value + (r"$^{\dagger}$" if field == "normalized" and group["failed"] else "")
    rows = []
    for method in ("AS-MINMPC", "ThreePhase", "CAMINO") + (("RBC",) if include_rbc else ()):
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
    note = (rf"Paired entries: $M_o$ / $M_f$. $J/\min\bar J$: mean realized stage-sum cost, "
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
            + rf"\caption{{Chiller benchmark: $m={case[0]}$, $N={case[1]}$.}}" + "\n"
            + r"\setlength{\tabcolsep}{3pt}" + "\n"
            + r"\begin{tabular}{@{}lrrr@{}}\toprule" + "\n"
            + r"Method & $J/\min\bar J$ & Prep. [s] & Feedback [s] \\ \midrule" + "\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n"
            + rf"\begin{{minipage}}{{{width}}}\scriptsize {note}\end{{minipage}}" + "\n"
            + rf"\end{{{environment}}}" + "\n")
        name = f"table_m{case[0]}_N{case[1]}_{layout}"
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
            tasks.append((f"tables {case}", lambda c=case: latex_tables(runs, c, table_horizons, tables, include_rbc=INCLUDE_RBC_IN_TABLE)))
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
