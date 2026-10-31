"""Fuel-efficient driving plots: tracking, sampled fuel/cost, torque, rpm and gears.

S. Mallick et al., Learning-Based MPC for Fuel Efficient Control of Autonomous
Vehicles with Discrete Gear Selection (2025), equations (1)–(12), Section VI.
https://arxiv.org/pdf/2503.11359v3
Source: https://github.com/SamuelMallick/mpcrl-vehicle-gears/tree/paper-2025
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

STYLES = {"three_phase": ("tab:blue", "-"), "mo": ("tab:orange", "--"),
          "mf": ("tab:green", "-."), "bonmin": ("tab:purple", ":")}


def plot_records(problem, records, reference, directory, experiment, show=False, *, save_results=False):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    c = problem.model.config
    figure, axes = plt.subplots(8, 1, figsize=(11, 14), sharex=True)
    timing, times = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    count = max((len(r["binary"]) for r in records.values()), default=0)
    t_ref = np.arange(count+1)*c.dt
    axes[0].axhline(0, color="black", linestyle=":", alpha=0.5)
    axes[1].plot(t_ref, reference[:count+1, 1], color="black", linestyle=":", label="Reference")
    handles = []
    for label, r in records.items():
        color, style = STYLES[r["kind"]]
        name = label+(" [FAILED / partial]" if r["failure"] else "")
        handles.append(Line2D([], [], color=color, linestyle=style, label=name))
        K = len(r["binary"])
        x = np.asarray(r["states"])
        tx = np.arange(K+1)*c.dt
        axes[0].plot(tx, x[:, 0]-reference[:K+1, 0], color=color, linestyle=style)
        axes[1].plot(tx, x[:, 1], color=color, linestyle=style)
        if not K:
            continue
        u, v = np.asarray(r["continuous"]), np.asarray(r["binary"])
        for ax, key in zip(axes[2:4], ("fuel", "stage_cost")):
            ax.plot(tx, np.r_[0., np.cumsum(r[key])], color=color, linestyle=style)
        for ax, values in ((axes[4], u[:, 0]), (axes[6], u[:, 1]), (axes[7], v@np.arange(1, 7))):
            ax.step(tx, np.r_[values, values[-1]], where="post", color=color, linestyle=style)
        axes[5].plot(np.repeat(tx, 2)[1:-1], np.asarray(r["rpm"]).reshape(-1), color=color, linestyle=style)
        for ax, key in zip(times, ("preparation", "feedback")):
            ax.plot(np.arange(len(r[key]))*c.dt, r[key], color=color, linestyle=style, marker=".", markersize=2)
        if experiment == "OCP" and r["M_f"] is not None:
            boundary = r["M_f"]*c.dt
            for ax in axes:
                ax.axvspan(boundary, K*c.dt, color=color, alpha=0.07)
            axes[-1].text(boundary+0.1, 1.4, "fractional tail: weighted gear index", color=color, fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    axes[4].set_ylim(0, 1.05*c.torque_max)
    axes[6].set_ylim(-0.02*c.brake_max, 1.02*c.brake_max)
    for bound in (c.rpm_min, c.rpm_max):
        axes[5].axhline(bound, color="black", linestyle=":", alpha=0.4)
    for ax, label in zip(axes, ("Position error [m]", "Speed [m/s]", "Cumulative fuel [g]",
        "Cumulative stage cost", "Torque [Nm]", "Engine speed [rpm]", "Brake force [N]", "Gear / relaxed index")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_yticks(np.arange(1, 7))
    axes[-1].set_ylim(0.8, 6.25)
    for ax, label in zip(times, ("Preparation [s]", "Feedback [s]")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for fig, group, filename in ((figure, axes, "time_traces"), (timing, times, "timing")):
        group[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Fuel-efficient driving — {experiment}, seed={c.seed}")
        fig.legend(handles=handles, loc="lower center", fontsize=8, ncol=2)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        for extension in ("pdf", "png"):
            if save_results:
                fig.savefig(directory/f"{filename}.{extension}", dpi=160)
        if not show:
            plt.close(fig)
    if show:
        plt.show()
