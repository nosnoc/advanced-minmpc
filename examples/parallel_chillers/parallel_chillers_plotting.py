"""Plant trajectory and timing plots.

J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and J. Drgona,
Data Center Chiller Plant Optimization via Mixed-Integer Nonlinear
Differentiable Predictive Control, 2026. https://arxiv.org/html/2603.28323v1
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_records(problem, records, raw, directory: Path, show=True, plot_individual=True, *, save_results=False):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    c = problem.model.config
    count = max((len(r["binary"]) for r in records.values()), default=0)
    if not count:
        return
    end = count*c.dt/3600
    upper_cooling = max(1.1*np.max(raw[:count]), 1.1*max(
        (np.max(np.sum(r["cooling"],axis=1)) for r in records.values() if len(r["cooling"])), default=0))
    return_values = np.concatenate([np.asarray(r["states"])[:,0] for r in records.values()])
    limits = (min(c.temperature_min-1, np.min(return_values)-1), max(c.return_max+1, np.max(return_values)+1))

    def save(figure, name):
        for extension in ("png", "pdf"):
            if save_results:
                figure.savefig(directory/f"{name}.{extension}", dpi=160, bbox_inches="tight")
        if not show:
            plt.close(figure)

    overlay, comparison = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    comparison[0].plot(np.arange(count)*c.dt/3600, raw[:count], "k--", label="Raw demand")
    comparison[1].axhline(c.temperature_min, color="black", linestyle=":", label="Operating bounds")
    comparison[1].axhline(c.return_max, color="black", linestyle=":")
    for label, record in records.items():
        K = len(record["binary"])
        if not K:
            continue
        t = np.arange(K)*c.dt/3600
        tx = np.arange(K+1)*c.dt/3600
        states, binary, continuous = (np.asarray(record[k]) for k in ("states", "binary", "continuous"))
        delivered = np.sum(record["cooling"], axis=1)
        active = np.sum(binary, axis=1)
        comparison_label = label + (" (failed)" if record["failure"] else "")
        comparison[0].plot(t, delivered, label=comparison_label)
        comparison[1].plot(tx, states[:,0], label=comparison_label)
        comparison[2].step(np.r_[t,tx[-1]], np.r_[active,active[-1]], where="post", label=comparison_label)
        if not plot_individual:
            continue
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, raw[:K], "k--", label="Raw demand")
        axes[0].plot(t, delivered, label="Delivered cooling")
        axes[1].plot(tx, states[:,0], label="Return temperature")
        axes[2].step(np.r_[t, tx[-1]], np.r_[active, active[-1]], where="post", label="Active chillers")
        axes[0].set_ylim(0, upper_cooling)
        axes[1].set_ylim(*limits)
        axes[2].set_ylim(0.5, c.M+0.5)
        axes[2].set_yticks(np.arange(1,c.M+1))
        axes[1].axhline(c.temperature_min, color="black", linestyle=":", label="Operating bounds")
        axes[1].axhline(c.return_max, color="black", linestyle=":")
        for ax, ylabel in zip(axes, ("Cooling [kW]", "Return [°C]", "Active units")):
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, end)
            if end >= 3:
                ax.axvspan(3, min(7, end), color="gray", alpha=0.08)
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("Time [h]")
        figure.suptitle(label + (" — FAILED / partial run" if record["failure"] else ""))
        figure.tight_layout()
        name = label.lower().replace(" ", "_").replace("=", "_").replace("-", "_")
        save(figure, name+"_fig5")
        detail, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        flows = continuous[:, :c.M] if c.use_active_flow else continuous[:, :c.M]*binary
        for i in range(c.M):
            axes[0].plot(tx, states[:,1+i], label=f"Unit {i+1}")
            for ax, values in ((axes[1], continuous[:,c.M+i]),
                               (axes[2], flows[:,i]), (axes[3], binary[:,i])):
                ax.step(np.r_[t,tx[-1]], np.r_[values,values[-1]], where="post", label=f"Unit {i+1}")
        for ax, ylabel in zip(axes, ("Supply [°C]", "Evaporator [°C]", "Actual flow [kg/s]", "Switch")):
            ax.set_ylabel(ylabel)
            ax.legend(ncol=c.M)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time [h]")
        detail.suptitle(label)
        detail.tight_layout()
        save(detail, name+"_detail")
    for ax, ylabel in zip(comparison, ("Cooling [kW]", "Return [°C]", "Active units")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlim(0,end)
    comparison[2].set_yticks(np.arange(1,c.M+1))
    comparison[-1].set_xlabel("Time [h]")
    overlay.tight_layout()
    save(overlay,"comparison")
    timing, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
    for label, record in records.items():
        K = len(record["binary"])
        t = np.arange(K)*c.dt/3600
        name = label + (" (failed)" if record["failure"] else "")
        axes[0].plot(t, np.cumsum(record["stage_cost"]), label=name)
        axes[1].plot(t, record["preparation"], label=name)
        axes[2].plot(t, record["feedback"], label=name)
    for ax, ylabel in zip(axes,("Stage-sum cost", "Preparation [s]", "Feedback [s]")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time [h]")
    timing.tight_layout()
    save(timing,"cost_and_timing")
    if show:
        plt.show()
