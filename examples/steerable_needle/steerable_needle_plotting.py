"""Needle positions, yaw/pitch/roll, controls, modes and spherical-obstacle paths.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)–(39).
https://arxiv.org/pdf/2002.09658v3
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

STYLES = {"three_phase": ("tab:blue", "-"), "mo": ("tab:orange", "--"),
          "mf": ("tab:green", "-."), "bonmin": ("tab:purple", ":"),
          "camino": ("tab:red", "-")}


def plot_records(problem, records, directory, experiment, show=False, *, save_results=False):
    if save_results:
        directory.mkdir(parents=True, exist_ok=True)
    c = problem.model.config
    traces, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
    angles, orientation = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    spatial = plt.figure(figsize=(9, 7))
    path = spatial.add_subplot(111, projection="3d")
    timing, times = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    handles, all_positions = [], [np.array([c.x_target[:3]])]
    for label, r in records.items():
        x = np.asarray(r["states"])
        K = len(r["binary"])
        color, style = STYLES[r["kind"]]
        name = label+(" [FAILED / partial]" if r["failure"] else "")
        handles.append(Line2D([], [], color=color, linestyle=style, label=name))
        tx = np.arange(len(x))*c.dt
        for i in range(3):
            axes[i].plot(tx, x[:, i], color=color, linestyle=style)
            orientation[i].plot(tx, x[:, i+3], color=color, linestyle=style)
        path.plot(*x[:, :3].T, color=color, linestyle=style)
        all_positions.append(x[:, :3])
        if not K:
            continue
        u, v = np.asarray(r["continuous"]), np.asarray(r["binary"])
        t = np.arange(K+1)*c.dt
        for ax, values in zip(axes[3:], (u[:, 0], u[:, 1], 1+v[:, 1])):
            ax.step(t, np.r_[values, values[-1]], where="post", color=color, linestyle=style)
        if experiment == "OCP" and r["M_f"] is not None:
            boundary = r["M_f"]*c.dt
            for ax in (*axes, *orientation):
                ax.axvspan(boundary, K*c.dt, color=color, alpha=0.07)
            axes[-1].text(boundary+0.04, 1.5, "fractional tail: 1 + rotate weight", fontsize=8, color=color)
            path.plot(*x[r["M_f"]:, :3].T, color=color, linestyle=style, linewidth=3)
        for ax, key in zip(times, ("preparation", "feedback")):
            ax.plot(np.arange(len(r[key]))*c.dt, r[key], color=color, linestyle=style, marker=".", markersize=2)
    for i in range(3):
        axes[i].axhline(c.x_target[i], color="black", linestyle=":", alpha=0.5)
    for ax, label in zip(axes, ("$p_1$", "$p_2$", "$p_3$", "$u_1$", "$u_2$ [rad/s]", "Mode / weight")):
        ax.set_ylabel(label)
    axes[-1].set_yticks([1, 2], ["1: insert", "2: rotate"])
    axes[-1].set_ylim(0.85, 2.15)
    for ax, label in zip(orientation, ("$x_4$ yaw [rad]", "$x_5$ pitch [rad]", "$x_6$ roll [rad]")):
        ax.set_ylabel(label)
    for ax, label in zip(times, ("Preparation [s]", "Feedback [s]")):
        ax.set_ylabel(label)
    for group in (axes, orientation, times):
        for ax in group:
            ax.grid(alpha=0.25)
        group[-1].set_xlabel("Time [s]")
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 45), np.linspace(0, np.pi, 25))
    sphere = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    for center in np.asarray(c.centers):
        surface = center[:, None, None]+c.radius*sphere
        path.plot_surface(*surface, color="gray", alpha=0.18, linewidth=0)
        all_positions.extend([np.array([center-c.radius, center+c.radius])])
    path.scatter(*c.x_target[:3], color="black", s=50, depthshade=False)
    algorithm_handles = handles.copy()
    handles.append(Line2D([], [], color="black", marker="o", linestyle="", label="Target"))
    positions = np.vstack(all_positions)
    midpoint = (positions.min(axis=0)+positions.max(axis=0))/2
    half_span = max(np.ptp(positions, axis=0).max()/2, 1)*1.05
    for setter, middle in zip((path.set_xlim, path.set_ylim, path.set_zlim), midpoint):
        setter(middle-half_span, middle+half_span)
    path.set_box_aspect((1, 1, 1))
    path.set(xlabel="$p_1$", ylabel="$p_2$", zlabel="$p_3$")
    for fig, name in ((traces, "time_traces"), (angles, "orientation"),
                      (spatial, "trajectory_3d"), (timing, "timing")):
        fig.suptitle(f"Steerable needle — {experiment}, ell={c.ell}")
        legend = handles if fig is spatial else algorithm_handles
        if fig is traces:
            legend = legend+[Line2D([], [], color="black", linestyle=":", label="Target")]
        fig.legend(handles=legend, loc="lower center", fontsize=8, ncol=2)
        fig.tight_layout(rect=(0, 0.075, 1, 0.96))
        for extension in ("pdf", "png"):
            if save_results:
                fig.savefig(directory/f"{name}.{extension}", dpi=170)
        if not show:
            plt.close(fig)
    if show:
        plt.show()
