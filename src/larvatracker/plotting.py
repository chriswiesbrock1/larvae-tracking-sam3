"""Figures for individual droplets and for group comparisons.

All functions take an explicit output path and close the figure afterwards, so
they are safe to call inside batch loops without leaking figure handles.
"""

from __future__ import annotations

import matplotlib

# Batch scripts run without a display; the Agg backend avoids a hard dependency
# on a GUI toolkit. Set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from larvatracker.config import AnalysisConfig


def plot_droplet_traces(
    frames: np.ndarray,
    metrics_per_part: dict[str, dict],
    out_path: str,
    title: str = "",
    dpi: int = 200,
) -> str:
    """One panel per keypoint: raw trace, smoothed trace, bursts and onset."""
    parts = list(metrics_per_part)
    fig, axes = plt.subplots(len(parts), 1, figsize=(12, 2.6 * len(parts)), sharex=True)

    if len(parts) == 1:
        axes = [axes]

    for ax, part in zip(axes, parts):
        m = metrics_per_part[part]

        ax.plot(frames, m["raw_step"], alpha=0.25, color="gray", label="raw")
        ax.plot(frames, m["ma_step"], lw=2, color="tab:blue", label="smoothed")

        if np.isfinite(m["onset_frame"]):
            ax.axvline(
                frames[int(m["onset_frame"])],
                color="green",
                ls="--",
                alpha=0.7,
                label="onset",
            )

        if len(m["peaks"]):
            ax.plot(
                frames[m["peaks"]],
                m["ma_step"][m["peaks"]],
                "ro",
                ms=3,
                label="bursts",
            )

        ax.set_ylabel(f"{part}\n(px/frame)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("frame")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_kinematics(
    frames: np.ndarray,
    theta: np.ndarray,
    ang_vel: np.ndarray,
    joints: np.ndarray,
    total_curvature: np.ndarray,
    out_path: str,
    joint_labels: list[str] | None = None,
    dpi: int = 200,
) -> str:
    """Body axis angle, angular velocity, joint angles and total curvature."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(frames, theta)
    axes[0].set_ylabel("body axis angle (rad)")

    axes[1].plot(frames, ang_vel)
    axes[1].set_ylabel("angular velocity\n(rad/frame)")

    labels = joint_labels or [f"joint {i + 1}" for i in range(joints.shape[1])]
    for i, label in enumerate(labels):
        axes[2].plot(frames, joints[:, i], label=label)
    axes[2].set_ylabel("joint angles (rad)")
    axes[2].legend(fontsize=8)

    axes[3].plot(frames, total_curvature)
    axes[3].set_ylabel("curvature (rad)")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_group_dashboard(
    summary: pd.DataFrame,
    out_path: str,
    reference_bodypart: str = "a",
    dpi: int = 200,
) -> str:
    """Burst frequency per group, plus onset latency for one reference keypoint.

    Only rows with ``Time_Bin == "full"`` are used, i.e. whole-recording values.
    """
    full = summary[summary["Time_Bin"] == "full"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    sns.boxplot(data=full, x="Group", y="Freq_Hz", hue="BodyPart", ax=axes[0])
    axes[0].set_title("Locomotion frequency per group")
    axes[0].set_ylabel("bursts / second")

    onset = full[full["BodyPart"] == reference_bodypart]
    if not onset.empty:
        sns.boxplot(data=onset, x="Group", y="Onset_Sec", ax=axes[1], boxprops=dict(alpha=0.5))
        sns.swarmplot(data=onset, x="Group", y="Onset_Sec", color=".25", ax=axes[1])
    axes[1].set_title(f"Onset of heavy movement (keypoint {reference_bodypart})")
    axes[1].set_ylabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_frequency_over_time(summary: pd.DataFrame, out_path: str, dpi: int = 200) -> str:
    """Burst frequency per time bin, one line per group."""
    binned = summary[summary["Time_Bin"] != "full"].copy()
    if binned.empty:
        return out_path

    binned["Time_Bin"] = pd.to_numeric(binned["Time_Bin"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=binned,
        x="Time_Sec",
        y="Freq_Hz",
        hue="Group",
        style="Group",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Locomotion frequency over time")
    ax.set_xlabel("time in recording (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_population_overview(summary: pd.DataFrame, out_path: str, dpi: int = 200) -> str:
    """Frequency distribution per keypoint and a droplet-by-keypoint heatmap."""
    full = summary[summary["Time_Bin"] == "full"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    sns.boxplot(data=full, x="BodyPart", y="Freq_Hz", ax=axes[0])
    sns.stripplot(data=full, x="BodyPart", y="Freq_Hz", color="orange", alpha=0.5, ax=axes[0])
    axes[0].set_title("Movement frequency across all larvae")
    axes[0].set_ylabel("bursts / second")

    pivot = full.pivot_table(index="Droplet", columns="BodyPart", values="Freq_Hz")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1], cbar_kws={"label": "Hz"})
    axes[1].set_title("Per-droplet frequency fingerprints")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Temperature response
# ---------------------------------------------------------------------------

# Colour families: controls in grey, each drug in its own hue with lightness
# encoding the dose. This makes a dose-dependent effect readable without
# consulting the legend.
TEMPERATURE_PALETTE = {
    "ETOH": "#555555",
    "DMSO": "#777777",
    "PBS": "#999999",
    "1 mM Asp": "#9ecae1", "5 mM Asp": "#4292c6", "10 mM Asp": "#08519c",
    "1 mM Ibu": "#fcae91", "5 mM Ibu": "#fb6a4a", "10 mM Ibu": "#cb181d",
    "1 mM Dic": "#c7e9c0", "5 mM Dic": "#74c476", "10 mM Dic": "#238b45",
    "1 mM Cis": "#dadaeb", "5 mM Cis": "#9e9ac8", "10 mM Cis": "#6a51a3",
}

# Which vehicle belongs with which drug, for the faceted figure.
DRUG_FAMILIES = {
    "Aspirin": (["1 mM Asp", "5 mM Asp", "10 mM Asp"], "ETOH"),
    "Ibuprofen": (["1 mM Ibu", "5 mM Ibu", "10 mM Ibu"], "ETOH"),
    "Diclofenac": (["1 mM Dic", "5 mM Dic", "10 mM Dic"], "DMSO"),
    "Cisplatin": (["1 mM Cis", "5 mM Cis", "10 mM Cis"], "PBS"),
}


def _order_groups(groups) -> list[str]:
    """Controls first, then drugs by family and ascending dose."""
    known = list(TEMPERATURE_PALETTE)
    present = set(groups)
    ordered = [g for g in known if g in present]
    return ordered + sorted(present - set(ordered))


def plot_temperature_response(
    summary: pd.DataFrame,
    out_path: str,
    dpi: int = 200,
) -> str:
    """Normalised movement against temperature, one line per treatment group.

    The shaded band is the standard error over *animals*, not frames. A dashed
    line at 1.0 marks the baseline: values above it mean the larvae moved more
    than during their own opening seconds.
    """
    groups = _order_groups(summary["Group"].unique())

    fig, ax = plt.subplots(figsize=(12, 7))

    for group in groups:
        data = summary[summary["Group"] == group].sort_values("Temp_Bin")
        colour = TEMPERATURE_PALETTE.get(group)

        ax.plot(data["Temp_Bin"], data["Mean"], label=group, color=colour, lw=2)
        ax.fill_between(
            data["Temp_Bin"],
            data["Mean"] - data["SEM"],
            data["Mean"] + data["SEM"],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )

    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("movement, normalised to first seconds")
    ax.set_title("Larval movement across temperature")
    ax.grid(alpha=0.25)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_temperature_response_by_family(
    summary: pd.DataFrame,
    out_path: str,
    dpi: int = 200,
) -> str:
    """One panel per drug, each with its own vehicle control for reference.

    Pooling every group into a single axis hides dose-response structure; this
    figure separates the drugs while repeating the relevant control in each
    panel so the comparison stays visible.
    """
    present = set(summary["Group"].unique())
    families = {
        name: (doses, control)
        for name, (doses, control) in DRUG_FAMILIES.items()
        if present & set(doses)
    }

    if not families:
        return out_path

    n = len(families)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    flat = axes.flat

    for ax, (name, (doses, control)) in zip(flat, families.items()):
        for group in [control] + [d for d in doses if d in present]:
            data = summary[summary["Group"] == group].sort_values("Temp_Bin")
            if data.empty:
                continue

            colour = TEMPERATURE_PALETTE.get(group)
            style = "--" if group == control else "-"

            ax.plot(data["Temp_Bin"], data["Mean"], label=group, color=colour, lw=2, ls=style)
            ax.fill_between(
                data["Temp_Bin"],
                data["Mean"] - data["SEM"],
                data["Mean"] + data["SEM"],
                color=colour,
                alpha=0.18,
                linewidth=0,
            )

        ax.axhline(1.0, ls=":", lw=0.8, color="grey")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in list(flat)[len(families):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("temperature (°C)")
    for row in axes:
        row[0].set_ylabel("movement (normalised)")

    fig.suptitle("Movement across temperature, by drug", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_significance_heatmap(
    stats: pd.DataFrame,
    out_path: str,
    dpi: int = 200,
) -> str:
    """Effect size per group and temperature bin, with significance marked.

    Colour shows the log2 ratio of treatment to control median, so zero means
    no difference. A dot marks bins that survive FDR correction — reading the
    colour without the dot would overstate the evidence.
    """
    pivot = stats.pivot_table(index="Group", columns="Temp_Bin", values="Log2_Ratio")
    signif = stats.pivot_table(index="Group", columns="Temp_Bin", values="signif")

    if pivot.empty:
        return out_path

    order = [g for g in _order_groups(pivot.index) if g in pivot.index]
    pivot = pivot.loc[order]
    signif = signif.reindex(index=order, columns=pivot.columns)

    # Scale to the 98th percentile rather than the maximum: a single extreme
    # bin — usually a small group in a sparsely covered temperature range —
    # would otherwise compress every other cell towards white.
    values = np.abs(pivot.to_numpy())
    limit = float(np.nanpercentile(values, 98)) if np.isfinite(values).any() else 1.0
    limit = max(limit, 0.1)

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * pivot.shape[1]), 0.55 * len(pivot) + 2.5))

    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        ax=ax,
        cbar_kws={"label": "log2(treatment / control)"},
    )

    # Overlay a marker on the significant cells.
    ys, xs = np.where(signif.fillna(False).to_numpy())
    ax.plot(xs + 0.5, ys + 0.5, "o", color="black", ms=3.5, ls="none")

    ax.set_xlabel("temperature bin (°C)")
    ax.set_ylabel("")
    ax.set_title("Effect versus vehicle control (dot = significant after FDR)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
