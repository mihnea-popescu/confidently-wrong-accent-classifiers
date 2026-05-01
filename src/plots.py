"""
Plotting utilities. Kept minimal — most data analysis output is JSON; plots
are generated on demand for the paper.

Figures used in the paper:
    accuracy_vs_confidence_scatter  — headline calibration figure (Track A)
    intervention_comparison_bars    — per-group ECE: baseline vs interventions
    track_b_confidence_chart        — trust-tax: OOV confidence under each condition

Supplementary:
    reliability_diagram, grid_reliability_diagrams — kept for completeness;
    less informative than the scatter when confidence is degenerate at one
    point (which is what we saw on this model).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.metrics import reliability_curve


# ----------------------------------------------------------------------
# Headline figure: per-group accuracy vs mean confidence
# ----------------------------------------------------------------------

def accuracy_vs_confidence_scatter(
    track_a_baseline: dict,
    track_a_intervention: dict | None,
    out_path: Path,
):
    """
    Scatter: x = per-group mean confidence, y = per-group top-1 accuracy.

    The diagonal y = x is perfect calibration — if the model says X% on
    average for a group, it should be right X% of the time on that group.
    Above the diagonal = underconfident; below = overconfident.

    If track_a_intervention is provided, draws an arrow from baseline
    point to intervention point per group, showing the intervention's
    effect.

    Inputs are dicts of {group_name: {"top1_accuracy": ..., "mean_confidence": ...}}
    """
    # Two panels if we have intervention; else one
    if track_a_intervention is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        axes = [ax]
        titles = ["Baseline (T = 1)"]
        sources = [track_a_baseline]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
        titles = ["Baseline (T = 1)", "Group-conditional T"]
        sources = [track_a_baseline, track_a_intervention]

    # Use one color per group, consistent across panels
    groups = sorted(track_a_baseline.keys())
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    for ax, src, title in zip(axes, sources, titles):
        # Diagonal first, behind data
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, alpha=0.4,
                label="perfect calibration", zorder=1)
        for g in groups:
            m = src[g]
            x, y = m["mean_confidence"], m["top1_accuracy"]
            ax.scatter([x], [y], s=120, color=colors[g], edgecolor="black",
                       linewidth=0.7, zorder=3)
            # Label slightly offset
            ax.annotate(
                g, (x, y), xytext=(8, 4), textcoords="offset points",
                fontsize=10,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Improved bar chart: per-group ECE with accuracy annotations
# ----------------------------------------------------------------------

def intervention_comparison_bars(
    baseline_a: dict,
    global_a: dict,
    group_a: dict,
    out_path: Path,
):
    """
    Per-group ECE under three conditions, with top-1 accuracy annotated
    beneath each group label so readers can tell whether 'low ECE' means
    'well calibrated' or 'uniformly wrong with low confidence'.

    Inputs are dicts of {group: {"ece": ..., "top1_accuracy": ...}}
    """
    groups = sorted(baseline_a.keys())
    x = np.arange(len(groups))
    width = 0.27

    base_vals = [baseline_a[g]["ece"] for g in groups]
    glob_vals = [global_a[g]["ece"] for g in groups]
    grp_vals = [group_a[g]["ece"] for g in groups]

    # Tick labels: "group\n(acc=0.65)"
    accs = [baseline_a[g]["top1_accuracy"] for g in groups]
    labels = [f"{g}\n(acc={a:.2f})" for g, a in zip(groups, accs)]

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(groups)), 5.0))
    ax.bar(x - width, base_vals, width, label="baseline (T=1)",
           color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar(x, glob_vals, width, label="global T",
           color="#DD8452", edgecolor="black", linewidth=0.6)
    ax.bar(x + width, grp_vals, width, label="group-conditional T",
           color="#55A868", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Per-group ECE: baseline vs interventions (Track A)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Track B trust-tax visual
# ----------------------------------------------------------------------

def track_b_confidence_chart(
    track_b_baseline: dict,
    track_b_global: dict,
    track_b_group: dict,
    out_path: Path,
    in_vocab_baseline_mean: float | None = None,
):
    """
    Per-OOV-group mean confidence under each condition. Used to show that
    temperature scaling fit on in-vocab groups *raises* OOV confidence as
    a side effect (the trust-tax point).

    If in_vocab_baseline_mean is provided, draws a horizontal reference
    line for it, so readers can compare OOV to in-vocab levels.
    """
    # Stable order, with OOV-aggregate last for visual separation
    named = [g for g in sorted(track_b_baseline.keys()) if g != "OOV-aggregate"]
    if "OOV-aggregate" in track_b_baseline:
        named.append("OOV-aggregate")
    groups = named

    x = np.arange(len(groups))
    width = 0.27

    base_vals = [track_b_baseline[g]["mean_confidence"] for g in groups]
    glob_vals = [track_b_global[g]["mean_confidence"] for g in groups]
    grp_vals = [track_b_group[g]["mean_confidence"] for g in groups]

    fig, ax = plt.subplots(figsize=(max(7.5, 1.0 * len(groups) + 2), 4.5))
    ax.bar(x - width, base_vals, width, label="baseline (T=1)",
           color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar(x, glob_vals, width, label="global T",
           color="#DD8452", edgecolor="black", linewidth=0.6)
    ax.bar(x + width, grp_vals, width, label="group-conditional T",
           color="#55A868", edgecolor="black", linewidth=0.6)

    if in_vocab_baseline_mean is not None:
        ax.axhline(
            in_vocab_baseline_mean, color="black", linestyle=":", linewidth=1,
            label=f"in-vocab baseline mean ({in_vocab_baseline_mean:.3f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("mean top-1 confidence")
    ax.set_title("Track B (out-of-vocabulary) confidence under each condition")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Tight y-range so the small differences are visible
    all_vals = base_vals + glob_vals + grp_vals
    pad = 0.005
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Existing utilities (kept for completeness; less useful when confidence
# is degenerate at a single point as in our results)
# ----------------------------------------------------------------------

def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    title: str = "Reliability",
    n_bins: int = 15,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    curve = reliability_curve(confidences, correctness, n_bins=n_bins)
    centers = curve["bin_centers"]
    accs = curve["bin_accs"]
    counts = curve["bin_counts"]

    width = 1.0 / n_bins
    valid = counts > 0
    ax.bar(centers[valid], accs[valid], width=width * 0.9, edgecolor="black",
           alpha=0.8, label="accuracy")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def grid_reliability_diagrams(
    per_group: dict,
    out_path: Path,
    n_bins: int = 15,
    cols: int = 4,
    figsize_per_panel: tuple = (3.2, 3.2),
):
    groups = list(per_group.keys())
    n = len(groups)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_panel[0], rows * figsize_per_panel[1]),
        squeeze=False,
    )
    for i, g in enumerate(groups):
        ax = axes[i // cols][i % cols]
        d = per_group[g]
        reliability_diagram(d["confidences"], d["correctness"], title=g, n_bins=n_bins, ax=ax)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)