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


def ci_yerr(per_group: dict, groups: list, key: str):
    """
    Asymmetric matplotlib `yerr` (2, N) from per-group `<key>_ci` entries,
    or None if the results predate bootstrap CIs.
    """
    if not all(f"{key}_ci" in per_group[g] for g in groups):
        return None
    pts = np.array([per_group[g][key] for g in groups])
    lo = np.array([per_group[g][f"{key}_ci"][0] for g in groups])
    hi = np.array([per_group[g][f"{key}_ci"][1] for g in groups])
    # Clip at 0: with lumpy few-speaker bootstraps the point can sit on an edge.
    return np.vstack([np.clip(pts - lo, 0, None), np.clip(hi - pts, 0, None)])


_ERR_KW = dict(capsize=3, error_kw=dict(elinewidth=1, ecolor="black"))


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

    # Tick labels: "group\nacc=0.65\n3 spk"
    accs = [baseline_a[g]["top1_accuracy"] for g in groups]
    labels = [
        f"{g}\nacc={a:.2f}"
        + (f"\n{baseline_a[g]['n_speakers']} spk" if "n_speakers" in baseline_a[g] else "")
        for g, a in zip(groups, accs)
    ]

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(groups)), 5.0))
    ax.bar(x - width, base_vals, width, label="baseline (T=1)",
           color="#4C72B0", edgecolor="black", linewidth=0.6,
           yerr=ci_yerr(baseline_a, groups, "ece"), **_ERR_KW)
    ax.bar(x, glob_vals, width, label="global T",
           color="#DD8452", edgecolor="black", linewidth=0.6,
           yerr=ci_yerr(global_a, groups, "ece"), **_ERR_KW)
    ax.bar(x + width, grp_vals, width, label="group-conditional T",
           color="#55A868", edgecolor="black", linewidth=0.6,
           yerr=ci_yerr(group_a, groups, "ece"), **_ERR_KW)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Per-group ECE: baseline vs interventions (Track A)")
    if ci_yerr(baseline_a, groups, "ece") is not None:
        ax.text(0.0, -0.26, "Error bars: 95% speaker-level bootstrap CI",
                transform=ax.transAxes, fontsize=8, alpha=0.7)
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
    errs = [
        ci_yerr(src, groups, "mean_confidence")
        for src in (track_b_baseline, track_b_global, track_b_group)
    ]
    ax.bar(x - width, base_vals, width, label="baseline (T=1)",
           color="#4C72B0", edgecolor="black", linewidth=0.6,
           yerr=errs[0], **_ERR_KW)
    ax.bar(x, glob_vals, width, label="global T",
           color="#DD8452", edgecolor="black", linewidth=0.6,
           yerr=errs[1], **_ERR_KW)
    ax.bar(x + width, grp_vals, width, label="group-conditional T",
           color="#55A868", edgecolor="black", linewidth=0.6,
           yerr=errs[2], **_ERR_KW)

    if in_vocab_baseline_mean is not None:
        ax.axhline(
            in_vocab_baseline_mean, color="black", linestyle=":", linewidth=1,
            label=f"in-vocab baseline mean ({in_vocab_baseline_mean:.3f})",
        )

    ax.set_xticks(x)
    tick = [
        f"{g} ({track_b_baseline[g]['n_speakers']} spk)"
        if "n_speakers" in track_b_baseline[g] else g
        for g in groups
    ]
    ax.set_xticklabels(tick, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("mean top-1 confidence")
    title = "Track B (out-of-vocabulary) confidence under each condition"
    if errs[0] is not None:
        title += "\n(error bars: 95% speaker-level bootstrap CI)"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Tight y-range so the small differences are visible
    all_vals = base_vals + glob_vals + grp_vals
    for src in (track_b_baseline, track_b_global, track_b_group):
        for g in groups:
            all_vals += src[g].get("mean_confidence_ci", [])
    pad = 0.005
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Calibration-baseline comparison: any number of conditions
# ----------------------------------------------------------------------

_CONDITION_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def multi_condition_bars(
    per_condition: dict,
    key: str,
    out_path: Path,
    ylabel: str,
    title: str,
    groups: list | None = None,
    ref_line: float | None = None,
    ref_label: str | None = None,
):
    """
    Grouped bars of per-group `key` under each condition, with `<key>_ci`
    error bars where present.

    per_condition : {condition_label: {group: {key: ..., f"{key}_ci": [lo, hi]}}}
                    Insertion order sets bar order and color.
    groups        : x-axis order; defaults to the sorted groups of the first
                    condition.
    """
    labels = list(per_condition.keys())
    if groups is None:
        groups = sorted(per_condition[labels[0]].keys())
    x = np.arange(len(groups))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(groups)), 5.0))
    for i, lab in enumerate(labels):
        src = per_condition[lab]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width,
               [src[g][key] for g in groups], width, label=lab,
               color=_CONDITION_COLORS[i % len(_CONDITION_COLORS)],
               edgecolor="black", linewidth=0.6,
               yerr=ci_yerr(src, groups, key), **_ERR_KW)
    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle=":", linewidth=1, label=ref_label)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    if ci_yerr(per_condition[labels[0]], groups, key) is not None:
        title += "\n(error bars: 95% speaker-level bootstrap CI)"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def abstention_curves(
    within_group: dict,
    pooled_sweep: dict,
    threshold: float,
    out_path: Path,
):
    """
    Two panels for max-softmax abstention.

    Left: within-group accuracy-coverage curves — does confidence rank this
    group's correct predictions above its incorrect ones?
    within_group : {group: {"coverage": (N,), "selective_accuracy": (N,)}}

    Right: per-group fraction answered as one pooled confidence threshold is
    swept — does any threshold stop answering for the groups the model
    fails on while still answering for the others?
    pooled_sweep : {"thresholds": (M,), "coverage": {group: (M,)}}
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    groups = sorted(within_group.keys())
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(sorted(pooled_sweep["coverage"].keys()))}

    for g in groups:
        c = within_group[g]
        ax_l.plot(c["coverage"], c["selective_accuracy"], color=colors[g], label=g)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.set_xlabel("coverage (fraction of the group answered)")
    ax_l.set_ylabel("accuracy on answered utterances")
    ax_l.set_title("Within-group accuracy vs coverage (T = 1)")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=9)

    t = pooled_sweep["thresholds"]
    for g, cov in sorted(pooled_sweep["coverage"].items()):
        ax_r.plot(t, cov, color=colors[g], label=g,
                  linestyle="--" if g == "OOV-aggregate" else "-")
    ax_r.axvline(threshold, color="black", linestyle=":", linewidth=1,
                 label=f"threshold for 80% pooled coverage ({threshold:.3f})")
    ax_r.set_xlim(t.min(), t.max())
    ax_r.set_ylim(0, 1.02)
    ax_r.set_xlabel("confidence threshold (abstain below)")
    ax_r.set_ylabel("fraction of the group answered")
    ax_r.set_title("One pooled threshold, per-group fraction answered")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="lower left")

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