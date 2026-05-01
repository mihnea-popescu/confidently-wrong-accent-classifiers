"""
Plotting utilities. Kept minimal — most data analysis output is JSON; plots
are generated on demand for the paper.

Reliability diagrams: per-bin accuracy vs confidence. The diagonal y=x is
perfect calibration. Bars below the diagonal = overconfident, above =
underconfident.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.metrics import reliability_curve


def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    title: str = "Reliability",
    n_bins: int = 15,
    ax=None,
):
    """Single reliability diagram."""
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
    """
    Small-multiples reliability diagrams.

    per_group: dict mapping group_name -> dict with keys 'confidences',
    'correctness'.
    """
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
    # blank out unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
