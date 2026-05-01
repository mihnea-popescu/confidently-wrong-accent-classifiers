"""
07_make_figures.py

Regenerate all paper figures from the existing JSON results. Run this any
time you want to update the figures without re-running inference or
evaluation.

Reads:
  results/baseline_metrics.json
  results/intervention_metrics.json

Writes:
  figures/calibration_scatter.png         <- headline calibration figure
  figures/baseline_vs_intervention.png    <- improved bar chart
  figures/track_b_confidence.png          <- trust-tax visual

Run with:  python scripts/07_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.constants import FIGURES_DIR, RESULTS_DIR
from src.plots import (
    accuracy_vs_confidence_scatter,
    intervention_comparison_bars,
    track_b_confidence_chart,
)


def main():
    baseline = json.loads((RESULTS_DIR / "baseline_metrics.json").read_text())
    intervention = json.loads((RESULTS_DIR / "intervention_metrics.json").read_text())

    # The intervention dict has per-condition metrics; pull what we need.
    track_a_baseline = intervention["track_a"]["baseline"]
    track_a_global = intervention["track_a"]["global_T"]
    track_a_group = intervention["track_a"]["group_conditional_T"]

    track_b_baseline = intervention["track_b"]["baseline"]
    track_b_global = intervention["track_b"]["global_T"]
    track_b_group = intervention["track_b"]["group_conditional_T"]

    # Figure 1: calibration scatter, baseline + intervention side-by-side
    out_scatter = FIGURES_DIR / "calibration_scatter.png"
    accuracy_vs_confidence_scatter(
        track_a_baseline=track_a_baseline,
        track_a_intervention=track_a_group,
        out_path=out_scatter,
    )
    print(f"Wrote {out_scatter}")

    # Figure 2: bar chart with accuracy annotations
    out_bars = FIGURES_DIR / "baseline_vs_intervention.png"
    intervention_comparison_bars(
        baseline_a=track_a_baseline,
        global_a=track_a_global,
        group_a=track_a_group,
        out_path=out_bars,
    )
    print(f"Wrote {out_bars}")

    # Figure 3: Track B trust-tax visual
    # Use mean of in-vocab baseline confidence as the reference line
    in_vocab_mean = float(np.mean(
        [m["mean_confidence"] for m in track_a_baseline.values()]
    ))
    out_oov = FIGURES_DIR / "track_b_confidence.png"
    track_b_confidence_chart(
        track_b_baseline=track_b_baseline,
        track_b_global=track_b_global,
        track_b_group=track_b_group,
        out_path=out_oov,
        in_vocab_baseline_mean=in_vocab_mean,
    )
    print(f"Wrote {out_oov}")


if __name__ == "__main__":
    main()