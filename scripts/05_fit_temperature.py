"""
05_fit_temperature.py

Fit temperature scaling parameters on the CALIBRATION split:
  - one global T (over all Track A examples pooled)
  - one T per Track A group (group-conditional)

Track B groups have no ground-truth label, so we cannot fit T for them.
At test time, OOV speakers will use the global T as a fallback. This is
the H4 trust-tax point: post-hoc calibration cannot fix overconfidence on
OOV speakers, only redistribute mass within the existing label simplex.

Output: results/temperatures.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.constants import RESULTS_DIR, SPLITS_DIR
from src.temperature import (
    fit_group_conditional_temperatures,
    fit_single_temperature,
)


def main():
    cal_df = pd.read_csv(SPLITS_DIR / "cal.csv")
    cal_npz = np.load(RESULTS_DIR / "scores_cal.npz")

    # Align: filter cal_df to rows that have scores (should be all of them)
    inf_df = pd.DataFrame({
        "utterance_id": cal_npz["utterance_ids"],
    })
    cal_df = cal_df.merge(inf_df, on="utterance_id", how="inner")
    id_to_idx = {int(uid): i for i, uid in enumerate(cal_npz["utterance_ids"])}
    perm = np.array([id_to_idx[int(u)] for u in cal_df["utterance_id"]], dtype=int)
    cal_scores = cal_npz["scores"][perm]

    # Track A only — those have target labels
    track_a_mask = cal_df["track"] == "A"
    a_scores = cal_scores[track_a_mask.to_numpy()]
    a_targets = cal_df.loc[track_a_mask, "target_idx"].to_numpy(dtype=int)
    a_groups = cal_df.loc[track_a_mask, "group"].to_numpy()

    print(f"Calibration set: {len(cal_df)} total, {track_a_mask.sum()} Track A")
    print(f"Track A groups in cal: {sorted(set(a_groups))}")

    # ---- Single global temperature ----
    print("\nFitting single global T ...")
    global_t = fit_single_temperature(a_scores, a_targets)
    print(f"  Global T = {global_t:.4f}")

    # ---- Group-conditional temperatures ----
    print("\nFitting group-conditional T ...")
    per_group_t = fit_group_conditional_temperatures(
        a_scores, a_targets, a_groups, min_examples_per_group=50,
    )
    for g, T in sorted(per_group_t.items()):
        n = int((a_groups == g).sum())
        print(f"  T[{g}] = {T:.4f}  (fit on {n} examples)")

    out = {
        "global_T": float(global_t),
        "per_group_T": {str(g): float(T) for g, T in per_group_t.items()},
        "n_calibration_track_a": int(track_a_mask.sum()),
        "min_examples_per_group_for_fit": 50,
    }
    out_path = RESULTS_DIR / "temperatures.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
