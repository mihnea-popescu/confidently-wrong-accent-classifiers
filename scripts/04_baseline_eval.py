"""
04_baseline_eval.py

Evaluate the BASELINE (T=1) model on the test split:
  - per-group ECE, top-1 accuracy, Brier, selective accuracy (Track A only)
  - per-group mean confidence, mean entropy (Tracks A and B)
  - reliability diagrams (Track A only — needs ground truth)
  - the three pre-registered hard cases

Outputs:
  results/baseline_metrics.json
  figures/baseline_reliability.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import (
    ECE_BINS,
    FIGURES_DIR,
    RESULTS_DIR,
    SELECTIVE_ACCURACY_COVERAGE,
    SPLITS_DIR,
)
from src.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    predictive_entropy,
    selective_accuracy_at_coverage,
    top1_accuracy,
)
from src.plots import grid_reliability_diagrams


def load_inference(split_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load split CSV and corresponding scores, joined and aligned by
    utterance_id. Returns (df, probs) where probs is post-softmax probability.
    """
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    npz = np.load(RESULTS_DIR / f"scores_{split_name}.npz")
    inf_df = pd.DataFrame({
        "utterance_id": npz["utterance_ids"],
        "duration_sec_inference": npz["durations_sec"],
    })
    df = df.merge(inf_df, on="utterance_id", how="inner")
    # Reorder scores to match df order
    id_to_idx = {int(uid): i for i, uid in enumerate(npz["utterance_ids"])}
    perm = np.array([id_to_idx[int(u)] for u in df["utterance_id"]], dtype=int)
    scores = npz["scores"][perm]

    # Model returns log-probabilities; convert to probabilities for metrics.
    # softmax is invariant to additive shifts, so log_softmax -> exp gives probs.
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return df, probs


def compute_track_a_metrics(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Per-group metrics for groups in Track A (have ground-truth label)."""
    out = {}
    for grp, sub_idx in df.groupby("group").groups.items():
        sub = df.loc[sub_idx]
        if sub["track"].iloc[0] != "A":
            continue
        idx = sub_idx
        sub_probs = probs[df.index.get_indexer(idx)]
        targets = sub["target_idx"].to_numpy(dtype=int)
        preds = sub_probs.argmax(axis=1)
        confs = sub_probs.max(axis=1)
        correct = (preds == targets).astype(int)

        out[grp] = {
            "n_speakers": int(sub["speaker"].nunique()),
            "n_utterances": int(len(sub)),
            "top1_accuracy": top1_accuracy(sub_probs, targets),
            "ece": expected_calibration_error(confs, correct, n_bins=ECE_BINS),
            "brier": brier_score_multiclass(sub_probs, targets),
            "selective_accuracy_80": selective_accuracy_at_coverage(
                confs, correct, coverage=SELECTIVE_ACCURACY_COVERAGE
            ),
            "mean_confidence": float(confs.mean()),
            "mean_entropy": float(predictive_entropy(sub_probs).mean()),
        }
    return out


def compute_track_b_metrics(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Per-group confidence/entropy for Track B (no ground-truth label)."""
    out = {}
    for grp, sub_idx in df.groupby("group").groups.items():
        sub = df.loc[sub_idx]
        if sub["track"].iloc[0] != "B":
            continue
        sub_probs = probs[df.index.get_indexer(sub_idx)]
        confs = sub_probs.max(axis=1)
        ents = predictive_entropy(sub_probs)
        out[grp] = {
            "n_speakers": int(sub["speaker"].nunique()),
            "n_utterances": int(len(sub)),
            "mean_confidence": float(confs.mean()),
            "mean_entropy": float(ents.mean()),
            "median_confidence": float(np.median(confs)),
            "p25_confidence": float(np.percentile(confs, 25)),
            "p75_confidence": float(np.percentile(confs, 75)),
        }
    return out


def evaluate_hard_cases(
    df: pd.DataFrame,
    probs: np.ndarray,
    track_a_metrics: dict,
    track_b_metrics: dict,
) -> dict:
    """Test the three pre-registered hard cases."""
    out = {}

    # HC1: in-vocab disparity. Worst-group ECE / best-group ECE >= 2.0?
    a_eces = [m["ece"] for m in track_a_metrics.values()]
    if a_eces:
        ratio = max(a_eces) / max(min(a_eces), 1e-9)
        out["HC1_in_vocab_ece_ratio"] = {
            "best_group_ece": float(min(a_eces)),
            "worst_group_ece": float(max(a_eces)),
            "ratio": float(ratio),
            "predicted": ">= 2.0",
            "held": bool(ratio >= 2.0),
        }

    # HC2: OOV overconfidence. |mean_conf(in-vocab) - mean_conf(OOV)| < 0.10?
    a_confs = [m["mean_confidence"] for m in track_a_metrics.values()]
    # For OOV, use the named groups (not OOV-aggregate) so the comparison is
    # against speakers we have meaningful per-group structure for, plus
    # also report including OOV-aggregate.
    b_named_confs = [
        m["mean_confidence"]
        for g, m in track_b_metrics.items()
        if g != "OOV-aggregate"
    ]
    b_all_confs = [m["mean_confidence"] for m in track_b_metrics.values()]

    if a_confs and b_named_confs:
        a_mean = float(np.mean(a_confs))
        b_mean = float(np.mean(b_named_confs))
        gap = abs(a_mean - b_mean)
        out["HC2_oov_overconfidence"] = {
            "in_vocab_mean_confidence": a_mean,
            "oov_named_mean_confidence": b_mean,
            "oov_all_mean_confidence": float(np.mean(b_all_confs)) if b_all_confs else None,
            "gap": float(gap),
            "predicted": "< 0.10 (model fails to lower confidence on OOV)",
            "held": bool(gap < 0.10),
        }

    # HC3: utterance length. Track A only (needs labels for ECE).
    df_a = df[df["track"] == "A"]
    probs_a = probs[df.index.get_indexer(df_a.index)]
    targets_a = df_a["target_idx"].to_numpy(dtype=int)
    confs_a = probs_a.max(axis=1)
    correct_a = (probs_a.argmax(axis=1) == targets_a).astype(int)
    durations = df_a["duration_sec"].to_numpy(dtype=float)

    short_mask = durations < 3.0
    long_mask = durations > 5.0

    short_ece = (
        expected_calibration_error(
            confs_a[short_mask], correct_a[short_mask], n_bins=ECE_BINS
        )
        if short_mask.sum() > 0 else float("nan")
    )
    long_ece = (
        expected_calibration_error(
            confs_a[long_mask], correct_a[long_mask], n_bins=ECE_BINS
        )
        if long_mask.sum() > 0 else float("nan")
    )

    if not (np.isnan(short_ece) or np.isnan(long_ece)) and long_ece > 0:
        ratio = short_ece / long_ece
    else:
        ratio = float("nan")

    out["HC3_utterance_length"] = {
        "short_ece (<3s)": float(short_ece) if not np.isnan(short_ece) else None,
        "long_ece (>5s)": float(long_ece) if not np.isnan(long_ece) else None,
        "n_short": int(short_mask.sum()),
        "n_long": int(long_mask.sum()),
        "ratio_short_to_long": float(ratio) if not np.isnan(ratio) else None,
        "predicted": ">= 1.30 (short utterances more miscalibrated)",
        "held": bool(ratio >= 1.30) if not np.isnan(ratio) else None,
    }

    return out


def make_reliability_grid(df: pd.DataFrame, probs: np.ndarray, out_path: Path):
    """Reliability diagrams for Track A groups only (need ground truth)."""
    per_group = {}
    for grp, sub_idx in df.groupby("group").groups.items():
        sub = df.loc[sub_idx]
        if sub["track"].iloc[0] != "A":
            continue
        sub_probs = probs[df.index.get_indexer(sub_idx)]
        targets = sub["target_idx"].to_numpy(dtype=int)
        confs = sub_probs.max(axis=1)
        correct = (sub_probs.argmax(axis=1) == targets).astype(int)
        per_group[grp] = {"confidences": confs, "correctness": correct}

    grid_reliability_diagrams(per_group, out_path, n_bins=ECE_BINS, cols=3)
    print(f"  Wrote {out_path}")


def main():
    df, probs = load_inference("test")
    print(f"Loaded test split: {len(df)} rows")

    track_a = compute_track_a_metrics(df, probs)
    track_b = compute_track_b_metrics(df, probs)
    hard_cases = evaluate_hard_cases(df, probs, track_a, track_b)

    out = {
        "track_a_per_group": track_a,
        "track_b_per_group": track_b,
        "hard_cases": hard_cases,
    }

    # Pretty print summary
    print("\n=== Track A (in-vocabulary) ===")
    print(f"  {'group':<12} {'n_utt':>6} {'acc':>6} {'ece':>7} {'brier':>7} {'sel80':>6} {'conf':>6}")
    for g, m in sorted(track_a.items()):
        print(
            f"  {g:<12} {m['n_utterances']:>6} {m['top1_accuracy']:>6.3f} "
            f"{m['ece']:>7.4f} {m['brier']:>7.4f} {m['selective_accuracy_80']:>6.3f} "
            f"{m['mean_confidence']:>6.3f}"
        )

    print("\n=== Track B (out-of-vocabulary) ===")
    print(f"  {'group':<16} {'n_utt':>6} {'mean_conf':>10} {'mean_ent':>10}")
    for g, m in sorted(track_b.items()):
        print(
            f"  {g:<16} {m['n_utterances']:>6} {m['mean_confidence']:>10.3f} "
            f"{m['mean_entropy']:>10.3f}"
        )

    print("\n=== Pre-registered hard cases ===")
    for hc, info in hard_cases.items():
        print(f"  {hc}: held={info.get('held')}")
        for k, v in info.items():
            if k == "held":
                continue
            print(f"      {k}: {v}")

    out_path = RESULTS_DIR / "baseline_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")

    make_reliability_grid(df, probs, FIGURES_DIR / "baseline_reliability.png")


if __name__ == "__main__":
    main()
