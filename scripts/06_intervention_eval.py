"""
06_intervention_eval.py

Apply the fitted temperatures to the TEST split, recompute all per-group
metrics for three conditions:
  - baseline (T = 1)
  - global_T (single T fit on pooled Track A calibration)
  - group_conditional_T (per-group T; OOV groups get global_T fallback)

Apply the pre-registered decision rule:
  - Successful   if worst-group ECE reduced by >= 30% relative
                 AND best-group ECE not increased by > 20% relative
  - Partial      if exactly one criterion met
  - Failed       if neither

Outputs:
  results/intervention_metrics.json
  figures/baseline_vs_intervention.png
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
    BEST_GROUP_ECE_RELATIVE_INCREASE_LIMIT,
    ECE_BINS,
    FIGURES_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    WORST_GROUP_ECE_RELATIVE_REDUCTION_TARGET,
)
from src.metrics import (
    expected_calibration_error,
    predictive_entropy,
    top1_accuracy,
)
from src.temperature import apply_temperature, apply_group_conditional_temperature


def load_test():
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    npz = np.load(RESULTS_DIR / "scores_test.npz")
    inf_df = pd.DataFrame({"utterance_id": npz["utterance_ids"]})
    df = df.merge(inf_df, on="utterance_id", how="inner")
    id_to_idx = {int(uid): i for i, uid in enumerate(npz["utterance_ids"])}
    perm = np.array([id_to_idx[int(u)] for u in df["utterance_id"]], dtype=int)
    scores = npz["scores"][perm]
    return df, scores


def per_group_eces(
    df: pd.DataFrame,
    probs: np.ndarray,
    track: str = "A",
) -> dict:
    """Per-group ECE and confidence for groups in `track`."""
    out = {}
    for grp, sub_idx in df.groupby("group").groups.items():
        sub = df.loc[sub_idx]
        if sub["track"].iloc[0] != track:
            continue
        sub_probs = probs[df.index.get_indexer(sub_idx)]
        confs = sub_probs.max(axis=1)
        if track == "A":
            targets = sub["target_idx"].to_numpy(dtype=int)
            correct = (sub_probs.argmax(axis=1) == targets).astype(int)
            out[grp] = {
                "ece": expected_calibration_error(confs, correct, n_bins=ECE_BINS),
                "top1_accuracy": top1_accuracy(sub_probs, targets),
                "mean_confidence": float(confs.mean()),
                "n_utterances": int(len(sub)),
            }
        else:  # Track B
            ents = predictive_entropy(sub_probs)
            out[grp] = {
                "mean_confidence": float(confs.mean()),
                "mean_entropy": float(ents.mean()),
                "n_utterances": int(len(sub)),
            }
    return out


def apply_decision_rule(baseline_eces: dict, intervention_eces: dict) -> dict:
    """Apply the pre-registered decision rule."""
    base_eces = {g: m["ece"] for g, m in baseline_eces.items()}
    int_eces = {g: m["ece"] for g, m in intervention_eces.items()}

    base_worst_grp = max(base_eces, key=base_eces.get)
    base_best_grp = min(base_eces, key=base_eces.get)

    # Reduction in worst-group ECE (relative)
    base_worst = base_eces[base_worst_grp]
    int_worst_for_same_group = int_eces[base_worst_grp]
    worst_reduction = (base_worst - int_worst_for_same_group) / max(base_worst, 1e-9)

    # Increase in best-group ECE (relative). Use baseline best group.
    base_best = base_eces[base_best_grp]
    int_best_for_same_group = int_eces[base_best_grp]
    best_increase = (int_best_for_same_group - base_best) / max(base_best, 1e-9)

    crit1 = worst_reduction >= WORST_GROUP_ECE_RELATIVE_REDUCTION_TARGET
    crit2 = best_increase <= BEST_GROUP_ECE_RELATIVE_INCREASE_LIMIT

    if crit1 and crit2:
        verdict = "successful"
    elif crit1 or crit2:
        verdict = "partial"
    else:
        verdict = "failed"

    return {
        "baseline_worst_group": base_worst_grp,
        "baseline_worst_ece": float(base_worst),
        "intervention_ece_for_baseline_worst_group": float(int_worst_for_same_group),
        "worst_group_relative_reduction": float(worst_reduction),
        "criterion_1_met (>=30% reduction)": bool(crit1),
        "baseline_best_group": base_best_grp,
        "baseline_best_ece": float(base_best),
        "intervention_ece_for_baseline_best_group": float(int_best_for_same_group),
        "best_group_relative_increase": float(best_increase),
        "criterion_2_met (<=20% increase)": bool(crit2),
        "verdict": verdict,
    }


def plot_comparison(
    baseline_a: dict,
    global_a: dict,
    group_a: dict,
    out_path: Path,
):
    """Bar chart of per-group ECE under three conditions."""
    groups = sorted(baseline_a.keys())
    x = np.arange(len(groups))
    width = 0.27

    base_vals = [baseline_a[g]["ece"] for g in groups]
    glob_vals = [global_a[g]["ece"] for g in groups]
    grp_vals = [group_a[g]["ece"] for g in groups]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(groups)), 4.5))
    ax.bar(x - width, base_vals, width, label="baseline (T=1)", edgecolor="black")
    ax.bar(x, glob_vals, width, label="global T", edgecolor="black")
    ax.bar(x + width, grp_vals, width, label="group-conditional T", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Per-group ECE: baseline vs interventions (Track A)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df, scores = load_test()
    print(f"Loaded test: {len(df)} rows")

    # Load fitted temperatures
    temps = json.loads((RESULTS_DIR / "temperatures.json").read_text())
    global_T = float(temps["global_T"])
    per_group_T = {str(g): float(T) for g, T in temps["per_group_T"].items()}
    print(f"Loaded T values: global={global_T:.4f}, per-group={per_group_T}")

    # ---- Baseline: T=1 ----
    probs_baseline = apply_temperature(scores, T=1.0)

    # ---- Global T ----
    probs_global = apply_temperature(scores, T=global_T)

    # ---- Group-conditional T (with global_T as fallback for OOV) ----
    groups = df["group"].to_numpy()
    probs_group = apply_group_conditional_temperature(
        scores, groups, per_group_T, fallback_t=global_T,
    )

    # Per-group metrics under each condition
    out = {"track_a": {}, "track_b": {}, "decision_rule": {}}

    for cond, P in [
        ("baseline", probs_baseline),
        ("global_T", probs_global),
        ("group_conditional_T", probs_group),
    ]:
        a = per_group_eces(df, P, track="A")
        b = per_group_eces(df, P, track="B")
        out["track_a"][cond] = a
        out["track_b"][cond] = b

    # Decision rule: comparing baseline -> group_conditional_T (the headline)
    out["decision_rule"]["group_conditional_vs_baseline"] = apply_decision_rule(
        out["track_a"]["baseline"], out["track_a"]["group_conditional_T"],
    )
    # Also compare baseline -> global_T (fairer ablation)
    out["decision_rule"]["global_vs_baseline"] = apply_decision_rule(
        out["track_a"]["baseline"], out["track_a"]["global_T"],
    )

    # Pretty-print summary
    print("\n=== Per-group ECE (Track A) under each condition ===")
    print(f"  {'group':<12} {'baseline':>10} {'global_T':>10} {'group_T':>10}")
    for g in sorted(out["track_a"]["baseline"].keys()):
        b = out["track_a"]["baseline"][g]["ece"]
        gl = out["track_a"]["global_T"][g]["ece"]
        gp = out["track_a"]["group_conditional_T"][g]["ece"]
        print(f"  {g:<12} {b:>10.4f} {gl:>10.4f} {gp:>10.4f}")

    print("\n=== Track B: mean confidence under each condition ===")
    print(f"  {'group':<16} {'baseline':>10} {'global_T':>10} {'group_T':>10}")
    for g in sorted(out["track_b"]["baseline"].keys()):
        b = out["track_b"]["baseline"][g]["mean_confidence"]
        gl = out["track_b"]["global_T"][g]["mean_confidence"]
        gp = out["track_b"]["group_conditional_T"][g]["mean_confidence"]
        print(f"  {g:<16} {b:>10.3f} {gl:>10.3f} {gp:>10.3f}")

    print("\n=== Decision rule: group-conditional vs baseline ===")
    for k, v in out["decision_rule"]["group_conditional_vs_baseline"].items():
        print(f"  {k}: {v}")

    out_path = RESULTS_DIR / "intervention_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")

    plot_comparison(
        out["track_a"]["baseline"],
        out["track_a"]["global_T"],
        out["track_a"]["group_conditional_T"],
        FIGURES_DIR / "baseline_vs_intervention.png",
    )
    print(f"Wrote {FIGURES_DIR / 'baseline_vs_intervention.png'}")


if __name__ == "__main__":
    main()
