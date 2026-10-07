"""
09_calibration_baselines.py

Compare temperature scaling against two further post-hoc baselines, using
the scores already saved by 03 (no new inference):

  1. Isotonic regression on the top-1 confidence, pooled and per-group
     (src/isotonic.py). Non-parametric, so it tests whether temperature
     scaling's failures come from its one-parameter form.
  2. Abstention: answer only when the model looks sure, and refuse
     otherwise. Tests whether "refuse when unsure" protects the groups the
     model fails on. Two scores:
       - max-softmax: answer when top-1 confidence clears a threshold
       - entropy (the pre-registered variant, PREREGISTRATION.md section 9):
         answer when predictive entropy is below a threshold. Entropy needs
         the full probability vector, so it is computed for the temperature
         conditions only; isotonic recalibrates the top-1 confidence alone.

Calibration conditions (all leave the argmax, and so accuracy, unchanged):
  - baseline            T = 1
  - global_T            from results/temperatures.json
  - group_conditional_T from results/temperatures.json
  - isotonic_pooled     one map, fit on pooled Track A calibration
  - isotonic_group      one map per Track A group; Track B (no labels, no
                        per-group fit possible) falls back to the pooled map

Abstention: one threshold, chosen on the pooled Track A calibration split
so that SELECTIVE_ACCURACY_COVERAGE (80%) of it is answered, is applied to
every test group, Track A and B. Per group we report the fraction answered,
accuracy on answered utterances, and the fraction answered *and* wrong. For
Track B every answer is wrong (the true L1 is not in the label set), so the
fraction answered is the harm. The threshold is refit on each condition's
own confidence. Temperature changes which utterances clear it (top-1
softmax at T is not a monotone function of top-1 softmax at T = 1 across
different logit vectors); pooled isotonic is monotone in the T = 1
confidence, so it reorders nothing, but its step function puts ties at the
quantile, and the realized calibration coverage (reported) can overshoot
the target. Entropy thresholds are chosen the same way, on the same
pooled 80% target.

All CIs: 95% speaker-level bootstrap with a fixed seed, so conditions see
paired resamples. Fitted calibrators are held fixed (CIs exclude fitting
uncertainty), as in 06.

Outputs:
  results/calibration_baselines.json
  figures/calibration_baselines_ece.png
  figures/abstention_coverage.png
  figures/abstention_coverage_entropy.png
  figures/abstention_curves.png
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.constants import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    ECE_BINS,
    FIGURES_DIR,
    LOGIT_SCALE,
    RESULTS_DIR,
    SEED,
    SELECTIVE_ACCURACY_COVERAGE,
    SPLITS_DIR,
)
from src.isotonic import (
    apply_group_conditional_isotonic,
    apply_isotonic,
    fit_group_conditional_isotonic,
    fit_isotonic,
)
from src.metrics import (
    accuracy_coverage_curve,
    expected_calibration_error,
    predictive_entropy,
    speaker_bootstrap_ci,
)
from src.plots import abstention_curves, multi_condition_bars
from src.temperature import apply_group_conditional_temperature, apply_temperature

# Reuse 06's decision rule verbatim rather than copy it.
_spec = importlib.util.spec_from_file_location(
    "intervention_eval", Path(__file__).resolve().parent / "06_intervention_eval.py"
)
_06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_06)
apply_decision_rule = _06.apply_decision_rule

CONDITION_LABELS = {
    "baseline": "baseline (T=1)",
    "global_T": "global T",
    "group_conditional_T": "group-conditional T",
    "isotonic_pooled": "isotonic (pooled)",
    "isotonic_group": "isotonic (per group)",
}


def load_split(name: str):
    """Split CSV rows aligned to their saved scores, scores scaled to logits."""
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    npz = np.load(RESULTS_DIR / f"scores_{name}.npz")
    df = df.merge(pd.DataFrame({"utterance_id": npz["utterance_ids"]}),
                  on="utterance_id", how="inner").reset_index(drop=True)
    id_to_idx = {int(uid): i for i, uid in enumerate(npz["utterance_ids"])}
    perm = np.array([id_to_idx[int(u)] for u in df["utterance_id"]], dtype=int)
    return df, npz["scores"][perm] * LOGIT_SCALE


def correctness(df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """0/1 per row; Track B rows are always 0 (no correct label exists)."""
    targets = df["target_idx"].fillna(-1).to_numpy(dtype=int)
    return ((df["track"] == "A").to_numpy() & (preds == targets)).astype(int)


def per_group(df: pd.DataFrame, track: str, stat_fn, keys: list) -> dict:
    """
    Run `stat_fn(rows, idx) -> list` per group in `track`, where `rows` are
    the group's positional rows in df and `idx` indexes into them. Returns
    {group: {key: point, key_ci: [lo, hi], n_utterances, n_speakers}}.
    """
    out = {}
    for grp, sub in df[df["track"] == track].groupby("group"):
        rows = sub.index.to_numpy()
        speakers = sub["speaker"].to_numpy()
        point = stat_fn(rows, np.arange(len(rows)))
        with warnings.catch_warnings():
            # Empty answered sets give NaN accuracy in some resamples.
            warnings.simplefilter("ignore", RuntimeWarning)
            cis = np.atleast_2d(speaker_bootstrap_ci(
                lambda idx: stat_fn(rows, idx), speakers,
                n_boot=BOOTSTRAP_N, ci=BOOTSTRAP_CI, seed=SEED,
            ))
        m = {k: float(v) for k, v in zip(keys, point)}
        m.update({f"{k}_ci": [float(lo), float(hi)] for k, (lo, hi) in zip(keys, cis)})
        m["n_utterances"] = int(len(rows))
        m["n_speakers"] = int(sub["speaker"].nunique())
        out[str(grp)] = m
    return out


def calibration_metrics(df, confs, correct) -> dict:
    def stats_a(rows, idx):
        c, y = confs[rows][idx], correct[rows][idx]
        return [expected_calibration_error(c, y, n_bins=ECE_BINS), y.mean(), c.mean()]

    def stats_b(rows, idx):
        return [confs[rows][idx].mean()]

    return {
        "track_a": per_group(df, "A", stats_a, ["ece", "top1_accuracy", "mean_confidence"]),
        "track_b": per_group(df, "B", stats_b, ["mean_confidence"]),
    }


def abstention_metrics(df, confs, correct, threshold) -> dict:
    answered = (confs >= threshold).astype(int)

    def stats_a(rows, idx):
        a, y = answered[rows][idx], correct[rows][idx]
        sel = y[a == 1].mean() if a.any() else np.nan
        return [a.mean(), sel, (a * (1 - y)).mean()]

    def stats_b(rows, idx):
        return [answered[rows][idx].mean()]

    return {
        "track_a": per_group(df, "A", stats_a,
                             ["coverage", "selective_accuracy", "answered_wrong_rate"]),
        "track_b": per_group(df, "B", stats_b, ["coverage"]),
    }


def run_abstention(cal_scores, test_scores, cal_mask, test_df, test_correct) -> dict:
    """
    For each condition, pick the threshold on the calibration rows in
    `cal_mask` that answers SELECTIVE_ACCURACY_COVERAGE of them, then apply it
    to every test group. Scores are oriented so higher = answer.
    """
    out = {"track_a": {}, "track_b": {}, "thresholds": {}}
    ta = (test_df["track"] == "A").to_numpy()
    for cond in test_scores:
        cal_c = cal_scores[cond][cal_mask]
        thr = float(np.quantile(cal_c, 1.0 - SELECTIVE_ACCURACY_COVERAGE))
        answered_a = test_scores[cond][ta] >= thr
        out["thresholds"][cond] = {
            "threshold": thr,
            "cal_track_a_coverage": float((cal_c >= thr).mean()),
            "test_track_a_coverage": float(answered_a.mean()),
            "test_track_a_selective_accuracy": float(test_correct[ta][answered_a].mean()),
            "test_track_a_accuracy_no_abstention": float(test_correct[ta].mean()),
        }
        m = abstention_metrics(test_df, test_scores[cond], test_correct, thr)
        out["track_a"][cond] = m["track_a"]
        out["track_b"][cond] = m["track_b"]
    return out


def within_group_aurc(df, confs, correct) -> dict:
    """Within-group ranking quality of a score (higher = more sure), Track A."""
    def stats(rows, idx):
        return [accuracy_coverage_curve(confs[rows][idx], correct[rows][idx])["aurc"],
                correct[rows][idx].mean()]

    out = per_group(df, "A", stats, ["aurc", "top1_accuracy"])
    for grp, sub in df[df["track"] == "A"].groupby("group"):
        # Best achievable AURC at this accuracy: all correct answers first.
        acc = out[str(grp)]["top1_accuracy"]
        y = np.sort(correct[sub.index.to_numpy()])[::-1]
        out[str(grp)]["aurc_oracle"] = accuracy_coverage_curve(
            np.arange(len(y), 0, -1, dtype=float), y)["aurc"]
        out[str(grp)]["aurc_random"] = float(1.0 - acc)
    return out


def main():
    cal_df, cal_logits = load_split("cal")
    test_df, test_logits = load_split("test")
    print(f"Loaded cal: {len(cal_df)} rows, test: {len(test_df)} rows")

    temps = json.loads((RESULTS_DIR / "temperatures.json").read_text())
    global_T = float(temps["global_T"])
    per_group_T = {str(g): float(T) for g, T in temps["per_group_T"].items()}

    # Raw (T=1) confidence and the argmax, which no condition changes.
    cal_p = apply_temperature(cal_logits, 1.0)
    test_p = apply_temperature(test_logits, 1.0)
    cal_conf, test_conf = cal_p.max(axis=1), test_p.max(axis=1)
    cal_correct = correctness(cal_df, cal_p.argmax(axis=1))
    test_correct = correctness(test_df, test_p.argmax(axis=1))

    # ---- Fit isotonic maps on Track A calibration ----
    # Fit on T=1 confidence. Any global temperature is a monotone transform
    # of it, so the fitted map's outputs would be identical.
    a = (cal_df["track"] == "A").to_numpy()
    iso_pooled = fit_isotonic(cal_conf[a], cal_correct[a])
    iso_group = fit_group_conditional_isotonic(
        cal_conf[a], cal_correct[a], cal_df.loc[a, "group"].to_numpy(),
        min_examples_per_group=temps["min_examples_per_group_for_fit"],
    )
    print("\nIsotonic maps (calibration, Track A):")
    print(f"  pooled: {len(iso_pooled.X_thresholds_)} knots, "
          f"output range [{iso_pooled.y_thresholds_.min():.3f}, "
          f"{iso_pooled.y_thresholds_.max():.3f}]")
    for g, iso in sorted(iso_group.items()):
        print(f"  {g:<10} {len(iso.X_thresholds_):>4} knots, output range "
              f"[{iso.y_thresholds_.min():.3f}, {iso.y_thresholds_.max():.3f}]")

    # ---- Confidence under each condition, on both splits ----
    def confidences(df, logits, raw_conf):
        groups = df["group"].to_numpy()
        return {
            "baseline": raw_conf,
            "global_T": apply_temperature(logits, global_T).max(axis=1),
            "group_conditional_T": apply_group_conditional_temperature(
                logits, groups, per_group_T, fallback_t=global_T).max(axis=1),
            "isotonic_pooled": apply_isotonic(iso_pooled, raw_conf),
            "isotonic_group": apply_group_conditional_isotonic(
                raw_conf, groups, iso_group, fallback=iso_pooled),
        }

    cal_confs = confidences(cal_df, cal_logits, cal_conf)
    test_confs = confidences(test_df, test_logits, test_conf)

    # Negated predictive entropy, so higher = answer, as for confidence.
    def neg_entropies(df, logits):
        groups = df["group"].to_numpy()
        return {
            "baseline": -predictive_entropy(apply_temperature(logits, 1.0)),
            "global_T": -predictive_entropy(apply_temperature(logits, global_T)),
            "group_conditional_T": -predictive_entropy(apply_group_conditional_temperature(
                logits, groups, per_group_T, fallback_t=global_T)),
        }

    cal_negent = neg_entropies(cal_df, cal_logits)
    test_negent = neg_entropies(test_df, test_logits)

    out = {
        "calibration": {"track_a": {}, "track_b": {}},
        "decision_rule": {},
        "abstention": {},
        "abstention_entropy": {},
        "within_group_ranking": {},
        "within_group_ranking_entropy": {},
        "isotonic_fit": {
            "pooled_knots": int(len(iso_pooled.X_thresholds_)),
            "per_group_knots": {str(g): int(len(i.X_thresholds_)) for g, i in iso_group.items()},
            "per_group_output_range": {
                str(g): [float(i.y_thresholds_.min()), float(i.y_thresholds_.max())]
                for g, i in iso_group.items()
            },
        },
    }

    # ---- Calibration metrics ----
    for cond, c in test_confs.items():
        m = calibration_metrics(test_df, c, test_correct)
        out["calibration"]["track_a"][cond] = m["track_a"]
        out["calibration"]["track_b"][cond] = m["track_b"]

    # The temperature rows must reproduce 06 exactly.
    prev = json.loads((RESULTS_DIR / "intervention_metrics.json").read_text())
    for cond in ("baseline", "global_T", "group_conditional_T"):
        for g, m in prev["track_a"][cond].items():
            assert np.isclose(m["ece"], out["calibration"]["track_a"][cond][g]["ece"]), (cond, g)

    base_a = out["calibration"]["track_a"]["baseline"]
    for cond in ("global_T", "group_conditional_T", "isotonic_pooled", "isotonic_group"):
        out["decision_rule"][f"{cond}_vs_baseline"] = apply_decision_rule(
            base_a, out["calibration"]["track_a"][cond])

    # ---- Abstention ----
    out["abstention"] = run_abstention(cal_confs, test_confs, a, test_df, test_correct)
    out["abstention_entropy"] = run_abstention(cal_negent, test_negent, a, test_df, test_correct)
    for t in out["abstention_entropy"]["thresholds"].values():
        # Report the entropy threshold in nats: answer when entropy <= it.
        t["max_entropy_nats"] = -t.pop("threshold")

    out["within_group_ranking"] = within_group_aurc(test_df, test_conf, test_correct)
    out["within_group_ranking_entropy"] = within_group_aurc(
        test_df, test_negent["baseline"], test_correct)

    # ---- Print summary ----
    conds = list(CONDITION_LABELS)
    short = ["base", "glob_T", "grp_T", "iso_pool", "iso_grp"]
    print("\n=== Per-group ECE (Track A, test) ===")
    print(f"  {'group':<12}" + "".join(f"{s:>10}" for s in short) + f"{'acc':>8}")
    for g in sorted(base_a):
        row = [out["calibration"]["track_a"][c][g]["ece"] for c in conds]
        print(f"  {g:<12}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{base_a[g]['top1_accuracy']:>8.3f}")

    print("\n=== Mean confidence (Track B, test; no correct answer exists) ===")
    for g in sorted(out["calibration"]["track_b"]["baseline"]):
        row = [out["calibration"]["track_b"][c][g]["mean_confidence"] for c in conds]
        print(f"  {g:<14}" + "".join(f"{v:>10.3f}" for v in row))

    print("\n=== Decision rule (pre-registered) vs baseline ===")
    for k, v in out["decision_rule"].items():
        print(f"  {k:<36} worst-group reduction {v['worst_group_relative_reduction']:+.1%}, "
              f"best-group increase {v['best_group_relative_increase']:+.1%} -> {v['verdict']}")

    print(f"\n=== Abstention: threshold for {SELECTIVE_ACCURACY_COVERAGE:.0%} pooled "
          "cal coverage, fraction of each test group answered ===")
    print(f"  {'group':<14}" + "".join(f"{s:>10}" for s in short))
    for track in ("track_a", "track_b"):
        for g in sorted(out["abstention"][track]["baseline"]):
            row = [out["abstention"][track][c][g]["coverage"] for c in conds]
            print(f"  {g:<14}" + "".join(f"{v:>10.3f}" for v in row))
    print("  Pooled Track A test: coverage / selective acc (no-abstention acc "
          f"{out['abstention']['thresholds']['baseline']['test_track_a_accuracy_no_abstention']:.3f})")
    for c, s in zip(conds, short):
        t = out["abstention"]["thresholds"][c]
        print(f"    {s:<9} thr={t['threshold']:.4f}  cov={t['test_track_a_coverage']:.3f}  "
              f"sel_acc={t['test_track_a_selective_accuracy']:.3f}")

    ent = out["abstention_entropy"]
    ent_conds, ent_short = list(ent["thresholds"]), short[:3]
    print(f"\n=== Entropy abstention (pre-registered): threshold for "
          f"{SELECTIVE_ACCURACY_COVERAGE:.0%} pooled cal coverage, fraction answered ===")
    print(f"  {'group':<14}" + "".join(f"{s:>10}" for s in ent_short))
    for track in ("track_a", "track_b"):
        for g in sorted(ent[track]["baseline"]):
            row = [ent[track][c][g]["coverage"] for c in ent_conds]
            print(f"  {g:<14}" + "".join(f"{v:>10.3f}" for v in row))
    print("  Pooled Track A test:")
    for c, s in zip(ent_conds, ent_short):
        t = ent["thresholds"][c]
        print(f"    {s:<9} max_H={t['max_entropy_nats']:.4f}  cov={t['test_track_a_coverage']:.3f}  "
              f"sel_acc={t['test_track_a_selective_accuracy']:.3f}")

    print("\n=== Within-group ranking (T=1): AURC, lower is better ===")
    print(f"  {'group':<12}{'max-softmax':>12}{'entropy':>10}{'oracle':>9}{'random':>9}")
    for g, m in sorted(out["within_group_ranking"].items()):
        e = out["within_group_ranking_entropy"][g]["aurc"]
        print(f"  {g:<12}{m['aurc']:>12.3f}{e:>10.3f}{m['aurc_oracle']:>9.3f}"
              f"{m['aurc_random']:>9.3f}")

    out_path = RESULTS_DIR / "calibration_baselines.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")

    # ---- Figures ----
    multi_condition_bars(
        {CONDITION_LABELS[c]: out["calibration"]["track_a"][c] for c in conds},
        key="ece", out_path=FIGURES_DIR / "calibration_baselines_ece.png",
        ylabel="Expected Calibration Error",
        title="Per-group ECE under each post-hoc calibration method (Track A)",
    )

    abst_conds = ["baseline", "group_conditional_T", "isotonic_group"]
    a_groups = sorted(out["abstention"]["track_a"]["baseline"])
    b_groups = sorted(g for g in out["abstention"]["track_b"]["baseline"] if g != "OOV-aggregate")
    merged = {
        CONDITION_LABELS[c]: {**out["abstention"]["track_a"][c], **out["abstention"]["track_b"][c]}
        for c in abst_conds
    }
    multi_condition_bars(
        merged, key="coverage", out_path=FIGURES_DIR / "abstention_coverage.png",
        groups=a_groups + b_groups + ["OOV-aggregate"],
        ylabel="fraction of utterances answered",
        title=f"Max-softmax abstention at {SELECTIVE_ACCURACY_COVERAGE:.0%} pooled coverage: "
              "who still gets an answer",
        ref_line=SELECTIVE_ACCURACY_COVERAGE, ref_label="pooled target coverage",
    )
    multi_condition_bars(
        {CONDITION_LABELS[c]: {**ent["track_a"][c], **ent["track_b"][c]} for c in ent_conds},
        key="coverage", out_path=FIGURES_DIR / "abstention_coverage_entropy.png",
        groups=a_groups + b_groups + ["OOV-aggregate"],
        ylabel="fraction of utterances answered",
        title=f"Entropy-threshold abstention at {SELECTIVE_ACCURACY_COVERAGE:.0%} pooled "
              "coverage: who still gets an answer",
        ref_line=SELECTIVE_ACCURACY_COVERAGE, ref_label="pooled target coverage",
    )

    within = {}
    for g, sub in test_df[test_df["track"] == "A"].groupby("group"):
        rows = sub.index.to_numpy()
        within[str(g)] = accuracy_coverage_curve(test_conf[rows], test_correct[rows])
    thresholds = np.linspace(test_conf.min(), 1.0, 400)
    sweep_groups = a_groups + ["OOV-aggregate"]
    sweep = {
        "thresholds": thresholds,
        "coverage": {
            g: (test_conf[(test_df["group"] == g).to_numpy()][None, :]
                >= thresholds[:, None]).mean(axis=1)
            for g in sweep_groups
        },
    }
    abstention_curves(
        within, sweep, out["abstention"]["thresholds"]["baseline"]["threshold"],
        FIGURES_DIR / "abstention_curves.png",
    )
    for f in ("calibration_baselines_ece", "abstention_coverage",
              "abstention_coverage_entropy", "abstention_curves"):
        print(f"Wrote {FIGURES_DIR / f'{f}.png'}")


if __name__ == "__main__":
    main()
