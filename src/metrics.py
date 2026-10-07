"""
Calibration and accuracy metrics.

ECE (Expected Calibration Error) is the headline metric. It measures the
average gap between predicted confidence and actual accuracy, binned by
confidence level. A perfectly calibrated classifier has ECE = 0.

All functions take numpy arrays (not torch tensors) for portability and
to make unit-testing trivial.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error using equal-width bins on confidence.

    Parameters
    ----------
    confidences : (N,) array of float in [0, 1]
        Top-1 predicted probability for each example.
    correctness : (N,) array of bool / int 0/1
        Whether the top-1 prediction was correct.
    n_bins : int
        Number of equal-width bins on [0, 1].

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better-calibrated.

    Formula (Guo et al. 2017):
        ECE = sum_b (|B_b| / N) * | acc(B_b) - conf(B_b) |
    where B_b is the set of examples whose confidence falls in bin b.
    Empty bins contribute 0.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    assert confidences.shape == correctness.shape
    assert confidences.ndim == 1

    n = len(confidences)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin includes 1.0
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """
    Per-bin accuracy and confidence, suitable for reliability diagrams.

    Returns
    -------
    dict with keys:
        bin_centers : (n_bins,) array
        bin_counts  : (n_bins,) array of int
        bin_accs    : (n_bins,) array, NaN where bin is empty
        bin_confs   : (n_bins,) array, NaN where bin is empty
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_counts = np.zeros(n_bins, dtype=int)
    bin_accs = np.full(n_bins, np.nan)
    bin_confs = np.full(n_bins, np.nan)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_counts[i] = int(mask.sum())
        if bin_counts[i] > 0:
            bin_accs[i] = correctness[mask].mean()
            bin_confs[i] = confidences[mask].mean()

    return {
        "bin_centers": bin_centers,
        "bin_counts": bin_counts,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
    }


def brier_score_multiclass(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Multiclass Brier score: mean squared error between one-hot labels and
    predicted probability vectors. Lower is better.

    Parameters
    ----------
    probs  : (N, K) array, rows sum to 1
    labels : (N,) array of int in [0, K)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, k = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def selective_accuracy_at_coverage(
    confidences: np.ndarray,
    correctness: np.ndarray,
    coverage: float = 0.80,
) -> float:
    """
    Accuracy on the top `coverage` fraction of examples sorted by confidence.

    A model that is well-ordered by confidence (even if miscalibrated in
    absolute terms) will have higher selective accuracy than its overall
    accuracy.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    assert 0 < coverage <= 1
    n = len(confidences)
    if n == 0:
        return float("nan")
    n_keep = max(1, int(round(coverage * n)))
    # indices of the top-n_keep most confident
    order = np.argsort(-confidences)
    keep = order[:n_keep]
    return float(correctness[keep].mean())


def accuracy_coverage_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> dict:
    """
    Max-softmax abstention swept over every threshold: answer the most
    confident k examples, abstain on the rest, for k = 1..N.

    Returns
    -------
    dict with keys:
        coverage           : (N,) array, k / N
        selective_accuracy : (N,) array, accuracy on the k answered examples
        aurc               : float, area under the risk (= 1 - accuracy)
                             vs coverage curve (Geifman & El-Yaniv 2017).
                             Lower is better; a confidence that perfectly
                             ranks correct above incorrect reaches the
                             minimum for its accuracy, a random one sits at
                             1 - accuracy.

    Ties in confidence are broken by input order (stable sort), so the curve
    is deterministic.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    assert confidences.shape == correctness.shape and confidences.ndim == 1
    n = len(confidences)
    if n == 0:
        return {"coverage": np.array([]), "selective_accuracy": np.array([]),
                "aurc": float("nan")}
    order = np.argsort(-confidences, kind="stable")
    k = np.arange(1, n + 1)
    sel_acc = np.cumsum(correctness[order]) / k
    return {
        "coverage": k / n,
        "selective_accuracy": sel_acc,
        "aurc": float(np.mean(1.0 - sel_acc)),
    }


def top1_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Standard top-1 accuracy."""
    probs = np.asarray(probs)
    labels = np.asarray(labels, dtype=int)
    preds = probs.argmax(axis=1)
    return float((preds == labels).mean())


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-example predictive entropy in nats. Higher = less confident overall.

    Used for the OOV analysis: if the model's mean entropy on out-of-vocab
    speakers isn't substantially higher than on in-vocab speakers, the
    model fails to "know it doesn't know."
    """
    probs = np.asarray(probs, dtype=float)
    return -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)


def speaker_bootstrap_ci(
    stat_fn,
    speakers: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> np.ndarray:
    """
    Percentile CI from a speaker-level (cluster) bootstrap.

    Utterances from one speaker are strongly correlated, so resampling
    utterances would treat a 2-speaker group as hundreds of independent
    points and give falsely narrow intervals. Instead we resample whole
    speakers with replacement and keep all of each drawn speaker's
    utterances.

    With very few speakers the bootstrap distribution is coarse (2 speakers
    -> only 3 distinct resamples), so the interval is roughly the range of
    the per-speaker values. That is the honest answer, not a bug.

    Parameters
    ----------
    stat_fn  : callable(idx) -> float or (M,) array
        Computes the statistic(s) on the rows at integer positions `idx`.
    speakers : (N,) array of speaker ids, one per row.
    n_boot   : number of bootstrap resamples.
    ci       : central coverage of the interval.
    seed     : RNG seed. Using the same seed across conditions gives paired
               resamples (the same speakers drawn under each condition).

    Returns
    -------
    (2,) array [lo, hi] for a scalar statistic, else (M, 2).
    """
    speakers = np.asarray(speakers)
    uniq = np.unique(speakers)
    by_spk = [np.flatnonzero(speakers == s) for s in uniq]
    rng = np.random.default_rng(seed)

    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_spk[k] for k in pick])
        boots.append(np.atleast_1d(stat_fn(idx)))
    boots = np.asarray(boots, dtype=float)  # (n_boot, M)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boots, [100 * alpha, 100 * (1 - alpha)], axis=0)
    out = np.stack([lo, hi], axis=-1)  # (M, 2)
    return out[0] if out.shape[0] == 1 else out
