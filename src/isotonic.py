"""
Isotonic regression: post-hoc calibration of the top-1 confidence by a
monotone step function fit on a held-out calibration set (Zadrozny & Elkan
2002).

Where temperature scaling has one parameter and a fixed functional form,
isotonic regression is non-parametric: it maps each raw confidence to the
accuracy observed at that confidence level on the calibration set, subject
only to the map being non-decreasing.

This is top-label calibration. It recalibrates the confidence of the
predicted class and leaves the argmax alone, so accuracy is unchanged — the
same property temperature scaling has. It does not produce a full
probability vector: a recalibrated confidence can fall below 1/K (e.g. a
group the model is 2% accurate on), which no distribution with that argmax
could have. Downstream code must therefore use (confidence, prediction)
pairs, not re-derive the prediction from a probability vector.

Two variants, mirroring src/temperature.py:
- fit_isotonic: one map for all examples
- fit_group_conditional_isotonic: one map per group

Usage:
    iso = fit_isotonic(cal_confs, cal_correct)
    new_confs = apply_isotonic(iso, test_confs)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(confidences: np.ndarray, correctness: np.ndarray) -> IsotonicRegression:
    """
    Fit a non-decreasing map from top-1 confidence to P(correct).

    Parameters
    ----------
    confidences : (N,) array of float in [0, 1] — raw top-1 confidence
    correctness : (N,) array of 0/1 — whether the top-1 prediction was correct

    Returns
    -------
    Fitted sklearn IsotonicRegression. Inputs outside the calibration range
    are clipped to the nearest fitted value.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    assert confidences.ndim == 1 and confidences.shape == correctness.shape
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(confidences, correctness)
    return iso


def fit_group_conditional_isotonic(
    confidences: np.ndarray,
    correctness: np.ndarray,
    groups: np.ndarray,
    min_examples_per_group: int = 50,
) -> dict:
    """
    Fit one isotonic map per group.

    Groups with fewer than `min_examples_per_group` examples map to None
    (caller must decide fallback behavior — typically the pooled map).

    Returns
    -------
    dict mapping group_label -> IsotonicRegression or None
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)
    groups = np.asarray(groups)
    assert len(confidences) == len(correctness) == len(groups)

    out = {}
    for g in np.unique(groups):
        mask = groups == g
        if int(mask.sum()) < min_examples_per_group:
            out[g] = None
            continue
        out[g] = fit_isotonic(confidences[mask], correctness[mask])
    return out


def apply_isotonic(iso: IsotonicRegression, confidences: np.ndarray) -> np.ndarray:
    """Recalibrated top-1 confidence, in [0, 1]."""
    return np.asarray(iso.predict(np.asarray(confidences, dtype=float)), dtype=float)


def apply_group_conditional_isotonic(
    confidences: np.ndarray,
    groups: np.ndarray,
    maps: dict,
    fallback: IsotonicRegression,
) -> np.ndarray:
    """
    Apply per-group isotonic maps. Groups not in `maps` (or mapped to None)
    use `fallback` (usually the pooled map).
    """
    confidences = np.asarray(confidences, dtype=float)
    groups = np.asarray(groups)
    out = np.empty_like(confidences)
    for g in np.unique(groups):
        mask = groups == g
        iso = maps.get(g)
        if iso is None:
            iso = fallback
        out[mask] = apply_isotonic(iso, confidences[mask])
    return out
