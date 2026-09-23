"""
Temperature scaling: post-hoc calibration by dividing logits by a scalar T
before softmax, fit by minimizing NLL on a held-out calibration set.

Two variants:
- fit_single_temperature: one T for all examples (Guo et al. 2017)
- fit_group_conditional_temperatures: one T per group

Both leave argmax (and therefore accuracy) unchanged. Only confidence shifts.

Usage:
    T = fit_single_temperature(cal_logits, cal_labels)
    new_probs = apply_temperature(test_logits, T)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_single_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
    init_t: float = 1.0,
    t_min: float = 0.05,
    t_max: float = 100.0,
) -> float:
    """
    Fit a single temperature scalar T by minimizing NLL via LBFGS.

    Parameters
    ----------
    logits : (N, K) array — pre-softmax model outputs
    labels : (N,) int array — true class indices in [0, K)
    max_iter : LBFGS iterations
    init_t : initial value for T
    t_min, t_max : float — the fitted T is clamped to [t_min, t_max]

    Returns
    -------
    T : float — fitted temperature, in [t_min, t_max]

    A returned T equal to t_max means the NLL had no finite minimum: the
    likelihood-optimal recalibration for this data is the uniform
    distribution. Callers must treat that as degeneracy, not as a fit.
    """
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    assert logits.ndim == 2 and labels.ndim == 1
    assert logits.shape[0] == labels.shape[0]

    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)

    # Optimize log(T) so T stays positive without constraints.
    #
    # The line search is load-bearing, not a tuning detail. Without it LBFGS
    # takes fixed steps of size `lr` and silently stops far short of the
    # optimum, and how far short depends on the scale of the logits — so the
    # fitted T stops being a function of the data alone. The identity
    # T*(S * logits) == S * T*(logits), which must hold exactly, is the test;
    # it fails without a line search and holds to 4 decimals with one.
    log_t = torch.tensor([float(np.log(init_t))], requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(torch.exp(log_t).detach().item())

    # NLL is unimodal in log T, so clamping the unconstrained minimizer to the
    # box gives the constrained minimizer exactly — no need to refit.
    return float(min(max(t, t_min), t_max))


def fit_group_conditional_temperatures(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    max_iter: int = 50,
    init_t: float = 1.0,
    min_examples_per_group: int = 50,
    t_min: float = 0.05,
    t_max: float = 100.0,
) -> dict:
    """
    Fit one temperature per group.

    Groups with fewer than `min_examples_per_group` examples will have their
    T set to NaN (caller must decide fallback behavior — typically use the
    pooled global T for those).

    Parameters
    ----------
    logits : (N, K) array
    labels : (N,) int array
    groups : (N,) array (any hashable type) — group label per example
    min_examples_per_group : int

    Returns
    -------
    dict mapping group_label -> T (float, NaN if too few examples)
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    assert len(logits) == len(labels) == len(groups)

    out = {}
    for g in np.unique(groups):
        mask = groups == g
        n = int(mask.sum())
        if n < min_examples_per_group:
            out[g] = float("nan")
            continue
        out[g] = fit_single_temperature(
            logits[mask], labels[mask], max_iter=max_iter, init_t=init_t,
            t_min=t_min, t_max=t_max,
        )
    return out


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature T to logits and return softmax probabilities.

    T > 1 -> flatter (less confident).
    T < 1 -> sharper (more confident).
    """
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / float(T)
    # numerically stable softmax
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


def apply_group_conditional_temperature(
    logits: np.ndarray,
    groups: np.ndarray,
    temperatures: dict,
    fallback_t: float = 1.0,
) -> np.ndarray:
    """
    Apply per-group temperatures. Groups not in `temperatures` (or with NaN T)
    fall back to `fallback_t` (usually the global pooled T).

    Returns (N, K) probability array.
    """
    logits = np.asarray(logits)
    groups = np.asarray(groups)
    n = len(logits)
    probs = np.zeros_like(logits, dtype=np.float64)
    for i in range(n):
        g = groups[i]
        T = temperatures.get(g, fallback_t)
        if T is None or (isinstance(T, float) and np.isnan(T)):
            T = fallback_t
        probs[i : i + 1] = apply_temperature(logits[i : i + 1], T)
    return probs
