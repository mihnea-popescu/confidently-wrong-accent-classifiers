"""
Unit tests for metrics.py.

The most important test is ECE on a synthetic case where we can compute the
correct answer by hand. Per the prereg, we cross-check this against
torchmetrics in 04_baseline_eval.py — but the by-hand test catches bugs
without requiring a heavy dependency.
"""

import numpy as np
import pytest

from src.metrics import (
    expected_calibration_error,
    reliability_curve,
    brier_score_multiclass,
    selective_accuracy_at_coverage,
    top1_accuracy,
    predictive_entropy,
)


def test_ece_perfect_calibration_is_zero():
    # 100 examples all at confidence 0.7, exactly 70 correct -> ECE = 0
    confs = np.full(100, 0.7)
    correct = np.array([1] * 70 + [0] * 30)
    ece = expected_calibration_error(confs, correct, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_completely_wrong_calibration():
    # 100 examples at confidence 0.99, all wrong -> ECE = 0.99
    confs = np.full(100, 0.99)
    correct = np.zeros(100)
    ece = expected_calibration_error(confs, correct, n_bins=15)
    assert ece == pytest.approx(0.99, abs=1e-9)


def test_ece_handcomputed_two_bins():
    """
    Two well-separated bins with known ECE.
    Bin A (low conf): 40 examples, conf=0.2, 8 correct -> acc=0.20, conf=0.20
    Bin B (high conf): 60 examples, conf=0.9, 30 correct -> acc=0.50, conf=0.90
    Expected ECE = (40/100)*|0.20-0.20| + (60/100)*|0.50-0.90|
                 = 0 + 0.6 * 0.40 = 0.24
    """
    confs = np.concatenate([np.full(40, 0.2), np.full(60, 0.9)])
    correct = np.concatenate(
        [np.array([1] * 8 + [0] * 32), np.array([1] * 30 + [0] * 30)]
    )
    ece = expected_calibration_error(confs, correct, n_bins=15)
    assert ece == pytest.approx(0.24, abs=1e-9)


def test_ece_empty_input_is_nan():
    ece = expected_calibration_error(np.array([]), np.array([]), n_bins=15)
    assert np.isnan(ece)


def test_reliability_curve_shapes():
    confs = np.random.RandomState(0).uniform(0, 1, 200)
    correct = (np.random.RandomState(1).uniform(0, 1, 200) < confs).astype(int)
    out = reliability_curve(confs, correct, n_bins=10)
    assert out["bin_centers"].shape == (10,)
    assert out["bin_counts"].shape == (10,)
    assert out["bin_accs"].shape == (10,)
    assert out["bin_confs"].shape == (10,)
    # bin counts must sum to N
    assert out["bin_counts"].sum() == 200


def test_brier_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = np.array([0, 1])
    assert brier_score_multiclass(probs, labels) == pytest.approx(0.0)


def test_brier_uniform_prediction():
    # K=3 classes, uniform probability, label = 0
    # squared error = (1 - 1/3)^2 + (0 - 1/3)^2 + (0 - 1/3)^2 = 6/9 = 0.6667
    probs = np.full((1, 3), 1 / 3)
    labels = np.array([0])
    assert brier_score_multiclass(probs, labels) == pytest.approx(2 / 3, abs=1e-6)


def test_selective_accuracy_picks_high_conf():
    # Top-2 most confident (0.9, 0.8) are both correct;
    # bottom one (0.1) is wrong. Coverage 0.66 -> accuracy 1.0.
    confs = np.array([0.9, 0.8, 0.1])
    correct = np.array([1, 1, 0])
    assert selective_accuracy_at_coverage(confs, correct, coverage=2 / 3) == pytest.approx(1.0)


def test_top1_accuracy():
    probs = np.array([[0.1, 0.9], [0.6, 0.4]])
    labels = np.array([1, 0])
    assert top1_accuracy(probs, labels) == 1.0


def test_predictive_entropy_one_hot_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    h = predictive_entropy(probs)
    assert h[0] == pytest.approx(0.0, abs=1e-9)


def test_predictive_entropy_uniform_is_log_k():
    # Uniform over K -> entropy = ln(K)
    k = 4
    probs = np.full((1, k), 1.0 / k)
    h = predictive_entropy(probs)
    assert h[0] == pytest.approx(np.log(k), abs=1e-9)
