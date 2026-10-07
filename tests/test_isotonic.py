"""Unit tests for isotonic.py."""

import numpy as np
import pytest

from src.isotonic import (
    apply_group_conditional_isotonic,
    apply_isotonic,
    fit_group_conditional_isotonic,
    fit_isotonic,
)


def test_isotonic_maps_constant_confidence_to_accuracy():
    # Every example at confidence 0.9, 30% correct -> map outputs 0.3.
    confs = np.full(100, 0.9)
    correct = np.array([1] * 30 + [0] * 70)
    iso = fit_isotonic(confs, correct)
    assert apply_isotonic(iso, [0.9])[0] == pytest.approx(0.3)


def test_isotonic_is_monotone_and_clipped():
    rng = np.random.default_rng(0)
    confs = rng.random(500)
    correct = (rng.random(500) < confs).astype(int)
    iso = fit_isotonic(confs, correct)
    out = apply_isotonic(iso, np.linspace(-0.5, 1.5, 50))
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0 and out.max() <= 1


def test_group_conditional_isotonic_fallback():
    confs = np.full(120, 0.8)
    correct = np.array([1] * 60 + [0] * 40 + [1] * 20)
    groups = np.array(["a"] * 100 + ["b"] * 20)  # b is below min size
    maps = fit_group_conditional_isotonic(confs, correct, groups, min_examples_per_group=50)
    assert maps["b"] is None
    pooled = fit_isotonic(confs, correct)
    test_groups = np.array(["a", "b", "unseen"])
    out = apply_group_conditional_isotonic(np.full(3, 0.8), test_groups, maps, pooled)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(80 / 120)
    assert out[2] == pytest.approx(80 / 120)
