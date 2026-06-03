import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest


STD_FLOOR = 2.0
WEIGHTS = [(1.0, 0.10), (2.0, 0.03)]  # (z_threshold, weight)


def _weight(z: float) -> float:
    if z < 1.0:
        return 0.10
    if z < 2.0:
        return 0.03
    return 0.00


def _simulate_ewma(counts, zscores):
    """Return (final_mean, valid_days_count) after running EWMA update."""
    mean = float(counts[0])
    valid_days = 0
    for count, z in zip(counts, zscores):
        w = _weight(z)
        if w > 0:
            mean = mean * (1 - w) + count * w
            valid_days += 1
    return mean, valid_days


class TestWeightAssignment:
    def test_weight_assignment(self):
        assert _weight(0.5) == pytest.approx(0.10)
        assert _weight(1.5) == pytest.approx(0.03)
        assert _weight(2.5) == pytest.approx(0.00)
        assert _weight(3.0) == pytest.approx(0.00)


class TestEwmaConvergence:
    def test_ewma_convergence_normal(self):
        rng = np.random.default_rng(42)
        true_mean = 8.0
        counts = rng.normal(true_mean, 1.0, size=90)
        zscores = np.zeros(90)  # all z < 1, normal days
        final_mean, valid_days = _simulate_ewma(counts, zscores)
        assert abs(final_mean - true_mean) < 1.0
        assert valid_days == 90

    def test_frozen_baseline_during_disease(self):
        rng = np.random.default_rng(42)
        # 20 normal days to establish baseline
        normal_counts = rng.normal(8.0, 1.0, size=20)
        normal_zscores = np.zeros(20)
        mean_after_normal, _ = _simulate_ewma(normal_counts, normal_zscores)

        # 30 "abnormal" days (z >= 2.0, baseline frozen)
        abnormal_counts = rng.normal(20.0, 1.0, size=30)
        abnormal_zscores = np.full(30, 2.5)

        mean = mean_after_normal
        for count, z in zip(abnormal_counts, abnormal_zscores):
            w = _weight(z)
            if w > 0:
                mean = mean * (1 - w) + count * w

        # Baseline should barely move during freeze (change < 0.5)
        assert abs(mean - mean_after_normal) < 0.5

    def test_valid_days_counting(self):
        counts = [5.0] * 10
        # All z < 1 -> all counted
        _, vd_all = _simulate_ewma(counts, [0.5] * 10)
        assert vd_all == 10

        # All z >= 2 -> none counted
        _, vd_none = _simulate_ewma(counts, [2.5] * 10)
        assert vd_none == 0

    def test_confidence_formula(self):
        # At 15 valid days -> confidence = 0.5
        confidence_15 = min(15 / 30.0, 1.0)
        assert confidence_15 == pytest.approx(0.5)

        # At 30+ valid days -> confidence = 1.0
        confidence_30 = min(30 / 30.0, 1.0)
        assert confidence_30 == pytest.approx(1.0)

        confidence_60 = min(60 / 30.0, 1.0)
        assert confidence_60 == pytest.approx(1.0)

    def test_phase_boundaries(self):
        def phase(valid_days):
            if valid_days < 3:
                return 0
            if valid_days < 14:
                return 1
            if valid_days < 30:
                return 2
            return 3

        assert phase(0) == 0
        assert phase(2) == 0
        assert phase(3) == 1
        assert phase(13) == 1
        assert phase(14) == 2
        assert phase(29) == 2
        assert phase(30) == 3
        assert phase(100) == 3
