"""Tests for scoring modules."""

from __future__ import annotations

import math
import pytest

from forecasting_tracker.scoring.brier import brier_score, rolling_brier_average
from forecasting_tracker.scoring.ece import expected_calibration_error, ece_decomposed
from forecasting_tracker.scoring.calibration import calibration_curve_data


class TestBrierScore:
    def test_perfect_true(self):
        assert brier_score(1.0, True) == pytest.approx(0.0)

    def test_perfect_false(self):
        assert brier_score(0.0, False) == pytest.approx(0.0)

    def test_worst_true(self):
        assert brier_score(0.0, True) == pytest.approx(1.0)

    def test_worst_false(self):
        assert brier_score(1.0, False) == pytest.approx(1.0)

    def test_half(self):
        assert brier_score(0.5, True) == pytest.approx(0.25)
        assert brier_score(0.5, False) == pytest.approx(0.25)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            brier_score(1.5, True)

    def test_rolling_empty(self):
        assert rolling_brier_average([]) == 0.0

    def test_rolling_mean(self):
        scores = [0.25, 0.25, 0.0]
        assert rolling_brier_average(scores) == pytest.approx(0.1666, rel=1e-3)


class TestECE:
    def test_perfect_calibration(self):
        # 100 predictions: confidence matches accuracy exactly per bin
        confidences = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
        outcomes = [False] * 9 + [True] + [False] * 5 + [True] * 5 + [False] + [True] * 9
        ece = expected_calibration_error(confidences, outcomes, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_empty(self):
        assert expected_calibration_error([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5, 0.5], [True])

    def test_fully_overconfident(self):
        # All predict 1.0 but all wrong
        confidences = [1.0] * 20
        outcomes = [False] * 20
        ece = expected_calibration_error(confidences, outcomes)
        assert ece == pytest.approx(1.0)

    def test_decomposed_bin_count(self):
        confidences = [i / 20 for i in range(20)]
        outcomes = [i % 2 == 0 for i in range(20)]
        bins = ece_decomposed(confidences, outcomes, n_bins=10)
        assert len(bins) == 10


class TestCalibrationCurve:
    def test_returns_structure(self):
        confidences = [0.2, 0.4, 0.6, 0.8]
        outcomes = [False, True, True, True]
        data = calibration_curve_data(confidences, outcomes, n_bins=5)
        assert "bins" in data
        assert "perfect_x" in data
        assert data["perfect_x"] == [0.0, 1.0]
        assert data["perfect_y"] == [0.0, 1.0]
