"""Tests for core business logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from forecasting_tracker.tracker import (
    DeadlineNotPassed,
    ForecastingError,
    ForecastingTracker,
    PredictionAlreadyResolved,
    PredictionNotFound,
)
from forecasting_tracker.db.models import PredictionStatus


def past(days: int = 1) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)


def future(days: int = 10) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=days)


class TestAddPrediction:
    def test_basic_add(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction(
            "Test prediction", confidence=0.7, deadline=future()
        )
        assert pred.id is not None
        assert pred.status == PredictionStatus.OPEN
        assert pred.confidence == 0.7

    def test_invalid_confidence(self, tracker: ForecastingTracker):
        with pytest.raises(ForecastingError):
            tracker.add_prediction("X", confidence=1.5, deadline=future())

    def test_empty_statement(self, tracker: ForecastingTracker):
        with pytest.raises(ForecastingError):
            tracker.add_prediction("   ", confidence=0.5, deadline=future())

    def test_domain_normalised(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction("X", confidence=0.5, deadline=future(), domain="TECH")
        assert pred.domain == "tech"


class TestResolvePrediction:
    def test_resolve_true(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction("X", confidence=0.8, deadline=past())
        tracker.session.flush()
        resolved = tracker.resolve_prediction(pred.id, True)
        assert resolved.status == PredictionStatus.RESOLVED_TRUE
        assert resolved.outcome is True
        assert resolved.brier_score == pytest.approx(0.04)  # (0.8-1)^2

    def test_resolve_false(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction("X", confidence=0.3, deadline=past())
        tracker.session.flush()
        resolved = tracker.resolve_prediction(pred.id, False)
        assert resolved.status == PredictionStatus.RESOLVED_FALSE
        assert resolved.brier_score == pytest.approx(0.09)  # (0.3-0)^2

    def test_deadline_not_passed(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction("X", confidence=0.5, deadline=future())
        tracker.session.flush()
        with pytest.raises(DeadlineNotPassed):
            tracker.resolve_prediction(pred.id, True)

    def test_double_resolve(self, tracker: ForecastingTracker):
        pred = tracker.add_prediction("X", confidence=0.5, deadline=past())
        tracker.session.flush()
        tracker.resolve_prediction(pred.id, True)
        with pytest.raises(PredictionAlreadyResolved):
            tracker.resolve_prediction(pred.id, False)

    def test_not_found(self, tracker: ForecastingTracker):
        with pytest.raises(PredictionNotFound):
            tracker.resolve_prediction(9999, True)


class TestListAndFilter:
    def test_filter_domain(self, tracker: ForecastingTracker):
        tracker.add_prediction("A", confidence=0.5, deadline=future(), domain="tech")
        tracker.add_prediction("B", confidence=0.5, deadline=future(), domain="finance")
        tracker.session.flush()
        tech = tracker.list_predictions(domain="tech")
        assert len(tech) == 1
        assert tech[0].domain == "tech"

    def test_filter_status(self, tracker: ForecastingTracker):
        tracker.add_prediction("A", confidence=0.5, deadline=past())
        p2 = tracker.add_prediction("B", confidence=0.5, deadline=future())
        tracker.session.flush()
        open_preds = tracker.list_predictions(status=PredictionStatus.OPEN)
        assert all(p.status == PredictionStatus.OPEN for p in open_preds)


class TestStats:
    def _seed(self, tracker: ForecastingTracker):
        preds = []
        for conf, outcome in [(0.9, True), (0.8, True), (0.3, False), (0.6, True)]:
            p = tracker.add_prediction("X", confidence=conf, deadline=past())
            tracker.session.flush()
            tracker.resolve_prediction(p.id, outcome)
            preds.append(p)
        return preds

    def test_summary(self, tracker: ForecastingTracker):
        self._seed(tracker)
        stats = tracker.summary_stats()
        assert stats["total"] == 4
        assert stats["resolved"] == 4
        assert stats["open"] == 0
        assert stats["mean_brier"] is not None

    def test_rolling_brier_length(self, tracker: ForecastingTracker):
        self._seed(tracker)
        rolling = tracker.rolling_brier()
        assert len(rolling) == 4

    def test_ece_range(self, tracker: ForecastingTracker):
        self._seed(tracker)
        ece = tracker.ece()
        assert 0.0 <= ece <= 1.0

    def test_calibration_data(self, tracker: ForecastingTracker):
        self._seed(tracker)
        data = tracker.calibration_data()
        assert "bins" in data


class TestExport:
    def test_csv_contains_header(self, tracker: ForecastingTracker):
        tracker.add_prediction("X", confidence=0.5, deadline=future())
        tracker.session.flush()
        csv = tracker.export_csv()
        assert "id" in csv
        assert "confidence" in csv
        assert "statement" in csv

    def test_csv_rows(self, tracker: ForecastingTracker):
        tracker.add_prediction("A", confidence=0.5, deadline=future())
        tracker.add_prediction("B", confidence=0.7, deadline=future())
        tracker.session.flush()
        csv = tracker.export_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
