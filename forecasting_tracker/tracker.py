"""Core business logic for the forecasting tracker."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from forecasting_tracker.db.models import Prediction, PredictionStatus
from forecasting_tracker.scoring.brier import brier_score, rolling_brier_average
from forecasting_tracker.scoring.calibration import calibration_curve_data
from forecasting_tracker.scoring.ece import expected_calibration_error


class ForecastingError(Exception):
    """Base exception for tracker errors."""


class PredictionNotFound(ForecastingError):
    def __init__(self, prediction_id: int) -> None:
        super().__init__(f"Prediction {prediction_id} not found")
        self.prediction_id = prediction_id


class PredictionAlreadyResolved(ForecastingError):
    def __init__(self, prediction_id: int) -> None:
        super().__init__(f"Prediction {prediction_id} is already resolved")
        self.prediction_id = prediction_id


class DeadlineNotPassed(ForecastingError):
    def __init__(self, prediction_id: int, deadline: datetime) -> None:
        super().__init__(
            f"Prediction {prediction_id} deadline {deadline.isoformat()} has not passed yet"
        )
        self.prediction_id = prediction_id
        self.deadline = deadline


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ForecastingTracker:
    """High-level API around the prediction database."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_prediction(
        self,
        statement: str,
        confidence: float,
        deadline: datetime,
        domain: str = "general",
        notes: str | None = None,
    ) -> Prediction:
        """Register a new prediction.

        The prediction is locked on creation; the confidence value cannot be
        changed once saved (pre-registration enforcement).
        """
        if not 0.0 <= confidence <= 1.0:
            raise ForecastingError(f"confidence must be in [0, 1], got {confidence}")
        if not statement.strip():
            raise ForecastingError("statement must not be empty")

        prediction = Prediction(
            statement=statement.strip(),
            confidence=confidence,
            deadline=deadline,
            domain=domain.strip().lower() or "general",
            notes=notes,
            status=PredictionStatus.OPEN,
            created_at=_now_utc(),
        )
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def get_prediction(self, prediction_id: int) -> Prediction:
        pred = self.session.get(Prediction, prediction_id)
        if pred is None:
            raise PredictionNotFound(prediction_id)
        return pred

    def list_predictions(
        self,
        domain: str | None = None,
        status: PredictionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Prediction]:
        q = self.session.query(Prediction)
        if domain:
            q = q.filter(Prediction.domain == domain.strip().lower())
        if status:
            q = q.filter(Prediction.status == status)
        return q.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()

    def resolve_prediction(self, prediction_id: int, outcome: bool) -> Prediction:
        """Resolve a prediction as true or false.

        Raises DeadlineNotPassed if the deadline has not yet passed.
        Raises PredictionAlreadyResolved if already resolved.
        """
        pred = self.get_prediction(prediction_id)
        if pred.status != PredictionStatus.OPEN:
            raise PredictionAlreadyResolved(prediction_id)

        now = _now_utc()
        if pred.deadline > now:
            raise DeadlineNotPassed(prediction_id, pred.deadline)

        pred.outcome = outcome
        pred.status = PredictionStatus.RESOLVED_TRUE if outcome else PredictionStatus.RESOLVED_FALSE
        pred.brier_score = brier_score(pred.confidence, outcome)
        pred.resolved_at = now
        self.session.flush()
        return pred

    def delete_prediction(self, prediction_id: int) -> None:
        pred = self.get_prediction(prediction_id)
        self.session.delete(pred)
        self.session.flush()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _resolved_predictions(self, domain: str | None = None) -> list[Prediction]:
        q = self.session.query(Prediction).filter(
            Prediction.status.in_(
                [PredictionStatus.RESOLVED_TRUE, PredictionStatus.RESOLVED_FALSE]
            )
        )
        if domain:
            q = q.filter(Prediction.domain == domain.strip().lower())
        return q.order_by(Prediction.resolved_at).all()

    def rolling_brier(self, domain: str | None = None) -> list[dict]:
        """Return a time-series of rolling mean Brier scores."""
        resolved = self._resolved_predictions(domain)
        if not resolved:
            return []

        scores = []
        result = []
        for pred in resolved:
            scores.append(pred.brier_score)  # type: ignore[arg-type]
            result.append(
                {
                    "prediction_id": pred.id,
                    "resolved_at": pred.resolved_at.isoformat() if pred.resolved_at else None,
                    "brier_score": pred.brier_score,
                    "rolling_mean": rolling_brier_average(scores),
                    "n": len(scores),
                }
            )
        return result

    def ece(self, domain: str | None = None, n_bins: int = 10) -> float:
        """Compute ECE over all resolved predictions."""
        resolved = self._resolved_predictions(domain)
        if not resolved:
            return 0.0
        confidences = [p.confidence for p in resolved]
        outcomes = [p.outcome for p in resolved]  # type: ignore[misc]
        return expected_calibration_error(confidences, outcomes, n_bins=n_bins)

    def calibration_data(self, domain: str | None = None, n_bins: int = 10) -> dict:
        """Return calibration curve data for all resolved predictions."""
        resolved = self._resolved_predictions(domain)
        if not resolved:
            return {"bins": [], "plotted_confidence": [], "plotted_accuracy": [],
                    "perfect_x": [0.0, 1.0], "perfect_y": [0.0, 1.0]}
        confidences = [p.confidence for p in resolved]
        outcomes = [p.outcome for p in resolved]  # type: ignore[misc]
        return calibration_curve_data(confidences, outcomes, n_bins=n_bins)

    def summary_stats(self, domain: str | None = None) -> dict:
        """Return summary statistics for the tracker."""
        all_preds = self.list_predictions(domain=domain, limit=10000)
        resolved = [p for p in all_preds if p.status != PredictionStatus.OPEN]
        open_preds = [p for p in all_preds if p.status == PredictionStatus.OPEN]
        brier_scores = [p.brier_score for p in resolved if p.brier_score is not None]
        return {
            "total": len(all_preds),
            "open": len(open_preds),
            "resolved": len(resolved),
            "mean_brier": rolling_brier_average(brier_scores) if brier_scores else None,
            "ece": self.ece(domain) if resolved else None,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, domain: str | None = None) -> str:
        """Return all predictions as a CSV string."""
        preds = self.list_predictions(domain=domain, limit=100_000)
        output = io.StringIO()
        fieldnames = [
            "id", "statement", "confidence", "deadline", "domain",
            "status", "outcome", "brier_score", "created_at", "resolved_at", "notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for p in preds:
            writer.writerow(
                {
                    "id": p.id,
                    "statement": p.statement,
                    "confidence": p.confidence,
                    "deadline": p.deadline.isoformat() if p.deadline else "",
                    "domain": p.domain,
                    "status": p.status.value,
                    "outcome": p.outcome,
                    "brier_score": p.brier_score,
                    "created_at": p.created_at.isoformat() if p.created_at else "",
                    "resolved_at": p.resolved_at.isoformat() if p.resolved_at else "",
                    "notes": p.notes or "",
                }
            )
        return output.getvalue()
