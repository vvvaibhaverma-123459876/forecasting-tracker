"""FastAPI routes for statistics and calibration."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from forecasting_tracker.api.models import (
    CalibrationData,
    RollingBrierPoint,
    SummaryStats,
)
from forecasting_tracker.db.database import get_session
from forecasting_tracker.tracker import ForecastingTracker

router = APIRouter(prefix="/stats", tags=["stats"])


def _tracker(session: Session = Depends(get_session)) -> ForecastingTracker:
    return ForecastingTracker(session)


@router.get("/summary", response_model=SummaryStats)
def summary(
    domain: Optional[str] = Query(default=None),
    tracker: ForecastingTracker = Depends(_tracker),
) -> SummaryStats:
    return SummaryStats(**tracker.summary_stats(domain=domain))


@router.get("/brier/rolling", response_model=list[RollingBrierPoint])
def rolling_brier(
    domain: Optional[str] = Query(default=None),
    tracker: ForecastingTracker = Depends(_tracker),
) -> list[RollingBrierPoint]:
    return [RollingBrierPoint(**r) for r in tracker.rolling_brier(domain=domain)]


@router.get("/calibration", response_model=CalibrationData)
def calibration(
    domain: Optional[str] = Query(default=None),
    n_bins: int = Query(default=10, ge=2, le=50),
    tracker: ForecastingTracker = Depends(_tracker),
) -> CalibrationData:
    data = tracker.calibration_data(domain=domain, n_bins=n_bins)
    return CalibrationData(**data)


@router.get("/export/csv", response_class=PlainTextResponse)
def export_csv(
    domain: Optional[str] = Query(default=None),
    tracker: ForecastingTracker = Depends(_tracker),
) -> str:
    return tracker.export_csv(domain=domain)
