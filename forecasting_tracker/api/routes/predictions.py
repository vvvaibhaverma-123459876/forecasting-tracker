"""FastAPI routes for prediction CRUD."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from forecasting_tracker.api.models import (
    PredictionCreate,
    PredictionOut,
    PredictionResolve,
)
from forecasting_tracker.db.database import get_session
from forecasting_tracker.db.models import PredictionStatus
from forecasting_tracker.tracker import (
    DeadlineNotPassed,
    ForecastingError,
    ForecastingTracker,
    PredictionAlreadyResolved,
    PredictionNotFound,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _tracker(session: Session = Depends(get_session)) -> ForecastingTracker:
    return ForecastingTracker(session)


@router.post("/", response_model=PredictionOut, status_code=201)
def create_prediction(
    body: PredictionCreate,
    tracker: ForecastingTracker = Depends(_tracker),
) -> PredictionOut:
    try:
        pred = tracker.add_prediction(
            statement=body.statement,
            confidence=body.confidence,
            deadline=body.deadline,
            domain=body.domain,
            notes=body.notes,
        )
    except ForecastingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionOut.model_validate(pred)


@router.get("/", response_model=list[PredictionOut])
def list_predictions(
    domain: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    tracker: ForecastingTracker = Depends(_tracker),
) -> list[PredictionOut]:
    status_enum = None
    if status:
        try:
            status_enum = PredictionStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Choose from: {[s.value for s in PredictionStatus]}",
            )
    preds = tracker.list_predictions(domain=domain, status=status_enum, limit=limit, offset=offset)
    return [PredictionOut.model_validate(p) for p in preds]


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: int,
    tracker: ForecastingTracker = Depends(_tracker),
) -> PredictionOut:
    try:
        pred = tracker.get_prediction(prediction_id)
    except PredictionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictionOut.model_validate(pred)


@router.post("/{prediction_id}/resolve", response_model=PredictionOut)
def resolve_prediction(
    prediction_id: int,
    body: PredictionResolve,
    tracker: ForecastingTracker = Depends(_tracker),
) -> PredictionOut:
    try:
        pred = tracker.resolve_prediction(prediction_id, body.outcome)
    except PredictionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionAlreadyResolved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeadlineNotPassed as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionOut.model_validate(pred)


@router.delete("/{prediction_id}", status_code=204)
def delete_prediction(
    prediction_id: int,
    tracker: ForecastingTracker = Depends(_tracker),
) -> Response:
    try:
        tracker.delete_prediction(prediction_id)
    except PredictionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
