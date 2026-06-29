"""Pydantic v2 request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionCreate(BaseModel):
    statement: str = Field(..., min_length=1, max_length=2000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    deadline: datetime
    domain: str = Field(default="general", max_length=100)
    notes: Optional[str] = Field(default=None, max_length=5000)

    @field_validator("domain", mode="before")
    @classmethod
    def normalise_domain(cls, v: str) -> str:
        return v.strip().lower() or "general"


class PredictionResolve(BaseModel):
    outcome: bool


class PredictionOut(BaseModel):
    id: int
    statement: str
    confidence: float
    deadline: datetime
    domain: str
    status: str
    outcome: Optional[bool]
    brier_score: Optional[float]
    created_at: datetime
    resolved_at: Optional[datetime]
    notes: Optional[str]

    model_config = {"from_attributes": True}


class SummaryStats(BaseModel):
    total: int
    open: int
    resolved: int
    mean_brier: Optional[float]
    ece: Optional[float]


class RollingBrierPoint(BaseModel):
    prediction_id: int
    resolved_at: Optional[str]
    brier_score: Optional[float]
    rolling_mean: float
    n: int


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: Optional[float]
    count: int


class CalibrationData(BaseModel):
    bins: list[CalibrationBin]
    plotted_confidence: list[float]
    plotted_accuracy: list[float]
    perfect_x: list[float]
    perfect_y: list[float]
