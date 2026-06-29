"""SQLAlchemy ORM models for the forecasting tracker."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionStatus(str, enum.Enum):
    OPEN = "open"
    RESOLVED_TRUE = "resolved_true"
    RESOLVED_FALSE = "resolved_false"


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    deadline: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False, default="general")
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus), nullable=False, default=PredictionStatus.OPEN
    )
    outcome: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    __table_args__ = (
        Index("ix_predictions_domain", "domain"),
        Index("ix_predictions_status", "status"),
        Index("ix_predictions_deadline", "deadline"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} confidence={self.confidence} "
            f"status={self.status} domain={self.domain!r}>"
        )
