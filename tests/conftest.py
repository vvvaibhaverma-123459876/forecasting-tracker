"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from forecasting_tracker.db.database import init_db
from forecasting_tracker.db.models import Base
from forecasting_tracker.tracker import ForecastingTracker


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    sess = factory()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture()
def tracker(session):
    return ForecastingTracker(session)
