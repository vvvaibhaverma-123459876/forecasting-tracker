"""SQLite database setup and session management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from forecasting_tracker.db.models import Base

DEFAULT_DB_PATH = Path.home() / ".forecasting_tracker" / "predictions.db"


def get_db_url(db_path: str | Path | None = None) -> str:
    url = os.environ.get("FORECASTING_TRACKER_DB_URL")
    if url:
        return url
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def create_db_engine(db_url: str | None = None) -> Engine:
    url = db_url or get_db_url()
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Module-level convenience — lazily initialised
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _ensure_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_db_engine()
        init_db(_engine)
    return _engine


def get_session() -> Generator[Session, None, None]:
    """Yield a database session; suitable for FastAPI Depends."""
    factory = get_session_factory(_ensure_engine())
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
