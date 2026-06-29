"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import Engine as _Engine

from forecasting_tracker.api.routes.predictions import router as predictions_router
from forecasting_tracker.api.routes.stats import router as stats_router
from forecasting_tracker.db.database import _ensure_engine, init_db


def create_app(engine: "_Engine | None" = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        eng = engine if engine is not None else _ensure_engine()
        init_db(eng)
        yield

    app = FastAPI(
        title="Forecasting Tracker",
        description="Personal forecasting / prediction market tool with Brier scoring and calibration.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions_router)
    app.include_router(stats_router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
