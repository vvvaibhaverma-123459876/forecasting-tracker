"""Tests for FastAPI routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from forecasting_tracker.api.app import create_app
from forecasting_tracker.db.database import get_session, init_db
from forecasting_tracker.db.models import Base


def past_iso(days: int = 2) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None).isoformat()


def future_iso(days: int = 10) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(tzinfo=None).isoformat()


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    TestSessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_session():
        session = TestSessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app = create_app(engine=engine)
    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as c:
        yield c

    Base.metadata.drop_all(engine)


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPredictionsEndpoints:
    def test_create(self, client: TestClient):
        resp = client.post(
            "/predictions/",
            json={"statement": "Test", "confidence": 0.7, "deadline": future_iso(), "domain": "tech"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] is not None
        assert data["status"] == "open"

    def test_create_invalid_confidence(self, client: TestClient):
        resp = client.post(
            "/predictions/",
            json={"statement": "X", "confidence": 1.5, "deadline": future_iso()},
        )
        assert resp.status_code == 422

    def test_list(self, client: TestClient):
        client.post(
            "/predictions/",
            json={"statement": "A", "confidence": 0.5, "deadline": future_iso()},
        )
        resp = client.get("/predictions/")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get(self, client: TestClient):
        create_resp = client.post(
            "/predictions/",
            json={"statement": "B", "confidence": 0.6, "deadline": future_iso()},
        )
        pid = create_resp.json()["id"]
        resp = client.get(f"/predictions/{pid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pid

    def test_get_not_found(self, client: TestClient):
        resp = client.get("/predictions/9999")
        assert resp.status_code == 404

    def test_resolve(self, client: TestClient):
        create_resp = client.post(
            "/predictions/",
            json={"statement": "C", "confidence": 0.9, "deadline": past_iso()},
        )
        pid = create_resp.json()["id"]
        resp = client.post(f"/predictions/{pid}/resolve", json={"outcome": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "resolved_true"
        assert data["brier_score"] == pytest.approx(0.01)

    def test_resolve_deadline_not_passed(self, client: TestClient):
        create_resp = client.post(
            "/predictions/",
            json={"statement": "D", "confidence": 0.5, "deadline": future_iso()},
        )
        pid = create_resp.json()["id"]
        resp = client.post(f"/predictions/{pid}/resolve", json={"outcome": True})
        assert resp.status_code == 422

    def test_delete(self, client: TestClient):
        create_resp = client.post(
            "/predictions/",
            json={"statement": "E", "confidence": 0.5, "deadline": future_iso()},
        )
        pid = create_resp.json()["id"]
        resp = client.delete(f"/predictions/{pid}")
        assert resp.status_code == 204
        assert client.get(f"/predictions/{pid}").status_code == 404

    def test_filter_by_status(self, client: TestClient):
        client.post(
            "/predictions/",
            json={"statement": "Open", "confidence": 0.5, "deadline": future_iso()},
        )
        resp = client.get("/predictions/?status=open")
        assert resp.status_code == 200
        for p in resp.json():
            assert p["status"] == "open"


class TestStatsEndpoints:
    def _setup(self, client: TestClient):
        for conf, deadline, resolve, outcome in [
            (0.8, past_iso(), True, True),
            (0.3, past_iso(), True, False),
            (0.6, future_iso(), False, None),
        ]:
            resp = client.post(
                "/predictions/",
                json={"statement": "X", "confidence": conf, "deadline": deadline},
            )
            pid = resp.json()["id"]
            if resolve:
                client.post(f"/predictions/{pid}/resolve", json={"outcome": outcome})

    def test_summary(self, client: TestClient):
        self._setup(client)
        resp = client.get("/stats/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["resolved"] == 2

    def test_rolling_brier(self, client: TestClient):
        self._setup(client)
        resp = client.get("/stats/brier/rolling")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_calibration(self, client: TestClient):
        self._setup(client)
        resp = client.get("/stats/calibration")
        assert resp.status_code == 200
        data = resp.json()
        assert "bins" in data

    def test_export_csv(self, client: TestClient):
        self._setup(client)
        resp = client.get("/stats/export/csv")
        assert resp.status_code == 200
        assert "confidence" in resp.text
