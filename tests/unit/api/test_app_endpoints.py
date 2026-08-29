"""Unit tests for FastAPI operational endpoints (Sprint 9)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from src.contracts.raw_alert import CanonicalRawAlert
from src.model.registry import ModelRegistry
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner
from src.runtime.durable_state import DurableStateManager
from src.runtime.service import LiveRBTAService
from src.api.app import create_app


def make_alert(idx: int, ts: datetime, group: str = "pam", level: int = 3, crit: int = 1) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"alert_{idx}",
        timestamp=ts,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary=group,
        rule_level=level,
        rule_id=f"550{idx % 5}",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=crit,
    )


def test_health_endpoint_liveness():
    """GET /health returns 200 OK with service identifier."""
    app = create_app(service=None)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "rbta-security-analytics"


def test_ready_endpoint_fails_503_when_no_active_model(tmp_path: Path):
    """GET /ready returns 503 Service Unavailable when model registry has no published bundle."""
    registry = ModelRegistry(base_dir=tmp_path / "models")
    app = create_app(service=None, model_registry=registry)
    client = TestClient(app)

    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["ready"] is False
    assert "Service not initialized" in data["reason"]

def test_ready_endpoint_with_service_but_no_registry():
    """GET /ready returns 200 using metadata when service exists but registry does not."""
    # Create dummy pipeline with metadata
    class DummyPipeline:
        metadata = {"model_version": "meta-v1"}
        bundle = None
    
    service = MagicMock()
    service.scoring_pipeline = DummyPipeline()
    app = create_app(service=service, model_registry=None)
    client = TestClient(app)
    
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    assert resp.json()["active_model_version"] == "meta-v1"


def test_ready_endpoint_passes_200_when_active_model_published(tmp_path: Path):
    """GET /ready returns 200 OK when model registry has active verified bundle."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="ready-v1")

    registry = ModelRegistry(base_dir=tmp_path / "models", explicit_version="ready-v1")
    registry.publish_bundle(bundle, "ready-v1")

    state_mgr = DurableStateManager(tmp_path / "state.json")
    service = LiveRBTAService(
        scoring_pipeline=ScoringPipeline(bundle),
        state_manager=state_mgr,
    )

    app = create_app(service=service, model_registry=registry)
    client = TestClient(app)

    resp = client.get("/ready")
    assert resp.status_code == 200, f"Ready endpoint failed with: {resp.json()}"
    data = resp.json()
    assert data["ready"] is True
    assert data["active_model_version"] == "ready-v1"


def test_ingest_alert_endpoint_and_outbox_ack(tmp_path: Path):
    """POST /api/v1/alerts/ingest accepts raw alert, updates stats, and allows outbox inspection and ack."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="api-v1")

    state_mgr = DurableStateManager(tmp_path / "state.json")
    service = LiveRBTAService(
        scoring_pipeline=ScoringPipeline(bundle),
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    app = create_app(service=service, api_key="test-api-key")
    client = TestClient(app)

    # Ingest 1 alert
    payload = {
        "id": "raw_100",
        "timestamp": "2026-08-28T10:00:00.000+0000",
        "agent": {"id": "001", "name": "soc-1"},
        "rule": {"id": "5501", "level": 3, "groups": ["pam"]},
    }

    # Unauthorized request
    resp_unauth = client.post("/api/v1/alerts/ingest", json=payload)
    assert resp_unauth.status_code == 401

    # Authorized request
    headers = {"Authorization": "Bearer test-api-key"}
    resp = client.post("/api/v1/alerts/ingest", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    assert resp.json()["is_duplicate"] is False

    # Stats check
    stats_resp = client.get("/runtime/stats", headers=headers)
    assert stats_resp.status_code == 200
    assert stats_resp.json()["seen_alerts_count"] == 1
    assert stats_resp.json()["active_buckets_count"] == 1

    # Ingest second alert after 20 minutes (triggers bucket finalization and outbox item)
    payload2 = {
        "id": "raw_101",
        "timestamp": "2026-08-28T10:20:00.000+0000",
        "agent": {"id": "001", "name": "soc-1"},
        "rule": {"id": "5501", "level": 3, "groups": ["pam"]},
    }
    resp2 = client.post("/api/v1/alerts/ingest", json=payload2, headers=headers)
    assert resp2.status_code == 200

    # Check outbox
    outbox_resp = client.get("/api/v1/outbox", headers=headers)
    assert outbox_resp.status_code == 200
    outbox_items = outbox_resp.json()
    assert len(outbox_items) == 1
    meta_id = outbox_items[0]["meta_id"]

    # Check meta-alert details
    meta_resp = client.get(f"/api/v1/meta-alerts/{meta_id}", headers=headers)
    assert meta_resp.status_code == 200
    assert meta_resp.json()["meta_id"] == meta_id

    # Acknowledge outbox
    ack_resp = client.post(f"/api/v1/outbox/{meta_id}/ack", headers=headers)
    assert ack_resp.status_code == 200
    assert ack_resp.json()["status"] == "acknowledged"

    # Outbox is now empty
    outbox_after = client.get("/api/v1/outbox", headers=headers).json()
    assert len(outbox_after) == 0
