import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import json

from src.api.app import create_app
from src.runtime.service import LiveRBTAService
from src.runtime.replay_controller import ReplayController
from src.model.scoring_pipeline import ScoringPipeline
from src.contracts.scored_meta_alert import ScoredMetaAlert

class DummyPipeline:
    def __init__(self):
        self.metadata = {"model_version": "test-v1", "score_calibration_version": "test-v1"}
        self.threshold = type('obj', (object,), {'threshold': 0.8})()

@pytest.fixture
def service():
    pipeline = DummyPipeline()
    svc = LiveRBTAService(scoring_pipeline=pipeline, adaptive=False)

    # Add dummy history for testing
    meta = ScoredMetaAlert(
        meta_id=1,
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="syslog",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        alert_count=5,
        max_severity=3,
        mitre_tactics=(),
        seven_features={},
        raw_model_score=0.9,
        anomaly_score=0.9,
        threshold_used=0.8,
        decision="CRITICAL",
        action="ESCALATE",
        escalate=True,
        model_version="test-v1",
        feature_schema_version="1.0",
        score_calibration_version="test-v1",
        source_alert_ids=("a1", "a2"),
        metadata={}
    )
    svc.finalized_history.append(meta)

    return svc

@pytest.fixture
def replay_controller(tmp_path):
    d = tmp_path / "test_datasets"
    d.mkdir(parents=True, exist_ok=True)
    file_path = d / "alerts.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(20):
            alert = {
                "wazuh_alert_id": f"id-{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": "001",
                "agent_name": "agent-1",
                "rule_group_primary": "syslog",
                "rule_level": 3,
                "rule_id": "1000",
                "mitre_tactics": [],
                "agent_criticality": 1,
                "metadata": {"rule_description": "test", "rule_groups_all": ["syslog"]},
            }
            f.write(json.dumps(alert) + "\n")
    return ReplayController(scoring_pipeline=DummyPipeline(), replay_data_dir=d)

@pytest.fixture
def client(service, replay_controller):
    app = create_app(service=service, api_key="test-key", replay_controller=replay_controller)
    return TestClient(app)

def test_dashboard_summary_returns_kpis(client):
    response = client.get("/api/v1/dashboard/summary", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["meta_alert_count"] == 1
    assert data["escalate_count"] == 1
    assert data["suppress_count"] == 0

def test_dashboard_agents_returns_list(client):
    response = client.get("/api/v1/dashboard/agents", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_dashboard_buckets_returns_list(client):
    response = client.get("/api/v1/dashboard/buckets", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_dashboard_system_returns_info(client):
    response = client.get("/api/v1/dashboard/system", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "ok"
    assert data["runtime_ready"] is True

def test_meta_alerts_list_paginated(client):
    response = client.get("/api/v1/meta-alerts?page=1&page_size=10", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["meta_id"] == 1

def test_meta_alerts_raw_alerts_paginated(client):
    response = client.get("/api/v1/meta-alerts/1/raw-alerts", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["meta_id"] == 1
    # Without raw evidence store, items should be empty but return schema structure
    assert data["items"] == []

def test_raw_alert_detail(client):
    # App is created without raw_evidence_store here so it should 503
    response = client.get("/api/v1/raw-alerts/a1", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 503

def test_replay_status_idle(client):
    response = client.get("/api/v1/replay/status", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "IDLE"

def test_replay_lifecycle_start_pause_resume_stop(client):
    response = client.post("/api/v1/replay/start?speed=1.0", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] in ("RUNNING", "COMPLETED")

    response = client.post("/api/v1/replay/pause", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] in ("PAUSED", "COMPLETED")

    response = client.post("/api/v1/replay/resume", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] in ("RUNNING", "COMPLETED")

    response = client.post("/api/v1/replay/stop", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"

    response = client.post("/api/v1/replay/reset", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "IDLE"
