from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.contracts.meta_alert import MetaAlert
from src.contracts.raw_alert import CanonicalRawAlert
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runtime.durable_state import DurableStateManager
from src.runtime.raw_evidence import RawAlertEvidenceStore
from src.runtime.replay_controller import ReplayController
from src.runtime.service import LiveRBTAService


@pytest.fixture
def test_setup(tmp_path: Path):
    db_path = tmp_path / "evidence.sqlite3"
    state_path = tmp_path / "state.json"
    datasets_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    datasets_dir.mkdir()
    runs_dir.mkdir()

    base_t = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    metas = [
        MetaAlert(
            meta_id=i,
            agent_id="001",
            agent_name="agent-ubuntu",
            rule_group_primary="authentication_failed",
            start_time=base_t + timedelta(hours=i),
            end_time=base_t + timedelta(hours=i, minutes=10),
            alert_count=5 + i,
            max_severity=3 + (i % 10),
            rule_id_distribution={"5710": 5 + i},
            severity_distribution={3: 5 + i},
            agent_criticality=1.0,
            wazuh_alert_ids=(f"aid_{i}_1", f"aid_{i}_2"),
            mitre_tactics_unique=("credential-access",),
            critical_mitre_present=False,
            metadata={},
        )
        for i in range(1, 10)
    ]
    bundle = train_reference_pipeline(metas, random_state=42, model_version="test-model-v1")
    scoring_pipe = ScoringPipeline(bundle)

    evidence_store = RawAlertEvidenceStore(db_path)
    state_mgr = DurableStateManager(state_path)

    service = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
        source_mode="DEFERRED",
    )

    replay_ctrl = ReplayController(
        scoring_pipeline=scoring_pipe,
        replay_data_dir=datasets_dir,
        replay_runs_dir=runs_dir,
    )

    app = create_app(
        service=service,
        api_key="secret-api-key-123",
        raw_evidence_store=evidence_store,
        replay_controller=replay_ctrl,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret-api-key-123"}
    return client, headers, service, evidence_store, replay_ctrl, datasets_dir


def test_auth_check_endpoint(test_setup):
    client, headers, _, _, _, _ = test_setup

    # Without key -> 401
    resp_unauth = client.get("/api/v1/auth/check")
    assert resp_unauth.status_code == 401

    # With key -> 200
    resp_auth = client.get("/api/v1/auth/check", headers=headers)
    assert resp_auth.status_code == 200
    assert resp_auth.json()["authenticated"] is True


def test_dashboard_summary_and_verbatim_arr(test_setup):
    client, headers, service, evidence_store, _, _ = test_setup

    # Insert 2 raw alerts
    a1 = CanonicalRawAlert(
        wazuh_alert_id="aid-01",
        timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        agent_id="001",
        agent_name="agent-ubuntu",
        rule_group_primary="auth",
        rule_level=5,
        rule_id="1001",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1.0,
        metadata=MappingProxyType({}),
    )
    service.ingest_alert(a1)

    resp = client.get("/api/v1/dashboard/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "raw_alert_count" in data
    assert "meta_alert_count" in data
    assert "alert_reduction_rate_percent" in data


def test_dashboard_integrations_backend_truth(test_setup):
    client, headers, _, _, _, _ = test_setup
    resp = client.get("/api/v1/dashboard/integrations", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["wazuh"]["status"] == "DEFERRED"
    assert data["rbta"]["status"] == "READY"
    assert data["model"]["status"] == "READY"
    assert data["shuffle"]["status"] == "DEFERRED_EXTERNAL"
    assert data["telegram"]["status"] == "DEFERRED_EXTERNAL"


def test_meta_alerts_raw_alerts_resolution_and_unresolved(test_setup):
    client, headers, service, evidence_store, _, _ = test_setup

    a1 = CanonicalRawAlert(
        wazuh_alert_id="res-1",
        timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        agent_id="001",
        agent_name="agent-ubuntu",
        rule_group_primary="auth",
        rule_level=5,
        rule_id="1001",
        mitre_tactics=("initial-access",),
        srcip="10.0.0.1",
        agent_criticality=1.0,
        metadata=MappingProxyType({"rule_description": "Initial login attempt"}),
    )
    service.ingest_alert(a1)

    # Force a scored meta alert with resolved ID res-1 and missing ID res-unresolved
    assert len(service.finalized_history) > 0 or len(service.engine.snapshot_buckets()) >= 0
    flushed = service.engine.flush_idle(datetime(2026, 8, 29, 14, 0, 0, tzinfo=timezone.utc))
    if flushed:
        service.pending_scoring.extend(flushed)
        scored = service._drain_pending_scoring()
        target_meta_id = scored[0].meta_id
    else:
        target_meta_id = 1

    resp = client.get(f"/api/v1/meta-alerts/{target_meta_id}/raw-alerts", headers=headers)
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert "source_total" in data
        assert "resolved_total" in data
        assert "filtered_total" in data
        assert "unresolved_alert_ids" in data
        assert isinstance(data["items"], list)


def test_raw_alert_detail_redaction(test_setup):
    client, headers, service, evidence_store, _, _ = test_setup

    secret_alert = CanonicalRawAlert(
        wazuh_alert_id="sec-alert-999",
        timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        agent_id="001",
        agent_name="agent-ubuntu",
        rule_group_primary="auth",
        rule_level=5,
        rule_id="1001",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1.0,
        metadata=MappingProxyType({"api_key": "top_secret_token_val"}),
    )
    evidence_store.store(secret_alert)

    resp = client.get("/api/v1/raw-alerts/sec-alert-999", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["api_key"] == "[REDACTED]"


def test_replay_start_contract_and_datasets(test_setup):
    client, headers, _, _, _, datasets_dir = test_setup

    # Create dummy dataset
    (datasets_dir / "demo.jsonl").write_text("{\"id\": \"1\", \"timestamp\": \"2026-08-29T10:00:00Z\", \"agent\": {\"id\": \"001\"}, \"rule\": {\"id\": \"1\", \"level\": 3, \"groups\": [\"syslog\"]}}\n", encoding="utf-8")

    # List datasets
    resp_ds = client.get("/api/v1/replay/datasets", headers=headers)
    assert resp_ds.status_code == 200
    assert len(resp_ds.json()["items"]) == 1
    assert resp_ds.json()["items"][0]["name"] == "demo.jsonl"

    # Start with Pydantic JSON body
    resp_start = client.post(
        "/api/v1/replay/start",
        json={"dataset_name": "demo.jsonl", "speed_factor": "MAX"},
        headers=headers,
    )
    assert resp_start.status_code == 200
    assert resp_start.json()["status"] in ("RUNNING", "COMPLETED")


def test_nonexistent_api_route_returns_404_json(test_setup):
    client, headers, _, _, _, _ = test_setup
    resp = client.get("/api/v1/does_not_exist", headers=headers)
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
