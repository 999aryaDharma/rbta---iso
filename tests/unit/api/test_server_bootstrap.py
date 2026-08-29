"""Unit tests for production server bootstrap and configuration validation."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from src.api.server import create_production_app
from src.contracts.raw_alert import CanonicalRawAlert
from src.model.registry import ModelRegistry
from src.model.scoring_pipeline import train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner


def make_raw_alert(idx: int, ts: datetime) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"bootstrap_alert_{idx}",
        timestamp=ts,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=(idx % 12) + 1,
        rule_id=f"550{idx % 5}",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
    )


def publish_test_model(registry_dir: Path, version: str = "boot-v1") -> None:
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [make_raw_alert(i, base_t + timedelta(minutes=i * 20)) for i in range(30)]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version=version)

    registry = ModelRegistry(base_dir=registry_dir)
    registry.publish_bundle(bundle, model_version=version)


def test_bootstrap_with_valid_config_and_model(tmp_path: Path):
    """Production bootstrap successfully loads configured model and exposes /health and /ready 200."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    publish_test_model(reg_dir, "boot-v1")

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "boot-v1",
        "RBTA_STATE_FILE": str(state_file),
        "RBTA_API_KEY": "test-secret-key-123",
    }

    app = create_production_app(env=env)
    client = TestClient(app)

    # 1. Health check
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json() == {"status": "ok", "service": "rbta-security-analytics"}

    # 2. Readiness check
    r_resp = client.get("/ready")
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["ready"] is True
    assert r_data["active_model_version"] == "boot-v1"

    # 3. Authenticated stats endpoint
    s_resp = client.get("/runtime/stats", headers={"Authorization": "Bearer test-secret-key-123"})
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert s_data["seen_alerts_count"] == 0
    assert s_data["active_buckets_count"] == 0


def test_bootstrap_missing_model_version_reports_ready_503(tmp_path: Path):
    """When no active model version is configured, /health is 200 but /ready is 503."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    reg_dir.mkdir(parents=True)

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "",
        "RBTA_STATE_FILE": str(state_file),
    }

    app = create_production_app(env=env)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    r_resp = client.get("/ready")
    assert r_resp.status_code == 503
    assert r_resp.json()["ready"] is False


def test_bootstrap_invalid_model_version_reports_ready_503(tmp_path: Path):
    """When configured model version does not exist, /ready returns 503."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    reg_dir.mkdir(parents=True)

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "nonexistent-version",
        "RBTA_STATE_FILE": str(state_file),
    }

    app = create_production_app(env=env)
    client = TestClient(app)

    r_resp = client.get("/ready")
    assert r_resp.status_code == 503
    assert r_resp.json()["ready"] is False


def test_bootstrap_inference_only_no_model_fitting(tmp_path: Path):
    """Bootstrap and server execution perform zero model training or fitting."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    publish_test_model(reg_dir, "boot-v1")

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "boot-v1",
        "RBTA_STATE_FILE": str(state_file),
    }

    with patch("src.model.scoring_pipeline.train_reference_pipeline") as mock_train:
        app = create_production_app(env=env)
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert mock_train.call_count == 0


def test_strict_bootstrap_missing_api_key_raises_runtime_error(tmp_path: Path):
    """Strict bootstrap fails closed when RBTA_API_KEY is missing or empty."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    publish_test_model(reg_dir, "boot-v1")

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "boot-v1",
        "RBTA_STATE_FILE": str(state_file),
        "RBTA_API_KEY": "",
    }

    with pytest.raises(RuntimeError, match="RBTA_API_KEY"):
        create_production_app(env=env, strict=True)


def test_strict_bootstrap_missing_model_version_raises_runtime_error(tmp_path: Path):
    """Strict bootstrap fails closed when RBTA_MODEL_VERSION is missing or empty."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"

    env = {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "   ",
        "RBTA_STATE_FILE": str(state_file),
        "RBTA_API_KEY": "test-key",
    }

    with pytest.raises(RuntimeError, match="RBTA_MODEL_VERSION"):
        create_production_app(env=env, strict=True)
