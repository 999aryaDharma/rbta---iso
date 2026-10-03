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


def _bootstrap_env(tmp_path: Path, extra: dict) -> dict:
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "runtime" / "state.json"
    publish_test_model(reg_dir, "boot-v1")
    return {
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "boot-v1",
        "RBTA_STATE_FILE": str(state_file),
        "RBTA_API_KEY": "test-secret-key-123",
        **extra,
    }


def test_worker_and_dispatcher_absent_by_default(tmp_path: Path):
    """Without opt-in, no worker thread and no dispatcher are attached."""
    app = create_production_app(env=_bootstrap_env(tmp_path, {}))
    assert app.state.live_worker is None
    assert app.state.telegram_dispatcher is None


def test_worker_enabled_attaches_disabled_dispatcher_without_creds(tmp_path: Path, monkeypatch):
    """Worker opt-in attaches a dispatcher that stays thread-stopped without credentials."""
    monkeypatch.delenv("RBTA_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("RBTA_TELEGRAM_CHAT_ID", raising=False)
    app = create_production_app(
        env=_bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"})
    )
    assert app.state.live_worker is not None
    dispatcher = app.state.telegram_dispatcher
    assert dispatcher is not None
    assert dispatcher.enabled is False


def test_worker_enabled_with_creds_attaches_enabled_dispatcher(tmp_path: Path):
    """Explicit Telegram credentials enable the dispatcher (thread starts in lifespan)."""
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {
                "RBTA_LIVE_WORKER_ENABLED": "true",
                "RBTA_TELEGRAM_BOT_TOKEN": "tok-test",
                "RBTA_TELEGRAM_CHAT_ID": "chat-test",
            },
        )
    )
    dispatcher = app.state.telegram_dispatcher
    assert dispatcher is not None
    assert dispatcher.enabled is True


# --- F13-server ---

def test_telegram_dry_run_env_forces_disabled_dispatcher(tmp_path: Path):
    """RBTA_TELEGRAM_DRY_RUN=true forces dry-run even when credentials exist."""
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {
                "RBTA_LIVE_WORKER_ENABLED": "true",
                "RBTA_TELEGRAM_BOT_TOKEN": "tok-test",
                "RBTA_TELEGRAM_CHAT_ID": "chat-test",
                "RBTA_TELEGRAM_DRY_RUN": "true",
            },
        )
    )
    dispatcher = app.state.telegram_dispatcher
    assert dispatcher is not None
    assert dispatcher.enabled is False


def test_worker_enabled_defaults_source_mode_live(tmp_path: Path):
    """Without explicit RBTA_SOURCE_MODE, the live service uses LIVE when the
    worker is enabled (replay controller keeps its own REPLAY mode)."""
    app = create_production_app(env=_bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"}))
    assert app.state.live_worker is not None
    assert app.state.live_worker.service.source_mode == "LIVE"


def test_worker_enabled_respects_explicit_source_mode(tmp_path: Path):
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {"RBTA_LIVE_WORKER_ENABLED": "true", "RBTA_SOURCE_MODE": "DEFERRED"},
        )
    )
    assert app.state.live_worker is not None
    assert app.state.live_worker.service.source_mode == "DEFERRED"


def test_tls_verify_false_warns_and_persists(tmp_path: Path, caplog):
    """WAZUH_INDEXER_VERIFY_TLS=false warns at startup and records tls_verify=false."""
    import logging

    with caplog.at_level(logging.WARNING, logger="rbta.server"):
        app = create_production_app(
            env=_bootstrap_env(
                tmp_path,
                {
                    "RBTA_LIVE_WORKER_ENABLED": "true",
                    "WAZUH_INDEXER_VERIFY_TLS": "false",
                },
            )
        )
    assert any("TLS" in rec.message or "verify" in rec.message.lower() for rec in caplog.records), (
        [rec.message for rec in caplog.records]
    )
    state = app.state.live_worker.service.get_live_source_state()
    assert state.get("tls_verify") is False


def test_worker_drain_defaults_to_false_in_bootstrap(tmp_path: Path):
    """Bootstrap wires drain_on_stop=False unless explicitly opted in."""
    base_env = _bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"})
    app = create_production_app(env=base_env)
    assert app.state.live_worker is not None
    assert app.state.live_worker.drain_on_stop is False

    # NOTE: separate directory — the OS state lock is held process-wide, so a
    # second bootstrap on the same state file must fail fast (see F12).
    opt_dir = tmp_path / "opt-in"
    opt_dir.mkdir()
    env_opt_in = _bootstrap_env(
        opt_dir,
        {"RBTA_LIVE_WORKER_ENABLED": "true", "RBTA_LIVE_WORKER_DRAIN_ON_STOP": "true"},
    )
    app_opt_in = create_production_app(env=env_opt_in)
    assert app_opt_in.state.live_worker.drain_on_stop is True


def test_second_bootstrap_on_same_state_file_fails_fast(tmp_path: Path):
    """F12: a second live writer on the same state file is rejected."""
    import pytest as _pytest

    from src.runtime.live_worker import acquire_state_lock

    base_env = _bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"})
    create_production_app(env=base_env)
    state_file = base_env["RBTA_STATE_FILE"]
    with _pytest.raises(RuntimeError, match="another live writer holds"):
        acquire_state_lock(state_file)


def test_order_buffer_off_by_default(tmp_path: Path):
    """Buffer stays off unless explicitly enabled (honest null in status)."""
    app = create_production_app(env=_bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"}))
    assert app.state.live_worker.coordinator.order_buffer is None


def test_order_buffer_enabled_via_env(tmp_path: Path):
    """RBTA_ORDER_BUFFER_ENABLED=true wires the waiting-room sorter."""
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {"RBTA_LIVE_WORKER_ENABLED": "true", "RBTA_ORDER_BUFFER_ENABLED": "true"},
        )
    )
    assert app.state.live_worker.coordinator.order_buffer is not None


# --- N4: state lock applies to every bootstrap, not only worker-enabled ---

def test_state_lock_applies_without_worker_enabled(tmp_path: Path):
    """Bootstrapping without the worker still takes the state lock (fail fast)."""
    from src.runtime.live_worker import acquire_state_lock, release_state_lock

    base_env = _bootstrap_env(tmp_path, {})
    create_production_app(env=base_env)
    state_file = base_env["RBTA_STATE_FILE"]
    try:
        with pytest.raises(RuntimeError, match="another live writer holds"):
            acquire_state_lock(state_file)
    finally:
        release_state_lock(state_file)
    # After release the same state file can be acquired again.
    try:
        acquire_state_lock(state_file)
    finally:
        release_state_lock(state_file)


# --- N6-worker: server forwards raw override values from env_map ---

def test_server_forwards_raw_model_override(tmp_path: Path):
    """RBTA_LIVE_MODEL_OVERRIDE is forwarded verbatim to the worker."""
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {
                "RBTA_LIVE_WORKER_ENABLED": "true",
                "RBTA_LIVE_MODEL_OVERRIDE": "boot-v1",
            },
        )
    )
    assert app.state.live_worker is not None
    assert app.state.live_worker.model_override == "boot-v1"


def test_tls_verify_choice_always_recorded_in_source_state(tmp_path: Path):
    """tls_verify is recorded for both secure and insecure choices (fail-visible)."""
    app_secure = create_production_app(env=_bootstrap_env(tmp_path / "s", {}))
    app_insecure = create_production_app(
        env=_bootstrap_env(tmp_path / "i", {"WAZUH_INDEXER_VERIFY_TLS": "false"})
    )
    assert app_secure.state.runtime_resolver.live_service.get_live_source_state()["tls_verify"] is True
    assert app_insecure.state.runtime_resolver.live_service.get_live_source_state()["tls_verify"] is False


# --- M3-server: lifespan always starts the dispatcher thread when a worker exists ---

def test_lifespan_starts_disabled_dispatcher_thread(tmp_path: Path):
    """Without credentials the dispatcher stays disabled, but lifespan must
    still run its thread (disabled/dry-run drains outbox + logs)."""
    app = create_production_app(env=_bootstrap_env(tmp_path, {"RBTA_LIVE_WORKER_ENABLED": "true"}))
    dispatcher = app.state.telegram_dispatcher
    assert dispatcher is not None
    assert dispatcher.enabled is False
    # Stub only the worker start (would spawn a network-polling thread);
    # the dispatcher thread under test is real.
    with patch.object(app.state.live_worker, "start", return_value=False):
        with TestClient(app):
            assert dispatcher._thread is not None and dispatcher._thread.is_alive()
    assert dispatcher._thread is None
    assert not app.state.live_worker.is_alive()
    dispatcher.stop()  # idempotent: safe to stop twice


def test_production_bootstrap_suppresses_and_dryrun_drains_old_escalate(tmp_path: Path):
    """M1+M3 wiring: first-started pin suppresses bootstrap backlog; dry-run drains + records."""
    import time as _time

    from tests.unit.runtime.test_telegram_dispatcher import _scored
    from src.runtime.live_coordinator import LiveCycleResult, LiveIngestionCoordinator

    def _noop_cycle(self, current_time=None):
        return LiveCycleResult(
            fast_candidates=0, recent_reconciliation_candidates=0,
            full_reconciliation_candidates=0, submitted_candidates=0,
            duplicate_noops=0, processed_new_ids=0, failures=0,
            new_scored_meta_alerts=0,
        )

    env = _bootstrap_env(
        tmp_path,
        {"RBTA_LIVE_WORKER_ENABLED": "true", "RBTA_TELEGRAM_DRY_RUN": "true"},
    )
    app = create_production_app(env=env)
    service = app.state.runtime_resolver.live_service
    old = _scored(900, end_time=datetime.now(timezone.utc) - timedelta(hours=5))
    service.outbox.append(old)

    with patch.object(LiveIngestionCoordinator, "run_cycle", _noop_cycle):
        with TestClient(app):
            deadline = _time.monotonic() + 10.0
            while service.get_outbox() and _time.monotonic() < deadline:
                _time.sleep(0.05)

    assert service.get_outbox() == []
    state = service.get_live_source_state()
    assert state.get("live_first_started_at")
    assert service.state_manager.notification_count() >= 1
    assert "SUPPRESSED_HISTORICAL" in service.state_manager.notification_verdicts()


def test_breaker_ack_env_reaches_coordinator(tmp_path: Path):
    """P1 latch: RBTA_QUARANTINE_BREAKER_ACK is forwarded to the coordinator."""
    app = create_production_app(
        env=_bootstrap_env(
            tmp_path,
            {"RBTA_LIVE_WORKER_ENABLED": "true", "RBTA_QUARANTINE_BREAKER_ACK": "qb-test-1"},
        )
    )
    assert app.state.live_worker.coordinator.breaker_ack == "qb-test-1"
