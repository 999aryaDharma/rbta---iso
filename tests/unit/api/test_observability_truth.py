"""Unit tests verifying observability truth, dynamic system_status derivation, truthful integrations, and raw timeseries."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.registry import ModelRegistry
from src.model.scoring_pipeline import train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner
from src.runtime.durable_state import DurableStateManager
from src.runtime.observability import (
    get_dashboard_integrations,
    get_dashboard_summary,
    get_dashboard_system,
    get_dashboard_timeseries,
)
from src.runtime.raw_evidence import RawAlertEvidenceStore
from src.runtime.replay_controller import ReplayController
from src.runtime.service import LiveRBTAService
from src.api.server import create_production_app


def _make_test_alert(i: int, ts: datetime) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"art_{i}",
        timestamp=ts,
        agent_id=f"00{1 + (i % 3)}",
        agent_name=f"agent-{1 + (i % 3)}",
        rule_group_primary="pam" if i % 2 == 0 else "syslog",
        rule_level=(i % 12) + 1,
        rule_id=f"550{i % 6}",
        mitre_tactics=("credential-access",) if i % 3 == 0 else (),
        srcip=f"192.168.1.{50 + i % 10}",
        agent_criticality=(i % 3) + 1,
    )


@pytest.fixture
def test_scoring_pipeline(tmp_path: Path):
    base_t = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [_make_test_alert(i, base_t + timedelta(minutes=i * 5)) for i in range(40)]
    runner = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False)
    batch_res = runner.run(alerts)
    bundle = train_reference_pipeline(
        batch_res.meta_alerts,
        random_state=123,
        model_version="rbta-truth-v42",
    )
    from src.model.scoring_pipeline import ScoringPipeline
    return ScoringPipeline(bundle)


def test_dashboard_system_observability_metadata_truth(tmp_path: Path, test_scoring_pipeline):
    """Observability system metadata must reflect exact artifact fields without invented fallbacks."""
    state_mgr = DurableStateManager(tmp_path / "state.json")
    evidence_store = RawAlertEvidenceStore(tmp_path / "evidence.sqlite3")

    service = LiveRBTAService(
        scoring_pipeline=test_scoring_pipeline,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
        source_mode="DEFERRED",
    )

    sys_meta = get_dashboard_system(service)

    assert sys_meta["model_version"] == "rbta-truth-v42"
    assert sys_meta["random_state"] == 123
    assert sys_meta["tukey_threshold"] == test_scoring_pipeline.threshold.threshold
    assert len(sys_meta["feature_names"]) == 7
    assert sys_meta["feature_names"] == list(test_scoring_pipeline.schema["features"])
    assert sys_meta["source_mode"] == "DEFERRED"
    assert sys_meta["system_status"] == "READY"


def test_system_status_derived_ready_and_degraded(tmp_path: Path, test_scoring_pipeline):
    """system_status is dynamically derived: READY when all components intact, DEGRADED when broken."""
    state_mgr = DurableStateManager(tmp_path / "state.json")
    evidence_store = RawAlertEvidenceStore(tmp_path / "evidence.sqlite3")

    service = LiveRBTAService(
        scoring_pipeline=test_scoring_pipeline,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
    )
    assert get_dashboard_system(service)["system_status"] == "READY"
    assert get_dashboard_summary(service, evidence_store)["system_status"] == "READY"

    # Corrupt pipeline metadata
    test_scoring_pipeline.metadata.pop("model_version", None)
    assert get_dashboard_system(service)["system_status"] == "DEGRADED"
    assert get_dashboard_summary(service, evidence_store)["system_status"] == "DEGRADED"


def test_production_source_mode_deferred_by_default(tmp_path: Path):
    """create_production_app defaults RBTA_SOURCE_MODE to DEFERRED and validates allowed values."""
    reg_dir = tmp_path / "models"
    state_file = tmp_path / "state.json"
    evidence_file = tmp_path / "evidence.sqlite3"

    base_t = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [_make_test_alert(i, base_t + timedelta(minutes=i * 5)) for i in range(40)]
    runner = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False)
    bundle = train_reference_pipeline(runner.run(alerts).meta_alerts, random_state=42, model_version="prod-m1")
    registry = ModelRegistry(base_dir=reg_dir)
    registry.publish_bundle(bundle, model_version="prod-m1")

    # 1. Default mode -> DEFERRED
    env_default = {
        "RBTA_API_KEY": "test-key",
        "RBTA_MODEL_REGISTRY_DIR": str(reg_dir),
        "RBTA_MODEL_VERSION": "prod-m1",
        "RBTA_STATE_FILE": str(state_file),
        "RBTA_RAW_EVIDENCE_DB": str(evidence_file),
    }
    app1 = create_production_app(env=env_default)
    srv1 = app1.state.runtime_resolver.live_service
    assert srv1.source_mode == "DEFERRED"

    # 2. Explicit LIVE mode
    env_live = dict(env_default, RBTA_SOURCE_MODE="LIVE")
    app2 = create_production_app(env=env_live)
    srv2 = app2.state.runtime_resolver.live_service
    assert srv2.source_mode == "LIVE"

    # 3. Invalid mode raises ValueError
    env_invalid = dict(env_default, RBTA_SOURCE_MODE="UNKNOWN_MODE")
    with pytest.raises(ValueError, match="RBTA_SOURCE_MODE"):
        create_production_app(env=env_invalid)


def test_dashboard_integrations_truthful_status(tmp_path: Path, test_scoring_pipeline):
    """get_dashboard_integrations reports truthful external statuses based on runtime context."""
    state_mgr = DurableStateManager(tmp_path / "state.json")
    evidence_store = RawAlertEvidenceStore(tmp_path / "evidence.sqlite3")

    # When source_mode is DEFERRED
    service_def = LiveRBTAService(
        scoring_pipeline=test_scoring_pipeline,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
        source_mode="DEFERRED",
    )
    integ_def = get_dashboard_integrations(service_def)
    assert integ_def["wazuh"]["status"] == "DEFERRED"
    assert integ_def["rbta"]["status"] == "READY"
    assert integ_def["model"]["status"] == "READY"
    assert integ_def["outbox"]["status"] == "READY"
    assert integ_def["shuffle"]["status"] == "DEFERRED_EXTERNAL"
    assert integ_def["telegram"]["status"] == "DEFERRED_EXTERNAL"

    # When source_mode is LIVE (no coordinator)
    service_live = LiveRBTAService(
        scoring_pipeline=test_scoring_pipeline,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
        source_mode="LIVE",
    )
    integ_live = get_dashboard_integrations(service_live)
    assert integ_live["wazuh"]["status"] == "UNKNOWN"


def test_timeseries_counts_active_buckets_as_raw_evidence(tmp_path: Path, test_scoring_pipeline):
    """Raw timeseries counts come strictly from raw evidence store even while in active bucket."""
    state_mgr = DurableStateManager(tmp_path / "state.json")
    evidence_store = RawAlertEvidenceStore(tmp_path / "evidence.sqlite3")

    service = LiveRBTAService(
        scoring_pipeline=test_scoring_pipeline,
        state_manager=state_mgr,
        raw_evidence_store=evidence_store,
    )

    base_t = datetime(2026, 8, 29, 10, 5, 0, tzinfo=timezone.utc)
    a1 = CanonicalRawAlert(
        wazuh_alert_id="raw-1",
        timestamp=base_t,
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="pam",
        rule_level=7,
        rule_id="5501",
        mitre_tactics=("credential-access",),
        srcip="192.168.1.50",
        agent_criticality=2,
    )
    a2 = CanonicalRawAlert(
        wazuh_alert_id="raw-2",
        timestamp=base_t + timedelta(minutes=5),
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="pam",
        rule_level=8,
        rule_id="5502",
        mitre_tactics=("credential-access",),
        srcip="192.168.1.51",
        agent_criticality=2,
    )

    # Ingest 2 alerts into service -> stored in raw evidence store, active bucket created, NO finalized MetaAlert
    service.ingest_alert(a1)
    service.ingest_alert(a2)
    assert len(service.finalized_history) == 0

    # Query timeseries for window including 10:00 UTC
    ts_bins = get_dashboard_timeseries(service, evidence_store, window_hours=24)
    hour_key = "2026-08-29 10:00"
    target_bin = next((b for b in ts_bins if b["timestamp"] == hour_key), None)
    assert target_bin is not None, f"Hour bin {hour_key} not found in {ts_bins}"
    assert target_bin["raw_alerts"] == 2
    assert target_bin["meta_alerts"] == 0

    # Now drain service to finalize MetaAlert
    scored = service.drain_and_score()
    assert len(scored) == 1
    assert len(service.finalized_history) == 1

    # Query timeseries again -> raw_alerts remains 2, meta_alerts is now 1
    ts_bins_after = get_dashboard_timeseries(service, evidence_store, window_hours=24)
    target_bin_after = next((b for b in ts_bins_after if b["timestamp"] == hour_key), None)
    assert target_bin_after is not None
    assert target_bin_after["raw_alerts"] == 2
    assert target_bin_after["meta_alerts"] == 1


def test_replay_run_provenance_uses_exact_model_version(tmp_path: Path, test_scoring_pipeline):
    """ReplayController writes exact loaded model version to run.json without fallback."""
    replay_data_dir = tmp_path / "replay_data"
    replay_data_dir.mkdir(parents=True)
    sample_jsonl = replay_data_dir / "test_sample.jsonl"
    sample_jsonl.write_text('{"id":"sample-1"}\n', encoding="utf-8")

    replay_runs_dir = tmp_path / "replay_runs"

    controller = ReplayController(
        scoring_pipeline=test_scoring_pipeline,
        replay_data_dir=replay_data_dir,
        replay_runs_dir=replay_runs_dir,
    )

    run_id = controller._init_run_workspace("test_sample.jsonl", "MAX")
    run_meta_file = replay_runs_dir / run_id / "run.json"
    assert run_meta_file.exists()

    import json
    with open(run_meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert meta_data["model_version"] == "rbta-truth-v42"
    assert meta_data["model_version"] != "v1"
