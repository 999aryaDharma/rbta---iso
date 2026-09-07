"""Unit tests for LiveRBTAService coordination, idle flushing, and controlled shutdown (Sprint 7)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner
from src.runtime.durable_state import DurableStateManager
from src.runtime.service import LiveRBTAService


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


def test_live_service_ingestion_scoring_and_idle_flush(tmp_path: Path):
    """Live service finalizes every score but queues only actionable delivery records."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Train model bundle
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="live-v1")
    scoring_pipe = ScoringPipeline(bundle)

    state_mgr = DurableStateManager(tmp_path / "service_state.json")
    service = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    # 2. Ingest alert at 10:00
    a1 = make_alert(1, base_t)
    out1 = service.ingest_alert(a1)
    assert out1 == []  # Aggregating in active bucket

    # 3. Trigger idle flush at 10:10 (10m idle <= 15m delta_t -> still merge eligible -> NO FLUSH)
    flushed_10 = service.check_idle_flush(base_t + timedelta(minutes=10))
    assert flushed_10 == []

    # 4. Trigger idle flush at 10:16 (16m idle > 15m delta_t -> FLUSH)
    flushed_16 = service.check_idle_flush(base_t + timedelta(minutes=16))
    assert len(flushed_16) == 1
    assert isinstance(flushed_16[0], ScoredMetaAlert)
    assert flushed_16[0].meta_id == 1

    # 5. This low-context score remains in history but not delivery outbox.
    outbox = service.get_outbox()
    assert flushed_16[0].action != "ESCALATE"
    assert outbox == []
    assert len(service.get_history()) == 1  # History survives ACK
    assert service.get_meta_detail(1) is not None


def test_live_service_controlled_shutdown_and_restart_recovery(tmp_path: Path):
    """Service shutdown persists state; new service instance restores active bucket and outbox."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="live-v1")
    scoring_pipe = ScoringPipeline(bundle)

    state_mgr = DurableStateManager(tmp_path / "service_state.json")
    service1 = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    # Ingest alert 1 into active bucket
    service1.ingest_alert(make_alert(1, base_t))
    # Flush so there's one in history
    service1.check_idle_flush(base_t + timedelta(minutes=20))

    # Controlled shutdown (without draining, preserving active bucket and history)
    service1.shutdown(drain=False)

    # Start fresh service2 with same state manager
    service2 = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    assert len(service2.get_history()) == 1  # History survives restart

    # Ingest alert 2 at 10:25 -> new active bucket
    a2 = make_alert(2, base_t + timedelta(minutes=25))
    service2.ingest_alert(a2)

    # Shutdown with drain -> produces another finalized meta-alert
    drained = service2.shutdown(drain=True)
    assert len(drained) == 1
    assert drained[0].alert_count == 1
    assert drained[0].source_alert_ids == ("alert_2",)
    assert len(service2.get_history()) == 2


def test_live_service_scoring_failure_durable_recovery(tmp_path: Path):
    """When scoring fails, MetaAlert is retained in durable pending_scoring queue and recovered on restart."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="live-v1")
    scoring_pipe = ScoringPipeline(bundle)

    state_mgr = DurableStateManager(tmp_path / "failing_service_state.json")
    service1 = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    # Ingest alert 1
    service1.ingest_alert(make_alert(1, base_t))

    # Mock scoring to simulate downstream failure during idle flush
    with patch.object(scoring_pipe, "score_single", side_effect=RuntimeError("Model inference service unavailable")):
        with pytest.raises(RuntimeError, match="Model inference service unavailable"):
            service1.check_idle_flush(base_t + timedelta(minutes=20))

    # Verify that pending_scoring is non-empty and persisted to disk
    assert len(service1.pending_scoring) == 1
    assert service1.pending_scoring[0].meta_id == 1
    assert len(service1.get_outbox()) == 0

    # Start service2 with healthy scoring pipeline
    service2 = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    # Service2 must have automatically recovered and scored the pending meta-alert!
    assert len(service2.pending_scoring) == 0
    assert service2.get_outbox() == []
    assert len(service2.get_history()) == 1


def test_non_escalate_scored_alert_is_history_not_actionable_outbox(tmp_path: Path):
    """The delivery outbox must contain only records whose action is ESCALATE."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    scoring_pipe = ScoringPipeline(train_reference_pipeline(batch_res.meta_alerts, model_version="outbox-v1"))
    service = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=DurableStateManager(tmp_path / "state.json"),
        adaptive=False,
    )
    meta = batch_res.meta_alerts[0]
    suppressed = ScoredMetaAlert(
        meta_id=meta.meta_id,
        agent_id=meta.agent_id,
        agent_name=meta.agent_name,
        rule_group_primary=meta.rule_group_primary,
        start_time=meta.start_time,
        end_time=meta.end_time,
        alert_count=meta.alert_count,
        max_severity=meta.max_severity,
        mitre_tactics=meta.mitre_tactics_unique,
        seven_features={
            "max_severity": float(meta.max_severity),
            "mitre_tactic_count": 0.0,
            "critical_mitre_tactic_present": 0.0,
            "alert_count_log": 0.0,
            "rule_diversity_shannon": 0.0,
            "severity_dispersion": 0.0,
            "agent_criticality": float(meta.agent_criticality),
        },
        raw_model_score=0.1,
        anomaly_score=0.1,
        threshold_used=0.5,
        decision="NOISE",
        action="SUPPRESS",
        escalate=False,
        model_version="outbox-v1",
        feature_schema_version="1.0",
        score_calibration_version="minmax-v1",
        source_alert_ids=meta.wazuh_alert_ids,
    )
    service.pending_scoring.append(meta)

    with patch.object(scoring_pipe, "score_single", return_value=suppressed):
        result = service._drain_pending_scoring()

    assert result == [suppressed]
    assert service.get_outbox() == []
    assert service.get_history() == [suppressed]


def test_seen_id_memory_is_bounded_by_sqlite_duplicate_index(tmp_path: Path):
    base_t = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
    alerts = [make_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1) for i in range(30)]
    metas = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(alerts).meta_alerts
    pipeline = ScoringPipeline(train_reference_pipeline(metas, model_version="seen-v1"))
    service = LiveRBTAService(pipeline, DurableStateManager(tmp_path / "state.json"), adaptive=False)

    first = make_alert(999, base_t)
    service.ingest_alert(first)
    assert len(service.engine._seen_alert_ids) <= 10_000
    assert service.is_seen(first.wazuh_alert_id)
    before = len(service.engine._active_buckets)
    assert service.ingest_alert(first) == []
    assert len(service.engine._active_buckets) == before
