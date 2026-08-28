"""Unit tests for LiveRBTAService coordination, idle flushing, and controlled shutdown (Sprint 7)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    """Live service ingests alerts, flushes idle buckets when idle_gap > delta_t, and enqueues to outbox."""
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

    # 5. Outbox contains the scored alert
    outbox = service.get_outbox()
    assert len(outbox) == 1
    assert outbox[0].meta_id == 1

    # 6. Acknowledge outbox item
    service.acknowledge_outbox(outbox[0].meta_id)
    assert len(service.get_outbox()) == 0


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

    # Controlled shutdown (without draining, preserving active bucket)
    service1.shutdown(drain=False)

    # Start fresh service2 with same state manager
    service2 = LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )

    # Ingest alert 2 at 10:05 -> merges into restored bucket from service1
    a2 = make_alert(2, base_t + timedelta(minutes=5))
    service2.ingest_alert(a2)

    # Shutdown with drain -> produces finalized meta-alert with 2 alerts
    drained = service2.shutdown(drain=True)
    assert len(drained) == 1
    assert drained[0].alert_count == 2
    assert drained[0].source_alert_ids == ("alert_1", "alert_2")
