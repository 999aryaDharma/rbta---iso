"""Unit tests for LiveIngestionCoordinator (fast poll + recent reconciliation + full-retention sweep)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner
from src.runtime.durable_state import DurableStateManager
from src.runtime.live_coordinator import LiveCycleResult, LiveIngestionCoordinator
from src.runtime.live_source import LiveCanonicalizationError, WazuhIndexerLivePoller
from src.runtime.service import LiveRBTAService


def make_raw_alert(
    idx: int,
    ts: datetime,
    group: str = "pam",
    level: int = 3,
    crit: int = 1,
) -> CanonicalRawAlert:
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


def create_test_service(tmp_path: Path) -> LiveRBTAService:
    base_t = datetime(2026, 8, 28, 8, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        make_raw_alert(i, base_t + timedelta(minutes=i * 20), level=(i % 12) + 1)
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="coord-test-v1")
    scoring_pipe = ScoringPipeline(bundle)

    state_mgr = DurableStateManager(tmp_path / "coordinator_service_state.json")
    return LiveRBTAService(
        scoring_pipeline=scoring_pipe,
        state_manager=state_mgr,
        base_delta_t=timedelta(minutes=15),
        adaptive=False,
    )


def test_coordinator_fast_poll_and_reconciliation_cycle(tmp_path: Path):
    """Coordinator executes reconciliation and fast poll, deduplicating and ingesting candidates."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 28, 10, 30, 0, tzinfo=timezone.utc)
    a1 = make_raw_alert(1, t_now - timedelta(minutes=2))
    a2 = make_raw_alert(2, t_now - timedelta(minutes=1))

    poller.poll_recent.return_value = [a1, a2]
    poller.poll_reconciliation.return_value = [a1]
    poller.poll_full_reconciliation.return_value = []

    coord = LiveIngestionCoordinator(
        service=service,
        poller=poller,
        recent_reconciliation_interval=timedelta(minutes=5),
        full_reconciliation_interval=timedelta(hours=1),
    )

    result = coord.run_cycle(current_time=t_now, force_recent_reconciliation=True)

    assert result.fast_candidates == 2
    assert result.recent_reconciliation_candidates == 1
    assert result.submitted_candidates == 2
    assert result.processed_new_ids == 2
    assert result.duplicate_noops == 0
    assert result.failures == 0

    assert service.is_seen("alert_1")
    assert service.is_seen("alert_2")


def test_coordinator_full_retention_reconciliation_recovers_old_alert(tmp_path: Path):
    """Full-retention sweep discovers an alert from days earlier (Aug 20 when today is Aug 29)."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    a_recent = make_raw_alert(99, t_now - timedelta(minutes=2))
    a_old_retained = make_raw_alert(10, datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc))

    poller.poll_recent.return_value = [a_recent]
    poller.poll_reconciliation.return_value = [a_recent]
    poller.poll_full_reconciliation.return_value = [a_old_retained, a_recent]

    coord = LiveIngestionCoordinator(service=service, poller=poller)

    result = coord.run_cycle(current_time=t_now, force_full_reconciliation=True)

    assert service.is_seen("alert_99")
    assert service.is_seen("alert_10")
    assert result.processed_new_ids == 2
    assert result.full_reconciliation_candidates == 2


def test_coordinator_canonicalization_failure_does_not_advance_success_timestamp(tmp_path: Path):
    """When a source document fails canonicalization, coordinator raises and does NOT advance success timestamp."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    poller.poll_full_reconciliation.side_effect = LiveCanonicalizationError("Corrupt payload in hit 5")

    coord = LiveIngestionCoordinator(service=service, poller=poller)

    with pytest.raises(LiveCanonicalizationError, match="Corrupt payload"):
        coord.run_cycle(current_time=t_now, force_full_reconciliation=True)

    # State was NOT marked successful
    assert coord.last_full_reconciliation_at is None
    state = service.get_live_source_state()
    assert state.get("last_full_reconciliation_at") is None


def test_coordinator_full_retention_duplicates_are_safe(tmp_path: Path):
    """When full retention sweeps all retained indices, already processed alerts are recognized as duplicate no-ops."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t1 = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    a1 = make_raw_alert(1, t1)
    a2 = make_raw_alert(2, t1 + timedelta(minutes=5))
    a3 = make_raw_alert(3, t1 + timedelta(minutes=10))

    poller.poll_recent.return_value = [a1, a2]
    poller.poll_reconciliation.return_value = []
    poller.poll_full_reconciliation.return_value = []

    coord = LiveIngestionCoordinator(service=service, poller=poller)
    res1 = coord.run_cycle(current_time=t1 + timedelta(minutes=15), force_full_reconciliation=False)
    assert res1.processed_new_ids == 2

    # Later full retention sweep returns a1, a2, and a3
    poller.poll_recent.return_value = []
    poller.poll_reconciliation.return_value = []
    poller.poll_full_reconciliation.return_value = [a1, a2, a3]

    res2 = coord.run_cycle(current_time=t1 + timedelta(hours=2), force_full_reconciliation=True)
    assert res2.processed_new_ids == 1
    assert res2.duplicate_noops == 2
    assert service.is_seen("alert_3")


def test_coordinator_persists_transport_state(tmp_path: Path):
    """Coordinator preserves transport state (cursors and timestamps) in service durable state."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)
    poller.poll_recent.return_value = []
    poller.poll_reconciliation.return_value = []
    poller.poll_full_reconciliation.return_value = []

    t_now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
    coord = LiveIngestionCoordinator(service=service, poller=poller)
    coord.run_cycle(current_time=t_now, force_recent_reconciliation=True, force_full_reconciliation=True)

    state = service.get_live_source_state()
    assert state["recent_poll_cursor"] == t_now.isoformat()
    assert state["last_fast_poll_at"] == t_now.isoformat()
    assert state["last_recent_reconciliation_at"] == t_now.isoformat()
    assert state["last_full_reconciliation_at"] == t_now.isoformat()
    assert state["recent_reconciliation_days"] == 2
