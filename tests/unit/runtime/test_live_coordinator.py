"""Unit tests for LiveIngestionCoordinator (fast poll + reconciliation + durable dedup)."""

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
from src.runtime.live_source import WazuhIndexerLivePoller
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
    poller.poll_reconciliation.return_value = [a1]  # duplicate candidate in recon scan

    coord = LiveIngestionCoordinator(
        service=service,
        poller=poller,
        reconciliation_interval=timedelta(minutes=5),
    )

    result = coord.run_cycle(current_time=t_now, force_reconciliation=True)

    assert result.fast_candidates == 2
    assert result.reconciliation_candidates == 1
    assert result.submitted_candidates == 2  # merged in-cycle
    assert result.processed_new_ids == 2
    assert result.duplicate_noops == 0
    assert result.failures == 0

    assert service.is_seen("alert_1")
    assert service.is_seen("alert_2")


def test_coordinator_reconciliation_recovers_alert_outside_fast_overlap(tmp_path: Path):
    """Reconciliation scan discovers an old alert (10:10) outside the fast 5-minute overlap window (10:25..10:30)."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 28, 10, 30, 0, tzinfo=timezone.utc)
    # Fast window only sees 10:28
    a_recent = make_raw_alert(99, t_now - timedelta(minutes=2))
    # Late document at 10:10 (20 minutes ago, outside 5m overlap)
    a_late = make_raw_alert(42, t_now - timedelta(minutes=20))

    poller.poll_recent.return_value = [a_recent]
    poller.poll_reconciliation.return_value = [a_late, a_recent]

    coord = LiveIngestionCoordinator(
        service=service,
        poller=poller,
        reconciliation_interval=timedelta(minutes=5),
    )

    result = coord.run_cycle(current_time=t_now, force_reconciliation=True)

    # Both alerts must be submitted and processed
    assert service.is_seen("alert_99")
    assert service.is_seen("alert_42")
    assert result.processed_new_ids == 2


def test_coordinator_very_old_unseen_alert_is_never_dropped(tmp_path: Path):
    """An alert from hours earlier (09:00 when current is 15:00) is processed, not discarded."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 28, 15, 0, 0, tzinfo=timezone.utc)
    a_very_old = make_raw_alert(1, datetime(2026, 8, 28, 9, 0, 0, tzinfo=timezone.utc))

    poller.poll_recent.return_value = []
    poller.poll_reconciliation.return_value = [a_very_old]

    coord = LiveIngestionCoordinator(service=service, poller=poller)
    result = coord.run_cycle(current_time=t_now, force_reconciliation=True)

    assert result.processed_new_ids == 1
    assert service.is_seen("alert_1")


def test_coordinator_already_processed_old_alert_is_duplicate_noop(tmp_path: Path):
    """When reconciliation scans full day and rereads previously processed alerts, they are duplicate no-ops."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t1 = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    a1 = make_raw_alert(1, t1)
    a2 = make_raw_alert(2, t1 + timedelta(minutes=5))

    poller.poll_recent.return_value = [a1, a2]
    poller.poll_reconciliation.return_value = []

    coord = LiveIngestionCoordinator(service=service, poller=poller)
    res1 = coord.run_cycle(current_time=t1 + timedelta(minutes=10), force_reconciliation=False)
    assert res1.processed_new_ids == 2

    # Later reconciliation scan re-reads a1, a2 and newly discovers a3
    t2 = t1 + timedelta(hours=2)
    a3 = make_raw_alert(3, t2)
    poller.poll_recent.return_value = [a3]
    poller.poll_reconciliation.return_value = [a1, a2, a3]

    res2 = coord.run_cycle(current_time=t2, force_reconciliation=True)
    assert res2.processed_new_ids == 1  # only a3 is new
    assert res2.duplicate_noops == 2    # a1, a2 are duplicate no-ops


def test_coordinator_failure_and_reconciliation_retry(tmp_path: Path):
    """If service processing fails on an alert, coordinator raises, and next cycle safely retries."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)

    t_now = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    a1 = make_raw_alert(1, t_now)

    poller.poll_recent.return_value = [a1]
    poller.poll_reconciliation.return_value = []

    coord = LiveIngestionCoordinator(service=service, poller=poller)

    # Simulate ingestion failure in service
    with patch.object(service, "ingest_alert", side_effect=RuntimeError("Transient database lock")):
        with pytest.raises(RuntimeError, match="Transient database lock"):
            coord.run_cycle(current_time=t_now)

    # Alert 1 was not committed
    assert not service.is_seen("alert_1")

    # Next reconciliation cycle retries Alert 1
    poller.poll_recent.return_value = []
    poller.poll_reconciliation.return_value = [a1]

    result = coord.run_cycle(current_time=t_now + timedelta(minutes=1), force_reconciliation=True)
    assert result.processed_new_ids == 1
    assert service.is_seen("alert_1")


def test_coordinator_persists_transport_state(tmp_path: Path):
    """Coordinator preserves transport state (cursors and timestamps) in service durable state."""
    service = create_test_service(tmp_path)
    poller = MagicMock(spec=WazuhIndexerLivePoller)
    poller.poll_recent.return_value = []
    poller.poll_reconciliation.return_value = []

    t_now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
    coord = LiveIngestionCoordinator(service=service, poller=poller)
    coord.run_cycle(current_time=t_now, force_reconciliation=True)

    state = service.get_live_source_state()
    assert state["recent_poll_cursor"] == t_now.isoformat()
    assert state["last_fast_poll_at"] == t_now.isoformat()
    assert state["last_reconciliation_at"] == t_now.isoformat()
    assert state["reconciliation_days"] == 2
