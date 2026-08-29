"""Live Ingestion Coordinator coordinating fast recent polling, reconciliation scans, and durable ingestion."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Set

from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.runtime.live_source import WazuhIndexerLivePoller
from src.runtime.service import LiveRBTAService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveCycleResult:
    """Operational observability metrics for a single live ingestion cycle."""

    fast_candidates: int
    reconciliation_candidates: int
    submitted_candidates: int
    duplicate_noops: int
    processed_new_ids: int
    failures: int
    new_scored_meta_alerts: int


class LiveIngestionCoordinator:
    """Coordinates fast recent polling, scheduled reconciliation scans, and submission to LiveRBTAService.

    Parameters
    ----------
    service : LiveRBTAService
        Target live stateful runtime service.
    poller : WazuhIndexerLivePoller
        Wazuh Indexer poller source.
    fast_poll_interval : timedelta
        Frequency between fast recent polling cycles (default 5 seconds).
    reconciliation_interval : timedelta
        Frequency between lossless reconciliation scans (default 5 minutes).
    reconciliation_days : int
        Number of recent daily indices to scan during reconciliation (default 2).
    """

    def __init__(
        self,
        service: LiveRBTAService,
        poller: Optional[WazuhIndexerLivePoller] = None,
        fast_poll_interval: timedelta = timedelta(seconds=5),
        reconciliation_interval: timedelta = timedelta(minutes=5),
        reconciliation_days: int = 2,
    ) -> None:
        self.service: LiveRBTAService = service
        self.poller: WazuhIndexerLivePoller = poller or WazuhIndexerLivePoller()
        self.fast_poll_interval: timedelta = fast_poll_interval
        self.reconciliation_interval: timedelta = reconciliation_interval
        self.reconciliation_days: int = reconciliation_days

        # Restore transport cursor state from service
        source_state = self.service.get_live_source_state()
        raw_cursor = source_state.get("recent_poll_cursor")
        self.recent_poll_cursor: Optional[datetime] = (
            datetime.fromisoformat(raw_cursor) if raw_cursor else None
        )
        raw_fast = source_state.get("last_fast_poll_at")
        self.last_fast_poll_at: Optional[datetime] = (
            datetime.fromisoformat(raw_fast) if raw_fast else None
        )
        raw_recon = source_state.get("last_reconciliation_at")
        self.last_reconciliation_at: Optional[datetime] = (
            datetime.fromisoformat(raw_recon) if raw_recon else None
        )

    def run_fast_poll(self, current_time: Optional[datetime] = None) -> List[CanonicalRawAlert]:
        """Execute fast recent polling path using current poll cursor hint."""
        now = current_time or datetime.now(timezone.utc)
        return self.poller.poll_recent(current_time=now, recent_poll_cursor=self.recent_poll_cursor)

    def run_reconciliation(
        self,
        current_time: Optional[datetime] = None,
        days: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Execute lossless reconciliation scan across retained daily indices."""
        now = current_time or datetime.now(timezone.utc)
        n_days = days or self.reconciliation_days
        return self.poller.poll_reconciliation(
            current_time=now,
            reconciliation_days=n_days,
            start_time=start_time,
            end_time=end_time,
        )

    def run_cycle(
        self,
        current_time: Optional[datetime] = None,
        force_reconciliation: bool = False,
    ) -> LiveCycleResult:
        """Execute a coordinated live ingestion cycle.

        1. Runs reconciliation if due or forced.
        2. Runs fast recent polling.
        3. Merges candidate streams in deterministic order.
        4. Submits each candidate to LiveRBTAService.
        5. Flushes idle buckets in LiveRBTAService.
        6. Updates and persists transport cursor state.

        Parameters
        ----------
        current_time : datetime | None
            Reference time for this cycle (defaults to UTC now).
        force_reconciliation : bool
            Whether to force a reconciliation scan in this cycle.

        Returns
        -------
        LiveCycleResult
            Operational metrics for this cycle.
        """
        now = current_time or datetime.now(timezone.utc)

        # 1. Determine reconciliation run
        should_reconcile = force_reconciliation
        if not should_reconcile:
            if self.last_reconciliation_at is None:
                should_reconcile = True
            elif (now - self.last_reconciliation_at) >= self.reconciliation_interval:
                should_reconcile = True

        recon_alerts: List[CanonicalRawAlert] = []
        if should_reconcile:
            try:
                recon_alerts = self.run_reconciliation(current_time=now)
                self.last_reconciliation_at = now
            except Exception as exc:
                logger.error("Reconciliation scan failed: %s", exc)
                raise

        # 2. Fast recent poll
        fast_alerts: List[CanonicalRawAlert] = []
        try:
            fast_alerts = self.run_fast_poll(current_time=now)
            self.last_fast_poll_at = now
        except Exception as exc:
            logger.error("Fast recent poll failed: %s", exc)
            raise

        # 3. Merge candidate streams with in-cycle deduplication
        all_candidates: List[CanonicalRawAlert] = []
        seen_in_cycle: Set[str] = set()

        for a in list(recon_alerts) + list(fast_alerts):
            if a.wazuh_alert_id not in seen_in_cycle:
                seen_in_cycle.add(a.wazuh_alert_id)
                all_candidates.append(a)

        # Stable sort by timestamp ASC, wazuh_alert_id ASC
        all_candidates.sort(key=lambda a: (a.timestamp, a.wazuh_alert_id))

        # 4. Ingest candidates through LiveRBTAService
        new_ids_count = 0
        duplicate_noops = 0
        failures = 0
        total_scored: List[ScoredMetaAlert] = []

        for candidate in all_candidates:
            is_already_seen = self.service.is_seen(candidate.wazuh_alert_id)
            try:
                scored_list = self.service.ingest_alert(candidate)
                total_scored.extend(scored_list)
                if is_already_seen:
                    duplicate_noops += 1
                else:
                    new_ids_count += 1
            except Exception as exc:
                logger.error("Failed to ingest alert '%s': %s", candidate.wazuh_alert_id, exc)
                failures += 1
                raise

        # 5. Flush idle buckets
        flushed_scored = self.service.check_idle_flush(now)
        total_scored.extend(flushed_scored)

        # 6. Advance fast cursor (as latency hint only)
        if self.recent_poll_cursor is None or now > self.recent_poll_cursor:
            self.recent_poll_cursor = now

        # 7. Persist transport state
        self.service.update_live_source_state({
            "recent_poll_cursor": self.recent_poll_cursor.isoformat() if self.recent_poll_cursor else None,
            "last_fast_poll_at": self.last_fast_poll_at.isoformat() if self.last_fast_poll_at else None,
            "last_reconciliation_at": self.last_reconciliation_at.isoformat() if self.last_reconciliation_at else None,
            "reconciliation_days": self.reconciliation_days,
        })

        return LiveCycleResult(
            fast_candidates=len(fast_alerts),
            reconciliation_candidates=len(recon_alerts),
            submitted_candidates=len(all_candidates),
            duplicate_noops=duplicate_noops,
            processed_new_ids=new_ids_count,
            failures=failures,
            new_scored_meta_alerts=len(total_scored),
        )
