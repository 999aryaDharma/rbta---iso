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
    recent_reconciliation_candidates: int
    full_reconciliation_candidates: int
    submitted_candidates: int
    duplicate_noops: int
    processed_new_ids: int
    failures: int
    new_scored_meta_alerts: int

    @property
    def reconciliation_candidates(self) -> int:
        """Backwards-compatible aggregate reconciliation candidate count."""
        return self.recent_reconciliation_candidates + self.full_reconciliation_candidates


class LiveIngestionCoordinator:
    """Coordinates fast recent polling, recent reconciliation scans, and full-retention sweeps.

    Parameters
    ----------
    service : LiveRBTAService
        Target live stateful runtime service.
    poller : WazuhIndexerLivePoller | None
        Wazuh Indexer poller source.
    fast_poll_interval : timedelta
        Frequency between fast recent polling cycles (default 5 seconds).
    recent_reconciliation_interval : timedelta
        Frequency between recent reconciliation scans (default 5 minutes).
    full_reconciliation_interval : timedelta
        Frequency between exhaustive full-retention sweeps (default 1 hour).
    recent_reconciliation_days : int
        Number of recent daily indices to scan during recent reconciliation (default 2).
    """

    def __init__(
        self,
        service: LiveRBTAService,
        poller: Optional[WazuhIndexerLivePoller] = None,
        fast_poll_interval: timedelta = timedelta(seconds=5),
        recent_reconciliation_interval: timedelta = timedelta(minutes=5),
        full_reconciliation_interval: timedelta = timedelta(hours=1),
        recent_reconciliation_days: int = 2,
        reconciliation_interval: Optional[timedelta] = None,
        reconciliation_days: Optional[int] = None,
    ) -> None:
        self.service: LiveRBTAService = service
        self.poller: WazuhIndexerLivePoller = poller or WazuhIndexerLivePoller()
        self.fast_poll_interval: timedelta = fast_poll_interval
        self.recent_reconciliation_interval: timedelta = (
            reconciliation_interval or recent_reconciliation_interval
        )
        self.full_reconciliation_interval: timedelta = full_reconciliation_interval
        self.recent_reconciliation_days: int = (
            reconciliation_days or recent_reconciliation_days
        )

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

        raw_recent_recon = source_state.get("last_recent_reconciliation_at") or source_state.get(
            "last_reconciliation_at"
        )
        self.last_recent_reconciliation_at: Optional[datetime] = (
            datetime.fromisoformat(raw_recent_recon) if raw_recent_recon else None
        )

        raw_full_recon = source_state.get("last_full_reconciliation_at")
        self.last_full_reconciliation_at: Optional[datetime] = (
            datetime.fromisoformat(raw_full_recon) if raw_full_recon else None
        )

    def run_fast_poll(self, current_time: Optional[datetime] = None) -> List[CanonicalRawAlert]:
        """Execute fast recent polling path using current poll cursor hint."""
        now = current_time or datetime.now(timezone.utc)
        return self.poller.poll_recent(current_time=now, recent_poll_cursor=self.recent_poll_cursor)

    def run_recent_reconciliation(
        self,
        current_time: Optional[datetime] = None,
        days: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Execute recent reconciliation scan across recent daily indices."""
        now = current_time or datetime.now(timezone.utc)
        n_days = days or self.recent_reconciliation_days
        return self.poller.poll_reconciliation(
            current_time=now,
            reconciliation_days=n_days,
            start_time=start_time,
            end_time=end_time,
        )

    def run_reconciliation(
        self,
        current_time: Optional[datetime] = None,
        days: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Backwards-compatible alias for run_recent_reconciliation."""
        return self.run_recent_reconciliation(
            current_time=current_time,
            days=days,
            start_time=start_time,
            end_time=end_time,
        )

    def run_full_reconciliation(self, prefix: str = "wazuh-alerts-4.x-") -> List[CanonicalRawAlert]:
        """Execute lossless full-retention reconciliation across all retained daily indices."""
        return self.poller.poll_full_reconciliation(prefix=prefix)

    def run_cycle(
        self,
        current_time: Optional[datetime] = None,
        force_recent_reconciliation: bool = False,
        force_full_reconciliation: bool = False,
        force_reconciliation: bool = False,
    ) -> LiveCycleResult:
        """Execute a coordinated live ingestion cycle.

        1. Runs full-retention sweep if due or forced.
        2. Runs recent reconciliation if due or forced.
        3. Runs fast recent polling.
        4. Merges candidate streams in deterministic order.
        5. Submits each candidate to LiveRBTAService.
        6. Flushes idle buckets in LiveRBTAService.
        7. Atomically updates and persists transport cursor state on success.

        Parameters
        ----------
        current_time : datetime | None
            Reference time for this cycle (defaults to UTC now).
        force_recent_reconciliation : bool
            Whether to force a recent reconciliation scan.
        force_full_reconciliation : bool
            Whether to force an exhaustive full-retention sweep.
        force_reconciliation : bool
            Backwards-compatible alias for force_recent_reconciliation.

        Returns
        -------
        LiveCycleResult
            Operational metrics for this cycle.
        """
        now = current_time or datetime.now(timezone.utc)

        # 1. Full-retention reconciliation schedule check
        should_full_recon = force_full_reconciliation
        if not should_full_recon:
            if self.last_full_reconciliation_at is None:
                should_full_recon = True
            elif (now - self.last_full_reconciliation_at) >= self.full_reconciliation_interval:
                should_full_recon = True

        full_recon_alerts: List[CanonicalRawAlert] = []
        if should_full_recon:
            full_recon_alerts = self.run_full_reconciliation()

        # 2. Recent reconciliation schedule check
        should_recent_recon = force_recent_reconciliation or force_reconciliation
        if not should_recent_recon:
            if self.last_recent_reconciliation_at is None:
                should_recent_recon = True
            elif (now - self.last_recent_reconciliation_at) >= self.recent_reconciliation_interval:
                should_recent_recon = True

        recent_recon_alerts: List[CanonicalRawAlert] = []
        if should_recent_recon:
            recent_recon_alerts = self.run_recent_reconciliation(current_time=now)

        # 3. Fast recent poll
        fast_alerts: List[CanonicalRawAlert] = self.run_fast_poll(current_time=now)

        # 4. Merge candidate streams with in-cycle deduplication
        all_candidates: List[CanonicalRawAlert] = []
        seen_in_cycle: Set[str] = set()

        for a in list(full_recon_alerts) + list(recent_recon_alerts) + list(fast_alerts):
            if a.wazuh_alert_id not in seen_in_cycle:
                seen_in_cycle.add(a.wazuh_alert_id)
                all_candidates.append(a)

        # Stable sort by timestamp ASC, wazuh_alert_id ASC
        all_candidates.sort(key=lambda a: (a.timestamp, a.wazuh_alert_id))

        # 5. Ingest candidates through LiveRBTAService
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

        # 6. Flush idle buckets
        flushed_scored = self.service.check_idle_flush(now)
        total_scored.extend(flushed_scored)

        # 7. Commit transport state atomically on successful cycle completion
        self.last_fast_poll_at = now
        if should_recent_recon:
            self.last_recent_reconciliation_at = now
        if should_full_recon:
            self.last_full_reconciliation_at = now

        if self.recent_poll_cursor is None or now > self.recent_poll_cursor:
            self.recent_poll_cursor = now

        self.service.update_live_source_state({
            "recent_poll_cursor": self.recent_poll_cursor.isoformat() if self.recent_poll_cursor else None,
            "last_fast_poll_at": self.last_fast_poll_at.isoformat() if self.last_fast_poll_at else None,
            "last_recent_reconciliation_at": (
                self.last_recent_reconciliation_at.isoformat()
                if self.last_recent_reconciliation_at
                else None
            ),
            "last_reconciliation_at": (
                self.last_recent_reconciliation_at.isoformat()
                if self.last_recent_reconciliation_at
                else None
            ),
            "last_full_reconciliation_at": (
                self.last_full_reconciliation_at.isoformat()
                if self.last_full_reconciliation_at
                else None
            ),
            "recent_reconciliation_days": self.recent_reconciliation_days,
        })

        return LiveCycleResult(
            fast_candidates=len(fast_alerts),
            recent_reconciliation_candidates=len(recent_recon_alerts),
            full_reconciliation_candidates=len(full_recon_alerts),
            submitted_candidates=len(all_candidates),
            duplicate_noops=duplicate_noops,
            processed_new_ids=new_ids_count,
            failures=failures,
            new_scored_meta_alerts=len(total_scored),
        )
