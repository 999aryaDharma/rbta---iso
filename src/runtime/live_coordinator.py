"""Live Ingestion Coordinator coordinating fast recent polling, reconciliation scans, and durable ingestion."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.rbta.engine import RBTAInvariantError
from src.rbta.temporal_state import TemporalStateError
from src.runtime.live_source import WazuhIndexerLivePoller
from src.runtime.order_buffer import OrderBuffer, partition_unseen
from src.runtime.raw_evidence import RawEvidenceConflictError, RawEvidenceIntegrityError
from src.runtime.service import LiveRBTAService

logger = logging.getLogger(__name__)

#: Deterministic per-record failures quarantined durably (QUARANTINE policy).
#: engine.process is failure-atomic for these: zero commit state is mutated,
#: the alert ID is never marked seen, so a retry reproduces the same error.
_QUARANTINE_ERRORS = (
    RawEvidenceConflictError,
    RawEvidenceIntegrityError,
    RBTAInvariantError,
    TemporalStateError,
)

#: N5 circuit breaker: halt the cycle instead of absorbing systemic corruption.
_QUARANTINE_BREAKER_FRACTION = 0.01
_QUARANTINE_BREAKER_ABSOLUTE = 100

#: M2(c): event timestamps beyond now + this skew are future-dated (clock skew
#: or corrupt event time) and must not shift newest_ingested_event_time.
_FUTURE_EVENT_SKEW = timedelta(minutes=5)


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
    quarantined: int = 0
    skipped_quarantined: int = 0

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
    order_buffer_enabled : bool
        When True, candidates are held in an OrderBuffer and only watermark /
        max_hold releases are ingested, closing the cross-cycle ordering gap
        (late reconciliation alerts vs fast-poll alerts). Default False, which
        preserves the legacy immediate-ingest behavior.
    order_buffer_hold_window : timedelta
        Event-time hold window for the buffer watermark (default 30 seconds).
    order_buffer_max_hold : timedelta
        Maximum wall-clock hold before force-release (default 5 minutes).
    order_buffer_max_items : int
        Bounded buffer capacity; exhaustion triggers backpressure, never drops.
    breaker_ack : str | None
        Operator acknowledgement for a latched breaker trip. The server passes
        ``RBTA_QUARANTINE_BREAKER_ACK`` here; a value matching the latched
        ``trip_id`` clears the latch and resumes cycling, anything else keeps
        halting. Default None (no acknowledgement).

    Operational distinction
    -----------------------
    quarantine = an isolated corrupt record is absorbed and the cycle keeps
    running (one bad apple never blinds ingestion). breaker-latch = systemic
    corruption halted the cycle and persisted a ``breaker_tripped`` latch into
    source state; every subsequent cycle raises immediately — without polling,
    ingesting, or committing — until the operator acknowledges with the
    matching ``trip_id``.
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
        order_buffer_enabled: bool = False,
        order_buffer_hold_window: timedelta = timedelta(seconds=30),
        order_buffer_max_hold: timedelta = timedelta(minutes=5),
        order_buffer_max_items: int = 1000,
        breaker_ack: Optional[str] = None,
        live_archive_enabled: Optional[bool] = None,
        live_archive_dir: Optional[Union[str, Path]] = None,
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
        self.order_buffer: Optional[OrderBuffer] = (
            OrderBuffer(
                hold_window=order_buffer_hold_window,
                max_hold=order_buffer_max_hold,
                max_items=order_buffer_max_items,
            )
            if order_buffer_enabled
            else None
        )
        self.breaker_ack: Optional[str] = breaker_ack

        # Design A: best-effort raw JSONL archive of live-ingested alerts.
        # Default disabled so replay/demo paths are unaffected; enable via
        # RBTA_LIVE_ARCHIVE_ENABLED=1 (dir via RBTA_LIVE_ARCHIVE_DIR).
        if live_archive_enabled is None:
            live_archive_enabled = os.environ.get(
                "RBTA_LIVE_ARCHIVE_ENABLED", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        self.live_archive_enabled: bool = bool(live_archive_enabled)
        if live_archive_dir is None:
            env_dir = os.environ.get("RBTA_LIVE_ARCHIVE_DIR", "").strip()
            live_archive_dir = env_dir if env_dir else "data/archive"
        self.live_archive_dir: Path = Path(live_archive_dir)

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
        """Execute lossless full-retention reconciliation across all retained daily indices.

        Strict index coverage: an unavailable discovered index fails hard
        instead of being silently skipped as an apparent empty result.
        """
        return self.poller.poll_full_reconciliation(prefix=prefix, allow_unavailable=False)

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
        6. Flushes the per-cycle quarantine batch once, then checks the breaker.
        7. Flushes idle buckets in LiveRBTAService.
        8. Atomically updates and persists transport cursor state on success.

        Deterministic per-record errors (RawEvidenceConflictError,
        RawEvidenceIntegrityError, RBTAInvariantError, TemporalStateError)
        plus per-hit poller bad documents (LiveCanonicalizationError collected
        in ``poller.last_bad_docs``) are quarantined durably in a single
        batched flush and skipped — the cycle continues so one corrupt record
        cannot blind live ingestion (QUARANTINE policy, not halt). Only the
        first offense for an ID logs at error level; repeats log at debug.

        M2(b): candidates whose ID is already quarantined are filtered before
        the order buffer (one quarantine_id_set snapshot per cycle, a single
        ID-only query, reported as skipped_quarantined) so persistently corrupt
        alerts are never re-ingested — and never occupy the buffer. Without
        the skip, re-polling a persistently corrupt alert re-quarantines it
        every cycle and the breaker below trips forever — a permanent halt
        with a frozen cursor. Skipped IDs add no failures.
        Transient errors still raise, the cursor does not advance, and the
        worker backs off.

        Circuit breaker: when NEWLY quarantined candidates exceed 1% of
        submitted (or 100 in one cycle) the cycle latches
        ``breaker_tripped`` (trip_id, quarantined, submitted, at) into source
        state via update_live_source_state and raises RuntimeError without
        committing the cursor — systemic corruption halts instead of being
        quietly absorbed, while the quarantine flush itself stays durably
        stored. Every later cycle raises on the latch — no polling, no
        ingest, no commit — until constructed with the matching breaker_ack,
        which clears the latch and resumes. Re-encountered
        (already-quarantined) IDs never reach the flush, and repeats that do
        (same ID twice in one cycle) return count > 1 from quarantine_add_many
        and are excluded from the breaker numerator, so only genuinely new
        corruption can trip it.

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

        # P1 breaker-latch: a previous trip persists {"breaker_tripped":
        # {"trip_id", "quarantined", "submitted", "at"}} in source state and
        # halts every cycle — no polling, no ingest, no commit — until the
        # operator acknowledges with the matching trip_id (falsy/None = no
        # latch, cycle proceeds). A matching ack clears the latch and resumes.
        latch = self.service.get_live_source_state().get("breaker_tripped")
        if latch:
            trip_id = latch.get("trip_id") if isinstance(latch, dict) else None
            if self.breaker_ack != trip_id:
                raise RuntimeError(
                    f"Quarantine breaker latch active (trip_id={trip_id}); "
                    f"cycle halted without work or commit — acknowledge with "
                    f"breaker_ack='{trip_id}' to resume"
                )
            self.service.update_live_source_state({"breaker_tripped": None})

        # 1. Full-retention reconciliation schedule check
        should_full_recon = force_full_reconciliation
        if not should_full_recon:
            if self.last_full_reconciliation_at is None:
                should_full_recon = True
            elif (now - self.last_full_reconciliation_at) >= self.full_reconciliation_interval:
                should_full_recon = True

        # N1b: per-hit bad documents drained from the poller after each poll
        # path (poller resets last_bad_docs on every paginate call).
        bad_docs: List[Dict[str, Any]] = []

        def _drain_bad_docs() -> None:
            raw = getattr(self.poller, "last_bad_docs", None)
            if isinstance(raw, list) and raw:
                bad_docs.extend(d for d in raw if isinstance(d, dict))

        full_recon_alerts: List[CanonicalRawAlert] = []
        if should_full_recon:
            full_recon_alerts = self.run_full_reconciliation()
            _drain_bad_docs()

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
            _drain_bad_docs()

        # 3. Fast recent poll
        fast_alerts: List[CanonicalRawAlert] = self.run_fast_poll(current_time=now)
        _drain_bad_docs()

        # 4. Merge candidate streams with in-cycle deduplication
        all_candidates: List[CanonicalRawAlert] = []
        seen_in_cycle: Set[str] = set()

        for a in list(full_recon_alerts) + list(recent_recon_alerts) + list(fast_alerts):
            if a.wazuh_alert_id not in seen_in_cycle:
                seen_in_cycle.add(a.wazuh_alert_id)
                all_candidates.append(a)

        # Stable sort by timestamp ASC, wazuh_alert_id ASC
        all_candidates.sort(key=lambda a: (a.timestamp, a.wazuh_alert_id))

        # M2(b): snapshot already-quarantined IDs once per cycle (single
        # query, ID column only) and filter BEFORE the order buffer, so the
        # buffer never holds persistently corrupt alerts. Without the skip,
        # re-polling them re-quarantines every cycle and the breaker below
        # trips forever — a permanent halt with a frozen cursor. Skipped IDs
        # add no failures.
        quarantined_ids: Set[str] = self.service.state_manager.quarantine_id_set()
        skipped_quarantined = 0
        if quarantined_ids:
            kept: List[CanonicalRawAlert] = []
            for candidate in all_candidates:
                if candidate.wazuh_alert_id in quarantined_ids:
                    logger.debug(
                        "Skipping already-quarantined alert '%s'; no re-ingest attempted",
                        candidate.wazuh_alert_id,
                    )
                    skipped_quarantined += 1
                else:
                    kept.append(candidate)
            all_candidates = kept

        # 4b. Optional order buffer: hold on-time candidates, release only what
        # is due by watermark / max_hold plus immediate late passthrough.
        # Alerts refused under backpressure are ingested directly (sorted) so
        # no valid alert is dropped; the event is metric-visible and logged.
        if self.order_buffer is not None:
            # F7b: already-seen duplicates bypass the buffer straight to
            # ingest (conflict check still applies) so reconciliation
            # re-polls never pollute the late metric.
            # Minor is_seen 2N: snapshot membership once per candidate per
            # cycle — partition and the ingest loop share this cache instead
            # of hitting SQLite twice. (service exposes no bulk loader;
            # loading the full seen table would be O(table), so per-cycle
            # memoization is the cheap correct snapshot.)
            seen_cache: Dict[str, bool] = {}

            def _cached_is_seen(alert_id: str) -> bool:
                if alert_id not in seen_cache:
                    seen_cache[alert_id] = self.service.is_seen(alert_id)
                return seen_cache[alert_id]

            unseen, already_seen = partition_unseen(all_candidates, _cached_is_seen)
            buffered_releases = self.order_buffer.add(unseen, now=now)
            refused = self.order_buffer.pop_refused()
            if refused:
                logger.warning(
                    "Order buffer backpressure: %d candidate(s) refused "
                    "(backpressure_count=%d); ingesting directly to avoid loss",
                    len(refused),
                    self.order_buffer.status()["backpressure_count"],
                )
                refused.sort(key=lambda a: (a.timestamp, a.wazuh_alert_id))
            all_candidates = [r.alert for r in buffered_releases] + refused + already_seen

        # 5. Ingest candidates through LiveRBTAService
        new_ids_count = 0
        duplicate_noops = 0
        failures = 0
        total_scored: List[ScoredMetaAlert] = []
        ingested_event_times: List[datetime] = []
        newly_ingested: List[CanonicalRawAlert] = []
        future_cutoff = now + _FUTURE_EVENT_SKEW

        # N5b: quarantine entries are collected per cycle and flushed once
        # (single transaction) before the cursor commit. Messages are kept
        # alongside for first-offense error logging.
        pending_quarantine: List[Dict[str, Any]] = []
        pending_messages: List[str] = []

        use_seen_cache = self.order_buffer is not None

        for candidate in all_candidates:
            if use_seen_cache and candidate.wazuh_alert_id in seen_cache:
                is_already_seen = seen_cache[candidate.wazuh_alert_id]
            else:
                is_already_seen = self.service.is_seen(candidate.wazuh_alert_id)
            try:
                scored_list = self.service.ingest_alert(candidate)
                total_scored.extend(scored_list)
                if is_already_seen:
                    duplicate_noops += 1
                else:
                    new_ids_count += 1
                    if use_seen_cache:
                        seen_cache[candidate.wazuh_alert_id] = True
                # Design A: only newly ingested IDs are archived, so the
                # archive stays duplicate-free for later replay.
                if not is_already_seen:
                    newly_ingested.append(candidate)
                # M2(c): only successfully ingested, non-future-dated event
                # times may advance newest_ingested_event_time.
                if candidate.timestamp.tzinfo is None or candidate.timestamp <= future_cutoff:
                    ingested_event_times.append(candidate.timestamp)
            except _QUARANTINE_ERRORS as exc:
                meta = dict(candidate.metadata)
                pending_quarantine.append({
                    "wazuh_alert_id": candidate.wazuh_alert_id,
                    "source_index": str(meta.get("source_index", "") or ""),
                    "source_document_id": str(meta.get("source_document_id", "") or ""),
                    "error_type": type(exc).__name__,
                })
                pending_messages.append(str(exc))
                failures += 1
                continue
            except Exception as exc:
                logger.error("Failed to ingest alert '%s': %s", candidate.wazuh_alert_id, exc)
                failures += 1
                raise

        # Design A: best-effort raw JSONL archive of this cycle's newly
        # ingested alerts (sorted, daily-rotated inside append_alerts).
        # append_alerts never raises; this belt-and-suspenders guard keeps
        # the live cycle green no matter what the archive path does.
        if self.live_archive_enabled and newly_ingested:
            try:
                from src.runtime.live_archive import append_alerts
                append_alerts(newly_ingested, self.live_archive_dir)
            except Exception as exc:
                logger.warning("Live archive batch failed: %s", exc)

        # N1b: every per-hit bad document is quarantined, then the cycle continues.
        # Unidentified docs key by content hash over {index, doc_id} ONLY —
        # page_pos (embedded in the error text) is excluded so a re-polled
        # identical doc maps to the same stable key across cycles and the
        # repeat is skipped as already-quarantined instead of being
        # re-quarantined under a new positional key each cycle.
        for bad in bad_docs:
            doc_id = bad.get("doc_id")
            if doc_id:
                quarantine_key = str(doc_id)
            else:
                digest = hashlib.sha256(
                    json.dumps(
                        {"index": bad.get("index"), "doc_id": bad.get("doc_id")},
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()[:12]
                quarantine_key = f"unidentified:{digest}"
            if quarantine_key in quarantined_ids:
                logger.debug(
                    "Skipping already-quarantined bad document '%s'; no re-ingest attempted",
                    quarantine_key,
                )
                continue
            pending_quarantine.append({
                "wazuh_alert_id": quarantine_key,
                "source_index": str(bad.get("index") or ""),
                "source_document_id": str(doc_id or ""),
                "error_type": "LiveCanonicalizationError",
            })
            pending_messages.append(str(bad.get("error") or ""))
            failures += 1

        # 6. Single batched quarantine flush (before the cursor commit) + breaker.
        quarantined = 0
        if pending_quarantine:
            counts = self.service.state_manager.quarantine_add_many(pending_quarantine)
            new_quarantined = 0
            for entry, message, count in zip(pending_quarantine, pending_messages, counts):
                if count <= 1:
                    new_quarantined += 1
                    logger.error(
                        "Quarantining alert '%s' (%s): %s",
                        entry["wazuh_alert_id"], entry["error_type"], message,
                    )
                else:
                    logger.debug(
                        "Quarantining repeat alert '%s' (%s, offense #%d)",
                        entry["wazuh_alert_id"], entry["error_type"], count,
                    )
            quarantined = len(pending_quarantine)

            # M2(a): the breaker counts only NEW quarantines (count == 1).
            submitted = len(all_candidates)
            fraction = (new_quarantined / submitted) if submitted else 0.0
            if new_quarantined > _QUARANTINE_BREAKER_ABSOLUTE or (
                submitted and fraction > _QUARANTINE_BREAKER_FRACTION
            ):
                trip_id = f"qb-{now.strftime('%Y%m%d%H%M%S')}-{new_quarantined}"
                # Latch BEFORE raising: the quarantine flush above stays
                # durably stored, the cursor is NOT committed, and every
                # later cycle halts on the latch until the operator
                # acknowledges with this trip_id.
                self.service.update_live_source_state({
                    "breaker_tripped": {
                        "trip_id": trip_id,
                        "quarantined": new_quarantined,
                        "submitted": submitted,
                        "at": now.isoformat(),
                    },
                })
                raise RuntimeError(
                    f"Quarantine circuit breaker tripped: {new_quarantined}/{submitted} "
                    f"new candidates quarantined in one cycle (fraction={fraction:.2%}, "
                    f"threshold=1% or 100); halting without committing the cursor — "
                    f"trip_id={trip_id}; inspect quarantined_alerts and acknowledge "
                    f"with breaker_ack='{trip_id}' before resuming"
                )

        # 7. Flush idle buckets
        flushed_scored = self.service.check_idle_flush(now)
        total_scored.extend(flushed_scored)

        # 8. Commit transport state atomically on successful cycle completion
        self.last_fast_poll_at = now
        if should_recent_recon:
            self.last_recent_reconciliation_at = now
        if should_full_recon:
            self.last_full_reconciliation_at = now

        if self.recent_poll_cursor is None or now > self.recent_poll_cursor:
            self.recent_poll_cursor = now

        # M2(c): max over successfully ingested, non-future-dated candidates
        # only; quarantined/corrupt and future-dated timestamps never shift
        # lag telemetry. Explicit None when nothing qualifies (never stale).
        newest_event = max(ingested_event_times, default=None)
        self.service.update_live_source_state({
            "recent_poll_cursor": self.recent_poll_cursor.isoformat() if self.recent_poll_cursor else None,
            "newest_ingested_event_time": newest_event.isoformat() if newest_event else None,
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
            quarantined=quarantined,
            skipped_quarantined=skipped_quarantined,
        )
