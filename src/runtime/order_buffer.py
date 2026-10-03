"""Order buffer: waiting-room sorter closing the cross-cycle ordering gap.

LiveIngestionCoordinator sorts candidates only within a single cycle, so a
late alert from reconciliation (older event-time, arriving in a later cycle)
would be ingested after newer fast-poll alerts. OrderBuffer holds on-time
candidates briefly and releases them in ``(timestamp, wazuh_alert_id)`` order.

Release rules
-------------
* ``watermark = max observed event-time - hold_window``. Held items with
  ``timestamp <= watermark`` are released in order.
* Items held longer than ``max_hold`` (wall-clock, ``now - arrived_at``) are
  force-released in order so a quiet stream cannot wedge the buffer.
* An arriving alert older than the current watermark is released immediately
  flagged ``late=True``. It is still processed downstream — per the locked
  rule, no valid alert may be dropped merely for arriving out of order.
* Honest scope (F7): ``hold_window`` only smooths jitter *smaller* than the
  window (cross-cycle reordering of seconds). Reconciliation delays of
  minutes/hours are *expected-late by design*: those alerts pass through
  immediately flagged ``late=True`` instead of being held. The buffer never
  claims to eliminate late data, only to order the on-time fraction.
* Event-times beyond ``now + future_tolerance`` (default 5 minutes) are
  treated as clock anomalies: they pass through immediately flagged
  ``late=True`` under the ``future_anomalies`` counter and never move the
  watermark, so one bad timestamp cannot wedge ordering for the whole stream.
* Capacity is bounded by ``max_items``. When full, new on-time arrivals are
  refused (backpressure): ``backpressure_count`` is incremented,
  ``buffer_dropped`` stays ``0``, and the refused alerts are handed back via
  :meth:`pop_refused` so the caller can retry or ingest them directly.
  Nothing is silently discarded. Honest trade-off (F7): refused alerts are
  intentionally *not* auto-ingested by the buffer — the caller decides
  (retry-later preserves ordering, ingest-now preserves freshness). Either
  way the buffer itself drops nothing.

Checkpoint format (restart recovery design; durable persistence is deferred
to the integration phase, the in-memory buffer intentionally does not touch
disk here)::

    {
        "max_observed_event_time": "<ISO-8601 str>" | None,
        "buffered": [
            {
                "wazuh_alert_id": str,
                "timestamp": "<ISO-8601 str>",
                "arrived_at": "<ISO-8601 str>",
                "agent_id": str,
                "agent_name": str,
                "rule_group_primary": str,
                "rule_level": int,
                "rule_id": str,
                "mitre_tactics": [str, ...],
                "srcip": str | None,
                "agent_criticality": int,
                "metadata": {JSON-safe mapping},
            },
            ...
        ],
        "late_total": int,
        "backpressure_count": int,
        "future_anomalies": int,
    }

A coordinator checkpoint stores the Indexer transport cursor plus this
snapshot; on restart the buffer is rebuilt with :meth:`from_checkpoint`
before polling resumes, so unreleased alerts are re-held rather than lost
(exactly-once effect continues to rely on the existing service dedup).
"""

from bisect import insort
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.json_safe import to_json_safe


@dataclass(frozen=True)
class BufferedRelease:
    """One alert released by the buffer with its lateness flag."""

    alert: CanonicalRawAlert
    late: bool = False


def partition_unseen(
    candidates: List[CanonicalRawAlert],
    is_seen_fn: Callable[[str], bool],
) -> Tuple[List[CanonicalRawAlert], List[CanonicalRawAlert]]:
    """Split candidates into ``(unseen, seen)`` using an id-membership predicate.

    Pure helper for the coordinator dedup pre-pass (F7): input order is
    preserved in both outputs, nothing is dropped, and the predicate is
    called exactly once per candidate. ``is_seen_fn`` receives the
    ``wazuh_alert_id`` and returns True when already processed.
    """
    unseen: List[CanonicalRawAlert] = []
    seen: List[CanonicalRawAlert] = []
    for alert in candidates:
        (seen if is_seen_fn(alert.wazuh_alert_id) else unseen).append(alert)
    return unseen, seen


class OrderBuffer:
    """Bounded waiting-room sorter keyed on event-time with late passthrough."""

    def __init__(
        self,
        hold_window: timedelta = timedelta(seconds=30),
        max_hold: timedelta = timedelta(minutes=5),
        max_items: int = 1000,
        future_tolerance: timedelta = timedelta(minutes=5),
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.hold_window: timedelta = hold_window
        self.max_hold: timedelta = max_hold
        self.max_items: int = max_items
        self.future_tolerance: timedelta = future_tolerance
        self._held: List[Tuple[datetime, str, CanonicalRawAlert, datetime]] = []
        self._held_ids: set = set()
        self._max_observed: Optional[datetime] = None
        self._late_total: int = 0
        self._backpressure_count: int = 0
        self._future_anomalies: int = 0
        self._refused: List[CanonicalRawAlert] = []

    def _watermark(self) -> Optional[datetime]:
        if self._max_observed is None:
            return None
        return self._max_observed - self.hold_window

    def add(
        self,
        alerts: List[CanonicalRawAlert],
        now: Optional[datetime] = None,
    ) -> List[BufferedRelease]:
        """Accept arrivals; return the releases due in ``(timestamp, id)`` order."""
        current = now or datetime.now(timezone.utc)
        released: List[BufferedRelease] = []
        seen_in_call: set = set()
        refused_now: List[CanonicalRawAlert] = []

        for alert in alerts:
            if alert.wazuh_alert_id in seen_in_call or alert.wazuh_alert_id in self._held_ids:
                continue
            seen_in_call.add(alert.wazuh_alert_id)
            if alert.timestamp > current + self.future_tolerance:
                # F7: clock anomaly — passthrough flagged late, watermark untouched.
                released.append(BufferedRelease(alert=alert, late=True))
                self._late_total += 1
                self._future_anomalies += 1
                continue
            if self._max_observed is None or alert.timestamp > self._max_observed:
                self._max_observed = alert.timestamp
            watermark = self._watermark()
            assert watermark is not None
            if alert.timestamp <= watermark:
                released.append(BufferedRelease(alert=alert, late=True))
                self._late_total += 1
            elif len(self._held) < self.max_items:
                insort(self._held, (alert.timestamp, alert.wazuh_alert_id, alert, current))
                self._held_ids.add(alert.wazuh_alert_id)
            else:
                refused_now.append(alert)

        if refused_now:
            # Backpressure: intake refused, never dropped. The caller retrieves
            # these via pop_refused() and must retry them on a later cycle.
            self._refused.extend(refused_now)
            self._backpressure_count += 1

        watermark = self._watermark()
        if watermark is not None or self._held:
            still_held: List[Tuple[datetime, str, CanonicalRawAlert, datetime]] = []
            for ts, aid, alert, arrived in self._held:
                due_by_watermark = watermark is not None and ts <= watermark
                due_by_max_hold = (current - arrived) >= self.max_hold
                if due_by_watermark or due_by_max_hold:
                    released.append(BufferedRelease(alert=alert, late=False))
                    self._held_ids.discard(aid)
                else:
                    still_held.append((ts, aid, alert, arrived))
            self._held = still_held

        released.sort(key=lambda r: (r.alert.timestamp, r.alert.wazuh_alert_id))
        return released

    def flush(self) -> List[BufferedRelease]:
        """Release everything still held, in ``(timestamp, id)`` order."""
        released = [BufferedRelease(alert=a, late=False) for _, _, a, _ in self._held]
        self._held = []
        self._held_ids = set()
        return released

    def pop_refused(self) -> List[CanonicalRawAlert]:
        """Take back alerts refused under backpressure (caller must retry them)."""
        refused, self._refused = self._refused, []
        return refused

    def status(self) -> Dict[str, Any]:
        """Operational counters; ``buffer_dropped`` is always 0 (never drop)."""
        return {
            "size": len(self._held),
            "late_total": self._late_total,
            "backpressure_count": self._backpressure_count,
            "future_anomalies": self._future_anomalies,
            "buffer_dropped": 0,
            "max_items": self.max_items,
        }

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize held items plus watermark/counters (see module docstring)."""
        return {
            "max_observed_event_time": (
                self._max_observed.isoformat() if self._max_observed else None
            ),
            "buffered": [
                {
                    "wazuh_alert_id": alert.wazuh_alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "arrived_at": arrived.isoformat(),
                    "agent_id": alert.agent_id,
                    "agent_name": alert.agent_name,
                    "rule_group_primary": alert.rule_group_primary,
                    "rule_level": alert.rule_level,
                    "rule_id": alert.rule_id,
                    "mitre_tactics": list(alert.mitre_tactics),
                    "srcip": alert.srcip,
                    "agent_criticality": alert.agent_criticality,
                    "metadata": to_json_safe(dict(alert.metadata)),
                }
                for _, _, alert, arrived in self._held
            ],
            "late_total": self._late_total,
            "backpressure_count": self._backpressure_count,
            "future_anomalies": self._future_anomalies,
        }

    @classmethod
    def from_checkpoint(
        cls,
        snapshot: Dict[str, Any],
        hold_window: timedelta = timedelta(seconds=30),
        max_hold: timedelta = timedelta(minutes=5),
        max_items: int = 1000,
    ) -> "OrderBuffer":
        """Rebuild a buffer from :meth:`to_checkpoint` output."""
        buf = cls(hold_window=hold_window, max_hold=max_hold, max_items=max_items)
        raw_max = snapshot.get("max_observed_event_time")
        buf._max_observed = datetime.fromisoformat(raw_max) if raw_max else None
        for entry in snapshot.get("buffered", []):
            alert = CanonicalRawAlert(
                wazuh_alert_id=entry["wazuh_alert_id"],
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                agent_id=entry["agent_id"],
                agent_name=entry["agent_name"],
                rule_group_primary=entry["rule_group_primary"],
                rule_level=entry["rule_level"],
                rule_id=entry["rule_id"],
                mitre_tactics=tuple(entry.get("mitre_tactics", ())),
                srcip=entry.get("srcip"),
                agent_criticality=entry["agent_criticality"],
                metadata=entry.get("metadata", {}),
            )
            arrived = datetime.fromisoformat(entry["arrived_at"])
            insort(buf._held, (alert.timestamp, alert.wazuh_alert_id, alert, arrived))
            buf._held_ids.add(alert.wazuh_alert_id)
        buf._late_total = int(snapshot.get("late_total", 0))
        buf._backpressure_count = int(snapshot.get("backpressure_count", 0))
        buf._future_anomalies = int(snapshot.get("future_anomalies", 0))
        return buf
