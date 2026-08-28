"""Single-bucket deterministic RBTA Engine with agent-local ETW."""

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

from src.config.domain import has_critical_mitre_tactic
from src.config.research import (
    DEFAULT_BASE_DELTA_T,
    MAX_BUCKET_DURATION,
)
from src.contracts.meta_alert import MetaAlert
from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.temporal_state import AgentTemporalState


class RBTAInvariantError(ValueError):
    """Raised when an internal RBTA aggregation invariant is violated."""
    pass


class _ActiveBucket:
    """Internal mutable accumulator for an active aggregation window."""

    def __init__(self, alert: CanonicalRawAlert, meta_id: int) -> None:
        self.meta_id: int = meta_id
        self.agent_id: str = alert.agent_id
        self.agent_name: str = alert.agent_name
        self.rule_group_primary: str = alert.rule_group_primary
        self.start_time: datetime = alert.timestamp
        self.end_time: datetime = alert.timestamp
        self.alert_count: int = 1
        self.max_severity: int = alert.rule_level
        self.rule_id_distribution: Counter[str] = Counter({alert.rule_id: 1})
        self.severity_distribution: Counter[int] = Counter({alert.rule_level: 1})
        self.agent_criticality: int = alert.agent_criticality
        self.wazuh_alert_ids: List[str] = [alert.wazuh_alert_id]

        # Case-insensitive MITRE deduplication preserving first encountered form
        self.mitre_tactics_order: List[str] = []
        self._mitre_seen: Set[str] = set()
        for t in alert.mitre_tactics:
            k = t.casefold()
            if k not in self._mitre_seen:
                self._mitre_seen.add(k)
                self.mitre_tactics_order.append(t)

        self.critical_mitre_present: bool = has_critical_mitre_tactic(self.mitre_tactics_order)

    def add(self, alert: CanonicalRawAlert) -> None:
        """Accumulate a merged alert into this active bucket."""
        self.alert_count += 1
        if alert.rule_level > self.max_severity:
            self.max_severity = alert.rule_level

        self.rule_id_distribution[alert.rule_id] += 1
        self.severity_distribution[alert.rule_level] += 1

        for t in alert.mitre_tactics:
            k = t.casefold()
            if k not in self._mitre_seen:
                self._mitre_seen.add(k)
                self.mitre_tactics_order.append(t)

        if not self.critical_mitre_present and has_critical_mitre_tactic(alert.mitre_tactics):
            self.critical_mitre_present = True

        self.wazuh_alert_ids.append(alert.wazuh_alert_id)

    def finalize(self) -> MetaAlert:
        """Convert this mutable active bucket into an immutable MetaAlert DTO."""
        duration = self.end_time - self.start_time
        if duration < timedelta(0):
            raise RBTAInvariantError(
                f"Negative bucket duration detected: start={self.start_time}, end={self.end_time}"
            )
        if duration > MAX_BUCKET_DURATION:
            raise RBTAInvariantError(
                f"Bucket duration {duration} exceeds maximum allowed {MAX_BUCKET_DURATION}"
            )

        return MetaAlert(
            meta_id=self.meta_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            rule_group_primary=self.rule_group_primary,
            start_time=self.start_time,
            end_time=self.end_time,
            alert_count=self.alert_count,
            max_severity=self.max_severity,
            rule_id_distribution=dict(self.rule_id_distribution),
            severity_distribution=dict(self.severity_distribution),
            mitre_tactics_unique=tuple(self.mitre_tactics_order),
            critical_mitre_present=self.critical_mitre_present,
            agent_criticality=self.agent_criticality,
            wazuh_alert_ids=tuple(self.wazuh_alert_ids),
        )


class RBTAEngine:
    """Deterministic Rule-Based Temporal Aggregation (RBTA) Engine.

    Aggregates incoming CanonicalRawAlert events into immutable MetaAlert buckets
    based on the exact single-bucket key (agent_id, rule_group_primary) and
    agent-local adaptive Elastic Time Windows.

    Parameters
    ----------
    base_delta_t : timedelta
        Experiment baseline temporal window.
    adaptive : bool
        Whether to enable agent-local EMA adaptation after warmup (True) or keep fixed window (False).
    """

    def __init__(
        self,
        base_delta_t: timedelta = DEFAULT_BASE_DELTA_T,
        adaptive: bool = True,
    ) -> None:
        if not isinstance(base_delta_t, timedelta) or base_delta_t.total_seconds() <= 0:
            raise RBTAInvariantError(f"base_delta_t must be a positive timedelta, got {base_delta_t}")

        self._base_delta_t: timedelta = base_delta_t
        self._adaptive: bool = adaptive
        self._temporal_states: Dict[str, AgentTemporalState] = {}
        self._active_buckets: Dict[Tuple[str, str], _ActiveBucket] = {}
        self._seen_alert_ids: Set[str] = set()
        self._meta_id_counter: int = 1

    def _next_meta_id(self) -> int:
        mid = self._meta_id_counter
        self._meta_id_counter += 1
        return mid

    def _get_agent_state(self, agent_id: str) -> AgentTemporalState:
        if agent_id not in self._temporal_states:
            self._temporal_states[agent_id] = AgentTemporalState(
                agent_id=agent_id,
                base_delta_t=self._base_delta_t,
                adaptive=self._adaptive,
            )
        return self._temporal_states[agent_id]

    def process(self, alert: CanonicalRawAlert) -> List[MetaAlert]:
        """Process an incoming canonical raw alert through RBTA aggregation.

        Parameters
        ----------
        alert : CanonicalRawAlert
            Normalized, immutable raw alert.

        Returns
        -------
        List[MetaAlert]
            List of 0 or more finalized MetaAlerts (e.g. from bucket splits or singleton residual events).
        """
        # ── 1. Idempotency Check ──────────────────────────────────────────────
        if alert.wazuh_alert_id in self._seen_alert_ids:
            return []
        self._seen_alert_ids.add(alert.wazuh_alert_id)

        # ── 2. Observe Agent Temporal State ───────────────────────────────────
        agent_state = self._get_agent_state(alert.agent_id)
        current_delta_t = agent_state.observe(alert.timestamp)

        # ── 3. Evaluate Bucket Aggregation ────────────────────────────────────
        bucket_key = (alert.agent_id, alert.rule_group_primary)

        if bucket_key not in self._active_buckets:
            new_bucket = _ActiveBucket(alert, meta_id=self._next_meta_id())
            self._active_buckets[bucket_key] = new_bucket
            return []

        active_bucket = self._active_buckets[bucket_key]

        # Domain Integrity Validation
        if alert.agent_criticality != active_bucket.agent_criticality:
            raise RBTAInvariantError(
                f"Contradictory agent_criticality ({alert.agent_criticality} vs "
                f"{active_bucket.agent_criticality}) for agent '{alert.agent_id}'"
            )

        # ── Case A: Normal Forward / Equal Arrival ────────────────────────────
        if alert.timestamp >= active_bucket.end_time:
            gap = alert.timestamp - active_bucket.end_time
            prospective_duration = alert.timestamp - active_bucket.start_time

            if gap <= current_delta_t and prospective_duration <= MAX_BUCKET_DURATION:
                active_bucket.end_time = alert.timestamp
                active_bucket.add(alert)
                return []
            else:
                # Split: finalize existing bucket, start new active bucket
                finalized_meta = active_bucket.finalize()
                new_bucket = _ActiveBucket(alert, meta_id=self._next_meta_id())
                self._active_buckets[bucket_key] = new_bucket
                return [finalized_meta]

        # ── Case B: In-Window Arrival (start <= timestamp < end) ───────────────
        elif alert.timestamp >= active_bucket.start_time:
            active_bucket.add(alert)
            return []

        # ── Case C: Earlier Residual Arrival (timestamp < start) ──────────────
        else:
            boundary_gap = active_bucket.start_time - alert.timestamp
            prospective_duration = active_bucket.end_time - alert.timestamp

            if boundary_gap <= current_delta_t and prospective_duration <= MAX_BUCKET_DURATION:
                active_bucket.start_time = alert.timestamp
                active_bucket.add(alert)
                return []
            else:
                # Extremely late non-mergeable event -> create immediate singleton
                singleton = _ActiveBucket(alert, meta_id=self._next_meta_id())
                finalized_singleton = singleton.finalize()
                return [finalized_singleton]

    def flush_idle(self, event_time: datetime) -> List[MetaAlert]:
        """Finalize and return active buckets whose idle duration strictly exceeds current_delta_t.

        Parameters
        ----------
        event_time : datetime
            Reference event timestamp to check idle timeout against.

        Returns
        -------
        List[MetaAlert]
            Finalized meta-alerts sorted deterministically by meta_id.
        """
        if event_time.tzinfo is None:
            raise RBTAInvariantError("event_time must be timezone-aware")

        finalized_list: List[MetaAlert] = []
        keys_to_remove: List[Tuple[str, str]] = []

        for key, bucket in self._active_buckets.items():
            if event_time >= bucket.end_time:
                idle_gap = event_time - bucket.end_time
                agent_state = self._get_agent_state(bucket.agent_id)
                # Strict inequality: at gap == delta_t, bucket is still merge-eligible
                if idle_gap > agent_state.current_delta_t:
                    keys_to_remove.append(key)
                    finalized_list.append(bucket.finalize())

        for key in keys_to_remove:
            del self._active_buckets[key]

        finalized_list.sort(key=lambda m: m.meta_id)
        return finalized_list

    def drain(self) -> List[MetaAlert]:
        """Finalize and return all remaining active buckets.

        Subsequent calls on an empty engine return an empty list (idempotent).

        Returns
        -------
        List[MetaAlert]
            All finalized meta-alerts sorted deterministically by meta_id.
        """
        finalized_list: List[MetaAlert] = []
        for bucket in self._active_buckets.values():
            finalized_list.append(bucket.finalize())

        self._active_buckets.clear()
        finalized_list.sort(key=lambda m: m.meta_id)
        return finalized_list
