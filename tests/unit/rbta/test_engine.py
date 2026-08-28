"""Unit tests for single-bucket deterministic RBTAEngine (Task 2.3)."""
from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.meta_alert import MetaAlert
from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine, RBTAInvariantError


def make_alert(
    alert_id: str,
    ts: datetime,
    agent_id: str = "001",
    agent_name: str = "soc-1",
    group: str = "pam",
    level: int = 3,
    rule_id: str = "5501",
    mitre: tuple[str, ...] = (),
    crit: int = 1,
) -> CanonicalRawAlert:
    """Helper to create CanonicalRawAlert with specific fields."""
    return CanonicalRawAlert(
        wazuh_alert_id=alert_id,
        timestamp=ts,
        agent_id=agent_id,
        agent_name=agent_name,
        rule_group_primary=group,
        rule_level=level,
        rule_id=rule_id,
        mitre_tactics=mitre,
        srcip="192.168.1.50",
        agent_criticality=crit,
    )


# ── Bucket Key Tests ─────────────────────────────────────────────────────────

def test_same_agent_same_group_aggregates_into_same_bucket():
    """Alerts for same agent and same rule group within delta_t merge into one bucket."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t, agent_id="001", group="pam")
    a2 = make_alert("2", base_t + timedelta(minutes=5), agent_id="001", group="pam")

    out1 = engine.process(a1)
    out2 = engine.process(a2)
    assert out1 == []
    assert out2 == []

    meta_list = engine.drain()
    assert len(meta_list) == 1
    meta = meta_list[0]
    assert meta.agent_id == "001"
    assert meta.rule_group_primary == "pam"
    assert meta.alert_count == 2
    assert meta.wazuh_alert_ids == ("1", "2")


def test_same_agent_different_group_creates_different_buckets():
    """Alerts for same agent but different rule groups create separate active buckets."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t, agent_id="001", group="pam")
    a2 = make_alert("2", base_t + timedelta(minutes=2), agent_id="001", group="syslog")

    engine.process(a1)
    engine.process(a2)

    meta_list = engine.drain()
    assert len(meta_list) == 2
    groups = {m.rule_group_primary for m in meta_list}
    assert groups == {"pam", "syslog"}


def test_different_agent_same_group_creates_different_buckets():
    """Alerts for different agents with same rule group create separate active buckets."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t, agent_id="001", group="pam")
    a2 = make_alert("2", base_t + timedelta(minutes=2), agent_id="002", group="pam")

    engine.process(a1)
    engine.process(a2)

    meta_list = engine.drain()
    assert len(meta_list) == 2
    agent_ids = {m.agent_id for m in meta_list}
    assert agent_ids == {"001", "002"}


# ── Merge / Split Boundary Semantics Tests ───────────────────────────────────

def test_gap_equal_to_delta_t_merges():
    """Inclusive boundary: gap == delta_t MUST merge."""
    base_dt = timedelta(minutes=10)
    engine = RBTAEngine(base_delta_t=base_dt)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t)
    # Exactly 10 minutes gap
    a2 = make_alert("2", base_t + timedelta(minutes=10))

    out1 = engine.process(a1)
    out2 = engine.process(a2)
    assert out1 == []
    assert out2 == []

    meta_list = engine.drain()
    assert len(meta_list) == 1
    assert meta_list[0].alert_count == 2


def test_gap_greater_than_delta_t_splits():
    """gap > delta_t MUST split (finalize previous bucket and create new one)."""
    base_dt = timedelta(minutes=10)
    engine = RBTAEngine(base_delta_t=base_dt)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t)
    # 10 minutes + 1 second -> split
    a2 = make_alert("2", base_t + timedelta(minutes=10, seconds=1))

    out1 = engine.process(a1)
    assert out1 == []
    out2 = engine.process(a2)
    assert len(out2) == 1
    assert out2[0].wazuh_alert_ids == ("1",)

    meta_list = engine.drain()
    assert len(meta_list) == 1
    assert meta_list[0].wazuh_alert_ids == ("2",)


def test_max_duration_60_minutes_boundary():
    """Prospective duration <= 60 minutes merges; duration > 60 minutes splits."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # Alerts arrive every 15 minutes (gap == delta_t)
    a1 = make_alert("1", base_t)
    a2 = make_alert("2", base_t + timedelta(minutes=15))
    a3 = make_alert("3", base_t + timedelta(minutes=30))
    a4 = make_alert("4", base_t + timedelta(minutes=45))
    # Exactly 60 minutes from start_time -> duration == 60m (eligible)
    a5 = make_alert("5", base_t + timedelta(minutes=60))

    assert engine.process(a1) == []
    assert engine.process(a2) == []
    assert engine.process(a3) == []
    assert engine.process(a4) == []
    out5 = engine.process(a5)
    assert out5 == []

    # 60m + 1 second (gap 1s <= 15m, but prospective duration > 60m) -> splits!
    a6 = make_alert("6", base_t + timedelta(minutes=60, seconds=1))
    out6 = engine.process(a6)
    assert len(out6) == 1
    assert out6[0].alert_count == 5
    assert out6[0].duration_sec == 3600.0
    assert out6[0].wazuh_alert_ids == ("1", "2", "3", "4", "5")

    drained = engine.drain()
    assert len(drained) == 1
    assert drained[0].wazuh_alert_ids == ("6",)


# ── Residual Out-of-Order Semantics Tests ─────────────────────────────────────

def test_earlier_residual_event_within_delta_t_expands_start_time():
    """An earlier event within delta_t from bucket.start_time expands start_time safely."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=10))
    base_t = datetime(2026, 8, 28, 10, 10, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t)
    engine.process(a1)

    # a0 arrives with timestamp 5 minutes earlier than start_time
    a0 = make_alert("0", base_t - timedelta(minutes=5))
    out0 = engine.process(a0)
    assert out0 == []

    meta_list = engine.drain()
    assert len(meta_list) == 1
    meta = meta_list[0]
    assert meta.alert_count == 2
    assert meta.start_time == base_t - timedelta(minutes=5)
    assert meta.end_time == base_t
    assert meta.duration_sec == 300.0


def test_extremely_late_non_mergeable_event_creates_immediate_singleton():
    """An extremely late event that cannot merge creates an immediate singleton without corrupting active bucket."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=10))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # Active bucket at 10:00
    a1 = make_alert("1", base_t)
    engine.process(a1)

    # Late event 5 hours in the past
    a_late = make_alert("LATE", base_t - timedelta(hours=5))
    out_late = engine.process(a_late)

    # Immediately finalized singleton
    assert len(out_late) == 1
    assert out_late[0].wazuh_alert_ids == ("LATE",)
    assert out_late[0].alert_count == 1

    # Active bucket a1 remains intact
    drained = engine.drain()
    assert len(drained) == 1
    assert drained[0].wazuh_alert_ids == ("1",)


# ── Aggregation & Field Correctness Tests ──────────────────────────────────────

def test_aggregation_distributions_max_severity_and_mitre():
    """Engine must accumulate distributions, max severity, case-insensitive MITRE tactics, and critical flag."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t, level=3, rule_id="5501", mitre=("Execution",))
    a2 = make_alert("2", base_t + timedelta(minutes=2), level=7, rule_id="5502", mitre=("execution", "Initial Access"))
    a3 = make_alert("3", base_t + timedelta(minutes=4), level=5, rule_id="5501", mitre=("Defense Evasion",))

    engine.process(a1)
    engine.process(a2)
    engine.process(a3)

    meta = engine.drain()[0]
    assert meta.alert_count == 3
    assert meta.max_severity == 7
    assert meta.rule_id_distribution == {"5501": 2, "5502": 1}
    assert meta.severity_distribution == {3: 1, 7: 1, 5: 1}
    assert meta.mitre_tactics_unique == ("Execution", "Initial Access", "Defense Evasion")
    assert meta.critical_mitre_present is True


def test_contradictory_agent_criticality_raises_error():
    """Contradictory agent_criticality for the same agent/bucket must fail fast."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t, agent_id="001", crit=1)
    a2 = make_alert("2", base_t + timedelta(minutes=2), agent_id="001", crit=3)

    engine.process(a1)
    with pytest.raises(RBTAInvariantError, match="Contradictory agent_criticality"):
        engine.process(a2)


# ── Idempotency Tests ─────────────────────────────────────────────────────────

def test_duplicate_wazuh_alert_id_is_idempotent_no_op():
    """Duplicate ingress with identical wazuh_alert_id must produce zero additional mutation."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("dup-1", base_t)
    out1 = engine.process(a1)
    assert out1 == []

    # Process exact same alert again
    out_dup = engine.process(a1)
    assert out_dup == []

    meta_list = engine.drain()
    assert len(meta_list) == 1
    assert meta_list[0].alert_count == 1
    assert meta_list[0].wazuh_alert_ids == ("dup-1",)


# ── Flush Idle Tests ──────────────────────────────────────────────────────────

def test_flush_idle_strict_greater_than_delta_t():
    """flush_idle only finalizes when idle_gap > current_delta_t (not at idle_gap == current_delta_t)."""
    base_dt = timedelta(minutes=10)
    engine = RBTAEngine(base_delta_t=base_dt)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t)
    engine.process(a1)

    # Exactly 10 minutes idle -> idle_gap == delta_t -> still merge eligible -> DO NOT FLUSH
    flush_exact = engine.flush_idle(base_t + timedelta(minutes=10))
    assert flush_exact == []

    # 10 minutes + 1 second idle -> idle_gap > delta_t -> FLUSH
    flush_over = engine.flush_idle(base_t + timedelta(minutes=10, seconds=1))
    assert len(flush_over) == 1
    assert flush_over[0].wazuh_alert_ids == ("1",)

    # Subsequent flush returns empty
    assert engine.flush_idle(base_t + timedelta(minutes=20)) == []


# ── Drain and Determinism Tests ───────────────────────────────────────────────

def test_drain_is_idempotent():
    """drain() finalizes all active buckets and clears state; subsequent drains return empty."""
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    engine.process(make_alert("1", base_t))
    engine.process(make_alert("2", base_t, group="syslog"))

    d1 = engine.drain()
    assert len(d1) == 2

    d2 = engine.drain()
    assert d2 == []


def test_deterministic_meta_ids_and_repeated_runs():
    """Identical input sequence through fresh engines produces identical MetaAlert objects and IDs."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [
        make_alert("1", base_t, group="pam"),
        make_alert("2", base_t + timedelta(minutes=5), group="pam"),
        make_alert("3", base_t + timedelta(minutes=30), group="pam"),
        make_alert("4", base_t + timedelta(minutes=2), group="syslog"),
    ]

    engine1 = RBTAEngine(base_delta_t=timedelta(minutes=15))
    out1 = []
    for a in alerts:
        out1.extend(engine1.process(a))
    out1.extend(engine1.drain())

    engine2 = RBTAEngine(base_delta_t=timedelta(minutes=15))
    out2 = []
    for a in alerts:
        out2.extend(engine2.process(a))
    out2.extend(engine2.drain())

    assert len(out1) == len(out2)
    for m1, m2 in zip(out1, out2):
        assert m1.meta_id == m2.meta_id
        assert m1.agent_id == m2.agent_id
        assert m1.rule_group_primary == m2.rule_group_primary
        assert m1.alert_count == m2.alert_count
        assert m1.wazuh_alert_ids == m2.wazuh_alert_ids
