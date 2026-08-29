"""Unit tests for LosslessReorderBuffer (Task 2.2)."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.reorder_buffer import LosslessReorderBuffer


def make_alert(alert_id: str, ts: datetime, agent_id: str = "001") -> CanonicalRawAlert:
    """Helper to create minimal valid CanonicalRawAlert."""
    return CanonicalRawAlert(
        wazuh_alert_id=alert_id,
        timestamp=ts,
        agent_id=agent_id,
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=3,
        rule_id="5501",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
    )


def test_reorder_buffer_invalid_capacity():
    """Buffer capacity must be at least 1."""
    with pytest.raises(ValueError, match="capacity"):
        LosslessReorderBuffer(capacity=0)

    with pytest.raises(ValueError, match="capacity"):
        LosslessReorderBuffer(capacity=-5)


def test_reorder_buffer_ordered_sequence():
    """Ordered input sequence is preserved and emitted lossless upon capacity overflow and drain."""
    buf = LosslessReorderBuffer(capacity=3)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t + timedelta(seconds=1))
    a2 = make_alert("2", base_t + timedelta(seconds=2))
    a3 = make_alert("3", base_t + timedelta(seconds=3))
    a4 = make_alert("4", base_t + timedelta(seconds=4))

    out = []
    out.extend(buf.push(a1))  # len 1 <= 3
    out.extend(buf.push(a2))  # len 2 <= 3
    out.extend(buf.push(a3))  # len 3 <= 3
    assert out == []

    # 4th push exceeds capacity 3 -> emits earliest (a1)
    out.extend(buf.push(a4))
    assert [a.wazuh_alert_id for a in out] == ["1"]

    # Drain emits remaining [a2, a3, a4]
    out.extend(buf.drain())
    assert [a.wazuh_alert_id for a in out] == ["1", "2", "3", "4"]


def test_reorder_buffer_disordered_sequence():
    """Disordered input sequence [t3, t1, t4, t2] is emitted in correct ascending event-time order."""
    buf = LosslessReorderBuffer(capacity=4)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    a1 = make_alert("1", base_t + timedelta(seconds=10))
    a2 = make_alert("2", base_t + timedelta(seconds=20))
    a3 = make_alert("3", base_t + timedelta(seconds=30))
    a4 = make_alert("4", base_t + timedelta(seconds=40))

    # Push in disordered order: a3, a1, a4, a2
    out = []
    out.extend(buf.push(a3))
    out.extend(buf.push(a1))
    out.extend(buf.push(a4))
    out.extend(buf.push(a2))
    assert out == []

    out.extend(buf.drain())
    assert [a.wazuh_alert_id for a in out] == ["1", "2", "3", "4"]


def test_reorder_buffer_identical_timestamps_preserves_arrival_order():
    """Alerts with identical timestamps preserve deterministic arrival (FIFO) tie order."""
    buf = LosslessReorderBuffer(capacity=3)
    same_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    aA = make_alert("A", same_t)
    aB = make_alert("B", same_t)
    aC = make_alert("C", same_t)

    buf.push(aA)
    buf.push(aB)
    buf.push(aC)

    out = buf.drain()
    assert [a.wazuh_alert_id for a in out] == ["A", "B", "C"]


def test_reorder_buffer_drain_is_idempotent():
    """First drain returns all remaining alerts; subsequent drains return empty lists."""
    buf = LosslessReorderBuffer(capacity=2)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    buf.push(make_alert("1", base_t))
    buf.push(make_alert("2", base_t + timedelta(seconds=1)))

    first_drain = buf.drain()
    assert len(first_drain) == 2

    second_drain = buf.drain()
    assert second_drain == []

    third_drain = buf.drain()
    assert third_drain == []


def test_reorder_buffer_conservation():
    """Every pushed alert must be emitted exactly once (Counter equality, zero loss)."""
    buf = LosslessReorderBuffer(capacity=5)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # 100 alerts with various timestamps and disordered arrivals
    import random
    rng = random.Random(42)
    timestamps = [base_t + timedelta(seconds=rng.randint(0, 500)) for _ in range(100)]
    alerts = [make_alert(f"id_{i}", ts) for i, ts in enumerate(timestamps)]

    emitted = []
    for a in alerts:
        emitted.extend(buf.push(a))
    emitted.extend(buf.drain())

    input_ids = [a.wazuh_alert_id for a in alerts]
    emitted_ids = [a.wazuh_alert_id for a in emitted]

    assert len(emitted_ids) == 100
    assert Counter(input_ids) == Counter(emitted_ids)


def test_reorder_buffer_late_residual_event_never_dropped():
    """An extremely late event pushed into the buffer is never dropped and eventually emitted."""
    buf = LosslessReorderBuffer(capacity=2)
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # Push normal future alerts
    a_future1 = make_alert("F1", base_t + timedelta(hours=5))
    a_future2 = make_alert("F2", base_t + timedelta(hours=6))
    # Push late event far in the past
    a_late = make_alert("LATE", base_t)

    out = []
    out.extend(buf.push(a_future1))
    out.extend(buf.push(a_future2))
    # Pushing late event exceeds capacity 2 -> emits earliest (LATE)
    out.extend(buf.push(a_late))

    assert len(out) == 1
    assert out[0].wazuh_alert_id == "LATE"

    out.extend(buf.drain())
    assert [a.wazuh_alert_id for a in out] == ["LATE", "F1", "F2"]
