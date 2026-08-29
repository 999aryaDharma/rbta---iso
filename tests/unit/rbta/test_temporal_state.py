"""Unit tests for AgentTemporalState (Task 2.1)."""
from datetime import datetime, timedelta, timezone
import pytest

from src.rbta.temporal_state import AgentTemporalState, TemporalStateError


def test_first_event_initializes_state():
    """Event 1 must initialize last_timestamp, set warmup count to 1, and return base_delta_t."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    t1 = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    dt_out = state.observe(t1)

    assert dt_out == base_dt
    assert state.warmup_event_count == 1
    assert state.last_timestamp == t1
    assert len(state.warmup_gaps) == 0
    assert state.is_warmed_up is False
    assert state.baseline_gap is None
    assert state.ema_gap is None


def test_warmup_events_1_through_99():
    """Events 1 through 99 must accumulate forward gaps and keep returning base_delta_t."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(99):
        t = start + timedelta(seconds=i * 10)
        dt_out = state.observe(t)
        assert dt_out == base_dt
        assert state.warmup_event_count == i + 1
        assert state.is_warmed_up is False

    assert len(state.warmup_gaps) == 98
    assert all(g == 10.0 for g in state.warmup_gaps)


def test_event_100_completes_warmup_and_calculates_baseline():
    """Event 100 must complete warmup, compute arithmetic mean baseline, initialize ema_gap, and NOT apply post-warmup EMA."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    # Feed 100 events with 10 second gaps (first event has no gap, next 99 events each have gap=10.0s)
    for i in range(100):
        t = start + timedelta(seconds=i * 10)
        dt_out = state.observe(t)
        assert dt_out == base_dt

    assert state.warmup_event_count == 100
    assert state.is_warmed_up is True
    assert len(state.warmup_gaps) == 99
    assert state.baseline_gap == pytest.approx(10.0)
    assert state.ema_gap == pytest.approx(10.0)
    assert state.current_delta_t == base_dt


def test_event_101_applies_first_adaptive_update_and_manual_math_verification():
    """Event 101 must apply EMA alpha=0.10 to update ema_gap and compute proportional delta_t."""
    base_dt = timedelta(minutes=10)  # 600 seconds
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(100):
        t = start + timedelta(seconds=i * 10)
        state.observe(t)

    assert state.baseline_gap == pytest.approx(10.0)
    assert state.ema_gap == pytest.approx(10.0)

    # Event 101 arrives after 5 seconds
    t_101 = start + timedelta(seconds=99 * 10 + 5)
    dt_101 = state.observe(t_101)

    # Manual math check:
    # current_gap = 5.0s
    # ema_gap = 0.10 * 5.0 + 0.90 * 10.0 = 0.5 + 9.0 = 9.5s
    # ratio = 9.5 / 10.0 = 0.95
    # candidate_delta_t = 600s * 0.95 = 570s = 9.5 minutes
    assert state.ema_gap == pytest.approx(9.5)
    assert dt_101.total_seconds() == pytest.approx(570.0)
    assert state.current_delta_t == dt_101


def test_lower_and_upper_etw_clamps():
    """ETW delta_t must be strictly clamped to [0.5 * base_delta_t, 1.5 * base_delta_t]."""
    base_dt = timedelta(minutes=10)  # 600 seconds
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(100):
        state.observe(start + timedelta(seconds=i * 10))

    # Fast burst: gap = 0.001s repeatedly
    last_t = start + timedelta(seconds=990)
    for _ in range(50):
        last_t += timedelta(milliseconds=1)
        dt_out = state.observe(last_t)

    # Must be clamped at 0.5 * 600 = 300 seconds (5 minutes)
    assert dt_out.total_seconds() == pytest.approx(300.0)
    assert state.current_delta_t == timedelta(minutes=5)

    # Slow drought: gap = 1000s repeatedly
    for _ in range(50):
        last_t += timedelta(seconds=1000)
        dt_out = state.observe(last_t)

    # Must be clamped at 1.5 * 600 = 900 seconds (15 minutes)
    assert dt_out.total_seconds() == pytest.approx(900.0)
    assert state.current_delta_t == timedelta(minutes=15)


def test_zero_baseline_raises_temporal_state_error():
    """If all warmup gaps are 0 (e.g. 100 events with identical timestamp), baseline is 0 and must fail fast."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    same_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(TemporalStateError, match="baseline gap is <= 0"):
        for _ in range(100):
            state.observe(same_t)


def test_invalid_warmup_baseline_becomes_terminal_and_does_not_extend_beyond_100_events():
    """When warmup fails with invalid baseline, state becomes terminally invalid and rejects event 101+."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    same_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    # Events 1 through 100 with same timestamp
    with pytest.raises(TemporalStateError, match="baseline gap is <= 0"):
        for _ in range(100):
            state.observe(same_t)

    assert state.warmup_event_count == 100
    assert state.is_warmed_up is False
    assert state.baseline_gap is None

    # Event 101 arrives with positive timestamp -> must still fail fast and NOT extend warmup
    event_101_t = same_t + timedelta(seconds=10)
    with pytest.raises(TemporalStateError, match="terminal invalid state|baseline gap is <= 0"):
        state.observe(event_101_t)

    # Invariants preserved
    assert state.warmup_event_count == 100
    assert state.is_warmed_up is False
    assert state.baseline_gap is None


def test_retrograde_timestamp_handling():
    """Residual out-of-order event (timestamp < last_timestamp) must not regress last_timestamp, nor create negative EMA gap."""
    base_dt = timedelta(minutes=10)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(100):
        state.observe(start + timedelta(seconds=i * 10))

    last_t = start + timedelta(seconds=990)
    assert state.last_timestamp == last_t
    ema_before = state.ema_gap
    dt_before = state.current_delta_t

    # Retrograde event timestamp 100 seconds in the past
    retrograde_t = start + timedelta(seconds=500)
    dt_out = state.observe(retrograde_t)

    # Must not regress last_timestamp
    assert state.last_timestamp == last_t
    # Must not change ema_gap or current_delta_t
    assert state.ema_gap == ema_before
    assert dt_out == dt_before


def test_agent_isolation():
    """Events observed on Agent A must never mutate or influence Agent B's temporal state."""
    base_dt = timedelta(minutes=15)
    state_a = AgentTemporalState(agent_id="agent_A", base_delta_t=base_dt)
    state_b = AgentTemporalState(agent_id="agent_B", base_delta_t=base_dt)

    t1 = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 28, 10, 0, 1, tzinfo=timezone.utc)
    t3 = datetime(2026, 8, 28, 10, 5, 0, tzinfo=timezone.utc)

    state_a.observe(t1)
    state_b.observe(t2)
    state_a.observe(t3)

    # Agent A gap should be 300s, not affected by Agent B's t2
    assert len(state_a.warmup_gaps) == 1
    assert state_a.warmup_gaps[0] == 300.0
    assert state_a.warmup_event_count == 2
    assert state_a.last_timestamp == t3

    # Agent B state remains isolated
    assert len(state_b.warmup_gaps) == 0
    assert state_b.warmup_event_count == 1
    assert state_b.last_timestamp == t2


def test_fixed_mode_never_alters_delta_t():
    """When adaptive=False, current_delta_t must remain base_delta_t throughout the run."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt, adaptive=False)

    start = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(150):
        dt_out = state.observe(start + timedelta(seconds=i * 2))
        assert dt_out == base_dt
        assert state.current_delta_t == base_dt


def test_fixed_mode_does_not_require_positive_baseline_and_supports_same_timestamps():
    """Fixed mode (adaptive=False) must not compute or require an adaptive baseline, supporting 150 same-timestamp events."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt, adaptive=False)

    same_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    for _ in range(150):
        dt_out = state.observe(same_t)
        assert dt_out == base_dt

    assert state.current_delta_t == base_dt
    assert state.baseline_gap is None
    assert state.ema_gap is None
    assert state.warmup_event_count == 150


def test_fixed_mode_arbitrary_and_retrograde_gaps():
    """Fixed mode must handle tiny, large, and retrograde gaps while keeping current_delta_t == base_delta_t."""
    base_dt = timedelta(minutes=15)
    state = AgentTemporalState(agent_id="001", base_delta_t=base_dt, adaptive=False)

    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    # Forward gap 1s
    assert state.observe(base_t + timedelta(seconds=1)) == base_dt
    # Huge gap 10 days
    assert state.observe(base_t + timedelta(days=10)) == base_dt
    # Retrograde gap 5 days ago
    assert state.observe(base_t + timedelta(days=5)) == base_dt
    assert state.current_delta_t == base_dt
