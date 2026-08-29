"""Unit tests for ReplayClock speed factors and delays (Sprint 6)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from src.runners.clock import ClockError, ReplayClock


def test_replay_clock_speed_factors():
    """Clock computes wall delay as event gap divided by speed factor."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    next_t = base_t + timedelta(seconds=100)

    # 1x speed: 100s gap -> 100s delay
    c1 = ReplayClock(speed_factor=1.0)
    with patch("time.sleep") as mock_sleep:
        delay = c1.wait(base_t, next_t)
        assert delay == pytest.approx(100.0)
        mock_sleep.assert_called_once_with(pytest.approx(100.0))

    # 10x speed: 100s gap -> 10s delay
    c10 = ReplayClock(speed_factor=10.0)
    with patch("time.sleep") as mock_sleep:
        delay = c10.wait(base_t, next_t)
        assert delay == pytest.approx(10.0)
        mock_sleep.assert_called_once_with(pytest.approx(10.0))

    # 100x speed: 100s gap -> 1.0s delay
    c100 = ReplayClock(speed_factor=100.0)
    with patch("time.sleep") as mock_sleep:
        delay = c100.wait(base_t, next_t)
        assert delay == pytest.approx(1.0)
        mock_sleep.assert_called_once_with(pytest.approx(1.0))


def test_replay_clock_max_speed_does_not_sleep():
    """MAX speed does not sleep (delay=0.0) and never calls time.sleep."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    next_t = base_t + timedelta(hours=5)

    c_max = ReplayClock(speed_factor="MAX")
    with patch("time.sleep") as mock_sleep:
        delay = c_max.wait(base_t, next_t)
        assert delay == 0.0
        mock_sleep.assert_not_called()


def test_replay_clock_retrograde_or_same_timestamp_does_not_sleep():
    """If next timestamp is <= previous timestamp, delay is 0.0."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    prev_t = base_t - timedelta(seconds=50)

    c = ReplayClock(speed_factor=1.0)
    with patch("time.sleep") as mock_sleep:
        delay = c.wait(base_t, prev_t)
        assert delay == 0.0
        mock_sleep.assert_not_called()


def test_replay_clock_invalid_speed_factor():
    """Invalid speed factor raises ClockError."""
    with pytest.raises(ClockError, match="Invalid speed factor"):
        ReplayClock(speed_factor=0.0)

    with pytest.raises(ClockError, match="Invalid speed factor"):
        ReplayClock(speed_factor=-5.0)

    with pytest.raises(ClockError, match="Invalid speed factor"):
        ReplayClock(speed_factor="FAST")
