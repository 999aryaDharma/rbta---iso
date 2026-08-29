"""Replay clock managing accelerated event-time wall-clock pacing."""

from datetime import datetime
import time
from typing import Union


class ClockError(ValueError):
    """Raised when replay clock parameters are invalid."""
    pass


class ReplayClock:
    """Manages wall-clock pacing for historical event replay.

    Parameters
    ----------
    speed_factor : float | str
        Replay speed multiplier (1.0, 10.0, 100.0, or "MAX").
    """

    def __init__(self, speed_factor: Union[float, int, str] = "MAX") -> None:
        if isinstance(speed_factor, str):
            if speed_factor.upper() in ("MAX", "INF", "INFINITY"):
                self.speed_factor: float = float("inf")
            else:
                try:
                    val = float(speed_factor)
                    if val <= 0.0:
                        raise ValueError()
                    self.speed_factor = val
                except ValueError:
                    raise ClockError(f"Invalid speed factor string: '{speed_factor}'. Allowed: 1, 10, 100, 'MAX'")
        elif isinstance(speed_factor, (int, float)):
            if speed_factor <= 0:
                raise ClockError(f"Invalid speed factor: {speed_factor}. Must be > 0.")
            self.speed_factor = float(speed_factor)
        else:
            raise ClockError(f"Unsupported speed factor type: {type(speed_factor)}")

    def wait(self, previous_event_time: datetime, current_event_time: datetime) -> float:
        """Calculate and wait for the appropriate wall-clock delay between two events.

        Parameters
        ----------
        previous_event_time : datetime
            Timestamp of the previously processed event.
        current_event_time : datetime
            Timestamp of the current incoming event.

        Returns
        -------
        float
            Wall-clock delay waited in seconds.
        """
        if self.speed_factor == float("inf"):
            return 0.0

        if previous_event_time is None or current_event_time is None:
            return 0.0

        gap_sec = (current_event_time - previous_event_time).total_seconds()
        if gap_sec <= 0.0:
            return 0.0

        delay = max(0.0, gap_sec / self.speed_factor)
        if delay > 0.0:
            time.sleep(delay)

        return delay
