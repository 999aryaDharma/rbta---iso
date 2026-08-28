"""Agent-local temporal state and EMA-based Elastic Time Window (ETW) calculation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from src.config.research import (
    DEFAULT_BASE_DELTA_T,
    EMA_ALPHA,
    ETW_MAX_MULTIPLIER,
    ETW_MIN_MULTIPLIER,
    WARMUP_EVENT_TARGET,
)


class TemporalStateError(ValueError):
    """Raised when temporal state encountered an invalid baseline or impossible time invariant."""
    pass


@dataclass
class AgentTemporalState:
    """Manages independent, agent-local temporal state and adaptive Elastic Time Window (ETW).

    Attributes
    ----------
    agent_id : str
        Unique Wazuh agent identifier.
    base_delta_t : timedelta
        Experiment baseline temporal window.
    adaptive : bool
        If True, applies adaptive ETW after 100-event warmup. If False, keeps fixed base_delta_t.
    last_timestamp : datetime | None
        Monotonic highest event-time timestamp observed for this agent.
    warmup_event_count : int
        Count of valid events processed during warmup phase (target = 100).
    warmup_gaps : list[float]
        Inter-arrival gaps (in seconds) collected during the first 100 events.
    baseline_gap : float | None
        Arithmetic mean of warm-up gaps (in seconds).
    ema_gap : float | None
        Exponential Moving Average of inter-arrival gaps (in seconds).
    current_delta_t : timedelta
        Current active time window for this agent.
    is_warmed_up : bool
        Flag indicating if the 100-event warmup has completed.
    """

    agent_id: str
    base_delta_t: timedelta = DEFAULT_BASE_DELTA_T
    adaptive: bool = True
    last_timestamp: datetime | None = None
    warmup_event_count: int = 0
    warmup_gaps: list[float] = field(default_factory=list)
    baseline_gap: float | None = None
    ema_gap: float | None = None
    current_delta_t: timedelta = field(init=False)
    is_warmed_up: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_delta_t, timedelta) or self.base_delta_t.total_seconds() <= 0:
            raise TemporalStateError(f"base_delta_t must be a positive timedelta, got {self.base_delta_t}")
        self.current_delta_t = self.base_delta_t

    def observe(self, timestamp: datetime) -> timedelta:
        """Observe an incoming event timestamp for this agent and return the active delta_t.

        Parameters
        ----------
        timestamp : datetime
            Timezone-aware event timestamp.

        Returns
        -------
        timedelta
            Active aggregation time window for this event.

        Raises
        ------
        TemporalStateError
            If timestamp is naive or if warmup baseline gap is <= 0.
        """
        if timestamp.tzinfo is None:
            raise TemporalStateError("Observed timestamp must be timezone-aware (tzinfo cannot be None)")

        # ── 1. First event initialization ─────────────────────────────────────
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.warmup_event_count = 1
            return self.current_delta_t

        # ── 2. Warmup Phase (Events 2 through 100) ───────────────────────────
        if not self.is_warmed_up:
            self.warmup_event_count += 1

            # Only forward/monotonic arrivals contribute valid non-negative gaps
            if timestamp >= self.last_timestamp:
                gap_sec = (timestamp - self.last_timestamp).total_seconds()
                self.warmup_gaps.append(gap_sec)
                self.last_timestamp = timestamp

            # Check if event 100 just completed warmup
            if self.warmup_event_count >= WARMUP_EVENT_TARGET:
                if not self.warmup_gaps or sum(self.warmup_gaps) <= 0:
                    raise TemporalStateError(
                        f"Agent '{self.agent_id}' baseline gap is <= 0 or undefined after "
                        f"{self.warmup_event_count} warmup events (sum={sum(self.warmup_gaps) if self.warmup_gaps else 0})"
                    )

                self.baseline_gap = sum(self.warmup_gaps) / len(self.warmup_gaps)
                self.ema_gap = self.baseline_gap
                self.is_warmed_up = True
                # Event 100 uses base_delta_t; adaptive update begins at event 101

            return self.current_delta_t

        # ── 3. Post-Warmup Adaptive Phase (Event 101+) ─────────────────────────
        if not self.adaptive:
            if timestamp > self.last_timestamp:
                self.last_timestamp = timestamp
            return self.current_delta_t

        # Handle forward arrival
        if timestamp >= self.last_timestamp:
            current_gap = (timestamp - self.last_timestamp).total_seconds()
            self.last_timestamp = timestamp

            # EMA formula: ema_gap = alpha * current_gap + (1 - alpha) * prev_ema
            assert self.ema_gap is not None
            assert self.baseline_gap is not None
            self.ema_gap = (EMA_ALPHA * current_gap) + ((1.0 - EMA_ALPHA) * self.ema_gap)

            # Proportional scaling: ratio = ema_gap / baseline_gap
            ratio = self.ema_gap / self.baseline_gap
            candidate_sec = self.base_delta_t.total_seconds() * ratio

            # Clamping to [0.5 * base, 1.5 * base]
            min_sec = ETW_MIN_MULTIPLIER * self.base_delta_t.total_seconds()
            max_sec = ETW_MAX_MULTIPLIER * self.base_delta_t.total_seconds()
            clamped_sec = min(max(candidate_sec, min_sec), max_sec)

            self.current_delta_t = timedelta(seconds=clamped_sec)

        # Note: If timestamp < last_timestamp (residual out-of-order event),
        # last_timestamp is NOT regressed and negative gap is NOT applied to EMA.

        return self.current_delta_t
