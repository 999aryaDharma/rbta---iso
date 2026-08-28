"""Lossless, bounded event-time reorder buffer implemented via min-heap."""

import heapq
from typing import List, Tuple
from src.contracts.raw_alert import CanonicalRawAlert


class LosslessReorderBuffer:
    """Bounded, deterministic event-time reorder buffer.

    Guarantees:
    - O(n log k) complexity for buffer capacity k.
    - Zero alert loss (lossless event conservation).
    - Monotonic arrival sequence tie-breaker preserving FIFO arrival order for identical timestamps.
    - Idempotent drain operation.

    Parameters
    ----------
    capacity : int
        Maximum number of alerts to buffer before emitting the earliest item (must be >= 1).
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError(f"Reorder buffer capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        # Heap elements: (timestamp, arrival_seq, CanonicalRawAlert)
        self._heap: List[Tuple[object, int, CanonicalRawAlert]] = []
        self._arrival_counter: int = 0

    @property
    def capacity(self) -> int:
        """Maximum capacity of the reorder buffer."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of buffered alerts."""
        return len(self._heap)

    def push(self, alert: CanonicalRawAlert) -> List[CanonicalRawAlert]:
        """Push a raw alert into the bounded reorder buffer.

        If capacity is exceeded after insertion, the earliest buffered alert is emitted.

        Parameters
        ----------
        alert : CanonicalRawAlert
            Incoming raw alert.

        Returns
        -------
        List[CanonicalRawAlert]
            List of 0 or 1 emitted alerts that became ready.
        """
        seq = self._arrival_counter
        self._arrival_counter += 1
        heapq.heappush(self._heap, (alert.timestamp, seq, alert))

        if len(self._heap) > self._capacity:
            _, _, earliest_alert = heapq.heappop(self._heap)
            return [earliest_alert]

        return []

    def drain(self) -> List[CanonicalRawAlert]:
        """Drain and return all remaining buffered alerts in ascending event-time order.

        Subsequent calls on an empty buffer return an empty list (idempotent).

        Returns
        -------
        List[CanonicalRawAlert]
            All remaining alerts in deterministic ascending event-time order.
        """
        emitted: List[CanonicalRawAlert] = []
        while self._heap:
            _, _, alert = heapq.heappop(self._heap)
            emitted.append(alert)
        return emitted
