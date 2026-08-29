"""Replay stream runner executing event-by-event inference over the shared Research Core."""

from datetime import datetime, timedelta
from typing import Iterable, Iterator, List, Optional

from src.config.research import DEFAULT_BASE_DELTA_T
from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.scoring_pipeline import ScoringPipeline
from src.rbta.engine import RBTAEngine
from src.rbta.reorder_buffer import LosslessReorderBuffer
from src.runners.clock import ReplayClock


class ReplayStreamRunner:
    """Replays raw alert streams with event-time wall-clock pacing and online inference.

    Parameters
    ----------
    scoring_pipeline : ScoringPipeline
        Pre-loaded model scoring pipeline (inference only).
    clock : ReplayClock | None
        Pacing clock controlling replay speed (default: MAX / non-blocking).
    base_delta_t : timedelta
        Experiment aggregation time window.
    adaptive : bool
        Whether to enable per-agent EMA adaptation after 100-event warmup.
    reorder_capacity : int
        Bounded reorder buffer capacity.
    """

    def __init__(
        self,
        scoring_pipeline: ScoringPipeline,
        clock: Optional[ReplayClock] = None,
        base_delta_t: timedelta = DEFAULT_BASE_DELTA_T,
        adaptive: bool = True,
        reorder_capacity: int = 50,
    ) -> None:
        self.scoring_pipeline: ScoringPipeline = scoring_pipeline
        self.clock: ReplayClock = clock or ReplayClock(speed_factor="MAX")
        self.base_delta_t: timedelta = base_delta_t
        self.adaptive: bool = adaptive
        self.reorder_capacity: int = reorder_capacity

    def run(self, alerts: Iterable[CanonicalRawAlert]) -> Iterator[ScoredMetaAlert]:
        """Stream raw canonical alerts, advance replay clock, and yield ScoredMetaAlerts on the fly.

        Parameters
        ----------
        alerts : Iterable[CanonicalRawAlert]
            Stream of raw canonical alerts.

        Yields
        ------
        ScoredMetaAlert
            Scored, prioritized meta-alerts as they are finalized by RBTA.
        """
        buffer = LosslessReorderBuffer(capacity=self.reorder_capacity)
        engine = RBTAEngine(base_delta_t=self.base_delta_t, adaptive=self.adaptive)
        prev_ts: Optional[datetime] = None

        for alert in alerts:
            # Advance replay clock
            self.clock.wait(prev_ts, alert.timestamp)
            prev_ts = alert.timestamp

            ready_alerts = buffer.push(alert)
            for ready in ready_alerts:
                finalized = engine.process(ready)
                for meta in finalized:
                    yield self.scoring_pipeline.score_single(meta)

        # Drain reorder buffer
        for ready in buffer.drain():
            finalized = engine.process(ready)
            for meta in finalized:
                yield self.scoring_pipeline.score_single(meta)

        # Drain remaining active buckets
        for meta in engine.drain():
            yield self.scoring_pipeline.score_single(meta)
