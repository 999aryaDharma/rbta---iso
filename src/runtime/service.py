"""Stateful live RBTA runtime service coordinating aggregation, scoring, and outbox."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

from src.config.research import DEFAULT_BASE_DELTA_T
from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.scoring_pipeline import ScoringPipeline
from src.rbta.engine import RBTAEngine
from src.runtime.durable_state import DurableStateManager

logger = logging.getLogger(__name__)


def _serialize_scored_alert(scored: ScoredMetaAlert) -> Dict[str, Any]:
    """Convert ScoredMetaAlert to JSON-serializable dictionary."""
    return {
        "meta_id": scored.meta_id,
        "agent_id": scored.agent_id,
        "agent_name": scored.agent_name,
        "rule_group_primary": scored.rule_group_primary,
        "start_time": scored.start_time.isoformat(),
        "end_time": scored.end_time.isoformat(),
        "alert_count": scored.alert_count,
        "max_severity": scored.max_severity,
        "mitre_tactics": list(scored.mitre_tactics),
        "seven_features": dict(scored.seven_features),
        "raw_model_score": scored.raw_model_score,
        "anomaly_score": scored.anomaly_score,
        "threshold_used": scored.threshold_used,
        "decision": scored.decision,
        "action": scored.action,
        "escalate": scored.escalate,
        "model_version": scored.model_version,
        "feature_schema_version": scored.feature_schema_version,
        "score_calibration_version": scored.score_calibration_version,
        "source_alert_ids": list(scored.source_alert_ids),
        "metadata": dict(scored.metadata),
    }


class LiveRBTAService:
    """Live stateful operational service managing RBTA, online scoring, and outbox queueing.

    Parameters
    ----------
    scoring_pipeline : ScoringPipeline
        Pre-loaded model scoring pipeline.
    state_manager : DurableStateManager | None
        Durable state manager for crash recovery.
    base_delta_t : timedelta
        Experiment aggregation time window.
    adaptive : bool
        Whether to enable per-agent EMA adaptation after 100-event warmup.
    """

    def __init__(
        self,
        scoring_pipeline: ScoringPipeline,
        state_manager: Optional[DurableStateManager] = None,
        base_delta_t: timedelta = DEFAULT_BASE_DELTA_T,
        adaptive: bool = True,
    ) -> None:
        self.scoring_pipeline: ScoringPipeline = scoring_pipeline
        self.state_manager: DurableStateManager = state_manager or DurableStateManager()
        self.base_delta_t: timedelta = base_delta_t
        self.adaptive: bool = adaptive

        self.engine: RBTAEngine = RBTAEngine(base_delta_t=self.base_delta_t, adaptive=self.adaptive)
        self.outbox: List[ScoredMetaAlert] = []
        self.source_checkpoint: Dict[str, Any] = {}

        # Restore from durable state on startup
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        """Attempt to restore engine state and outbox from disk."""
        restored = self.state_manager.restore_state(self.engine)
        self.source_checkpoint = restored.get("source_checkpoint", {})
        raw_outbox = restored.get("outbox", [])

        for item in raw_outbox:
            if isinstance(item, ScoredMetaAlert):
                self.outbox.append(item)
            elif isinstance(item, dict) and "meta_id" in item and "anomaly_score" in item:
                try:
                    meta = ScoredMetaAlert(
                        meta_id=item["meta_id"],
                        agent_id=item.get("agent_id", "001"),
                        agent_name=item.get("agent_name", "unknown"),
                        rule_group_primary=item.get("rule_group_primary", "unknown"),
                        start_time=datetime.fromisoformat(item["start_time"]) if "start_time" in item else datetime.now(timezone.utc),
                        end_time=datetime.fromisoformat(item["end_time"]) if "end_time" in item else datetime.now(timezone.utc),
                        alert_count=item.get("alert_count", 1),
                        max_severity=item.get("max_severity", 3),
                        mitre_tactics=tuple(item.get("mitre_tactics", ())),
                        seven_features=item.get("seven_features", {}),
                        raw_model_score=item.get("raw_model_score", 0.0),
                        anomaly_score=item.get("anomaly_score", 0.0),
                        threshold_used=item.get("threshold_used", 0.0),
                        decision=item.get("decision", "NOISE"),
                        action=item.get("action", "SUPPRESS"),
                        escalate=item.get("escalate", False),
                        model_version=item.get("model_version", "v1"),
                        feature_schema_version=item.get("feature_schema_version", "1.0"),
                        score_calibration_version=item.get("score_calibration_version", "minmax-v1"),
                        source_alert_ids=tuple(item.get("source_alert_ids", ())),
                        metadata=item.get("metadata", {}),
                    )
                    self.outbox.append(meta)
                except Exception as exc:
                    logger.warning("Error restoring outbox item: %s", exc)

    def _persist_to_disk(self) -> None:
        """Persist current state and outbox to disk."""
        outbox_payload = [_serialize_scored_alert(item) for item in self.outbox]
        self.state_manager.save_state(
            engine=self.engine,
            outbox=outbox_payload,
            source_checkpoint=self.source_checkpoint,
        )

    def ingest_alert(self, alert: CanonicalRawAlert) -> List[ScoredMetaAlert]:
        """Process an incoming canonical raw alert through RBTA and scoring.

        Parameters
        ----------
        alert : CanonicalRawAlert
            Incoming normalized alert.

        Returns
        -------
        List[ScoredMetaAlert]
            Any newly finalized and scored meta-alerts produced during this step.
        """
        finalized_metas = self.engine.process(alert)
        new_scored: List[ScoredMetaAlert] = []

        for meta in finalized_metas:
            scored = self.scoring_pipeline.score_single(meta)
            self.outbox.append(scored)
            new_scored.append(scored)

        self._persist_to_disk()
        return new_scored

    def check_idle_flush(self, current_event_time: datetime) -> List[ScoredMetaAlert]:
        """Flush and score active buckets whose idle duration strictly exceeds delta_t.

        Parameters
        ----------
        current_event_time : datetime
            Reference event timestamp.

        Returns
        -------
        List[ScoredMetaAlert]
            Any newly finalized and scored meta-alerts.
        """
        finalized_metas = self.engine.flush_idle(current_event_time)
        new_scored: List[ScoredMetaAlert] = []

        for meta in finalized_metas:
            scored = self.scoring_pipeline.score_single(meta)
            self.outbox.append(scored)
            new_scored.append(scored)

        if new_scored:
            self._persist_to_disk()

        return new_scored

    def get_outbox(self) -> List[ScoredMetaAlert]:
        """Retrieve unacknowledged scored meta-alerts in the outbox."""
        return list(self.outbox)

    def acknowledge_outbox(self, meta_id: int) -> None:
        """Acknowledge and remove a scored meta-alert from the outbox."""
        self.outbox = [item for item in self.outbox if item.meta_id != meta_id]
        self._persist_to_disk()

    def shutdown(self, drain: bool = False) -> List[ScoredMetaAlert]:
        """Perform a controlled shutdown, optionally draining active buckets.

        Parameters
        ----------
        drain : bool
            Whether to finalize and flush all active buckets on shutdown.

        Returns
        -------
        List[ScoredMetaAlert]
            Any finalized meta-alerts flushed during drain.
        """
        drained_scored: List[ScoredMetaAlert] = []
        if drain:
            drained_metas = self.engine.drain()
            for meta in drained_metas:
                scored = self.scoring_pipeline.score_single(meta)
                self.outbox.append(scored)
                drained_scored.append(scored)

        self._persist_to_disk()
        return drained_scored
