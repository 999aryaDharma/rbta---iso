"""Stateful live RBTA runtime service coordinating aggregation, scoring, and outbox."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

from src.config.research import DEFAULT_BASE_DELTA_T
from src.contracts.meta_alert import MetaAlert
from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.scoring_pipeline import ScoringPipeline
from src.rbta.engine import RBTAEngine
from src.runtime.durable_state import DurableStateManager

logger = logging.getLogger(__name__)


def _serialize_meta_alert(meta: MetaAlert) -> Dict[str, Any]:
    """Convert MetaAlert to JSON-serializable dictionary."""
    return {
        "meta_id": meta.meta_id,
        "agent_id": meta.agent_id,
        "agent_name": meta.agent_name,
        "rule_group_primary": meta.rule_group_primary,
        "start_time": meta.start_time.isoformat() if meta.start_time else None,
        "end_time": meta.end_time.isoformat() if meta.end_time else None,
        "alert_count": meta.alert_count,
        "max_severity": meta.max_severity,
        "rule_id_distribution": dict(meta.rule_id_distribution),
        "severity_distribution": {str(k): v for k, v in meta.severity_distribution.items()},
        "agent_criticality": meta.agent_criticality,
        "wazuh_alert_ids": list(meta.wazuh_alert_ids),
        "mitre_tactics_unique": list(meta.mitre_tactics_unique),
        "critical_mitre_present": meta.critical_mitre_present,
        "metadata": dict(meta.metadata),
    }


def _parse_meta_alert(item: Any) -> Optional[MetaAlert]:
    """Parse dictionary or MetaAlert object safely."""
    if isinstance(item, MetaAlert):
        return item
    if isinstance(item, dict) and "meta_id" in item:
        try:
            return MetaAlert(
                meta_id=item["meta_id"],
                agent_id=item.get("agent_id", "001"),
                agent_name=item.get("agent_name", "unknown"),
                rule_group_primary=item.get("rule_group_primary", "unknown"),
                start_time=datetime.fromisoformat(item["start_time"]) if item.get("start_time") else None,
                end_time=datetime.fromisoformat(item["end_time"]) if item.get("end_time") else None,
                alert_count=item.get("alert_count", 1),
                max_severity=item.get("max_severity", 3),
                rule_id_distribution=item.get("rule_id_distribution", {}),
                severity_distribution={int(k): v for k, v in item.get("severity_distribution", {}).items()},
                agent_criticality=item.get("agent_criticality", 1),
                wazuh_alert_ids=tuple(item.get("wazuh_alert_ids", ())),
                mitre_tactics_unique=tuple(item.get("mitre_tactics_unique", ())),
                critical_mitre_present=item.get("critical_mitre_present", False),
                metadata=item.get("metadata", {}),
            )
        except Exception as exc:
            logger.warning("Error parsing MetaAlert: %s", exc)
    return None


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
        raw_evidence_store: Optional[Any] = None,
        source_mode: str = 'LIVE',
    ) -> None:
        self.scoring_pipeline: ScoringPipeline = scoring_pipeline
        self.state_manager: DurableStateManager = state_manager or DurableStateManager()
        self.base_delta_t: timedelta = base_delta_t
        self.adaptive: bool = adaptive
        self.raw_evidence_store = raw_evidence_store
        self.source_mode = source_mode

        self.engine: RBTAEngine = RBTAEngine(base_delta_t=self.base_delta_t, adaptive=self.adaptive)
        self.pending_scoring: List[MetaAlert] = []
        self.outbox: List[ScoredMetaAlert] = []
        self.finalized_history: List[ScoredMetaAlert] = []
        self.source_checkpoint: Dict[str, Any] = {}

        # Restore from durable state on startup
        self._restore_from_disk()

    def _parse_scored_alert(self, item: Any) -> Optional[ScoredMetaAlert]:
        if isinstance(item, ScoredMetaAlert):
            return item
        if isinstance(item, dict) and "meta_id" in item and "anomaly_score" in item:
            try:
                return ScoredMetaAlert(
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
            except Exception as exc:
                logger.warning("Error parsing scored alert: %s", exc)
        return None

    def _restore_from_disk(self) -> None:
        """Attempt to restore engine state, pending scoring, and outbox from disk."""
        restored = self.state_manager.restore_state(self.engine)
        self.source_checkpoint = restored.get("source_checkpoint", {})
        raw_pending = restored.get("pending_scoring", [])
        raw_outbox = restored.get("outbox", [])
        raw_history = restored.get("finalized_history", [])

        for item in raw_pending:
            meta = _parse_meta_alert(item)
            if meta:
                self.pending_scoring.append(meta)

        for item in raw_outbox:
            meta = self._parse_scored_alert(item)
            if meta:
                self.outbox.append(meta)

        for item in raw_history:
            meta = self._parse_scored_alert(item)
            if meta:
                self.finalized_history.append(meta)

        if self.scoring_pipeline and self.pending_scoring:
            self._drain_pending_scoring()

    def _persist_to_disk(self) -> None:
        """Persist current state, pending scoring, and outbox to disk."""
        pending_payload = [_serialize_meta_alert(item) for item in self.pending_scoring]
        outbox_payload = [_serialize_scored_alert(item) for item in self.outbox]
        history_payload = [_serialize_scored_alert(item) for item in self.finalized_history]
        self.state_manager.save_state(
            engine=self.engine,
            outbox=outbox_payload,
            source_checkpoint=self.source_checkpoint,
            finalized_history=history_payload,
            pending_scoring=pending_payload,
        )

    def _drain_pending_scoring(self) -> List[ScoredMetaAlert]:
        """Score pending meta-alerts and safely commit them to outbox and history."""
        new_scored: List[ScoredMetaAlert] = []
        while self.pending_scoring:
            meta = self.pending_scoring[0]
            scored = self.scoring_pipeline.score_single(meta)
            self.outbox.append(scored)
            self.finalized_history.append(scored)
            new_scored.append(scored)
            self.pending_scoring.pop(0)
            self._persist_to_disk()
        return new_scored

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
        if self.raw_evidence_store is not None:
            # We don't have original payload here directly, pass None
            self.raw_evidence_store.store(alert, source_mode=self.source_mode)

        finalized_metas = self.engine.process(alert)
        if finalized_metas:
            self.pending_scoring.extend(finalized_metas)

        self._persist_to_disk()
        return self._drain_pending_scoring()

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
        if finalized_metas:
            self.pending_scoring.extend(finalized_metas)
            self._persist_to_disk()

        return self._drain_pending_scoring()

    def get_outbox(self) -> List[ScoredMetaAlert]:
        """Retrieve unacknowledged scored meta-alerts in the outbox."""
        return list(self.outbox)

    def acknowledge_outbox(self, meta_id: int) -> None:
        """Acknowledge and remove a scored meta-alert from the outbox."""
        self.outbox = [item for item in self.outbox if item.meta_id != meta_id]
        self._persist_to_disk()

    def commit_outbox(self, meta_ids: List[int]) -> int:
        """Acknowledge and remove multiple scored meta-alerts from the outbox."""
        initial = len(self.outbox)
        self.outbox = [item for item in self.outbox if item.meta_id not in meta_ids]
        self._persist_to_disk()
        return initial - len(self.outbox)

    def get_history(self) -> List[ScoredMetaAlert]:
        return list(self.finalized_history)

    def get_meta_detail(self, meta_id: int) -> Optional[ScoredMetaAlert]:
        for item in self.finalized_history:
            if item.meta_id == meta_id:
                return item
        return None

    def is_seen(self, wazuh_alert_id: str) -> bool:
        """Check if an alert ID has already been committed in RBTAEngine."""
        return self.engine.has_seen_alert(wazuh_alert_id) if hasattr(self.engine, "has_seen_alert") else (wazuh_alert_id in self.engine._seen_alert_ids)

    def drain_and_score(self) -> List[ScoredMetaAlert]:
        """Drain all currently active engine buckets, score them, and persist."""
        drained_metas = self.engine.drain()
        if drained_metas:
            self.pending_scoring.extend(drained_metas)
            self._persist_to_disk()
            return self._drain_pending_scoring()
        return []

    def get_live_source_state(self) -> Dict[str, Any]:
        """Retrieve copy of durable live source transport state."""
        return dict(self.source_checkpoint)

    def update_live_source_state(self, state: Dict[str, Any]) -> None:
        """Update and atomically persist live source transport state."""
        self.source_checkpoint.update(state)
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
        if drain:
            drained_metas = self.engine.drain()
            if drained_metas:
                self.pending_scoring.extend(drained_metas)

        self._persist_to_disk()
        return self._drain_pending_scoring()
