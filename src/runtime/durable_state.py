"""Durable runtime state persistence and crash recovery module."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine, _ActiveBucket
from src.rbta.temporal_state import AgentTemporalState


class DurableStateManager:
    """Manages durable state serialization and crash-recovery for RBTAEngine and runtime."""

    def __init__(self, filepath: Union[str, Path] = "state/runtime_state.json") -> None:
        self.filepath: Path = Path(filepath).resolve()

    def save_state(
        self,
        engine: RBTAEngine,
        outbox: Optional[List[Dict[str, Any]]] = None,
        source_checkpoint: Optional[Dict[str, Any]] = None,
        finalized_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Atomically persist engine state, active buckets, seen alert IDs, and outbox to disk."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.filepath.with_suffix(".tmp")

        # 1. Serialize Seen IDs and Meta Counter
        seen_ids = list(engine._seen_alert_ids)
        meta_id_counter = engine._meta_id_counter

        # 2. Serialize Agent Temporal States
        temporal_states_data: Dict[str, Dict[str, Any]] = {}
        for agent_id, state in engine._temporal_states.items():
            temporal_states_data[agent_id] = {
                "agent_id": state.agent_id,
                "base_delta_t_sec": state.base_delta_t.total_seconds(),
                "adaptive": state.adaptive,
                "last_timestamp": state.last_timestamp.isoformat() if state.last_timestamp else None,
                "warmup_event_count": state.warmup_event_count,
                "warmup_gaps": list(state.warmup_gaps),
                "baseline_gap": state.baseline_gap,
                "ema_gap": state.ema_gap,
                "current_delta_t_sec": state.current_delta_t.total_seconds(),
                "is_warmed_up": state.is_warmed_up,
                "_is_terminal_invalid": state._is_terminal_invalid,
                "_invalid_reason": state._invalid_reason,
            }

        # 3. Serialize Active Buckets
        active_buckets_data: List[Dict[str, Any]] = []
        for (agent_id, rule_group), bucket in engine._active_buckets.items():
            active_buckets_data.append({
                "agent_id": agent_id,
                "rule_group_primary": rule_group,
                "meta_id": bucket.meta_id,
                "agent_name": bucket.agent_name,
                "start_time": bucket.start_time.isoformat(),
                "end_time": bucket.end_time.isoformat(),
                "alert_count": bucket.alert_count,
                "max_severity": bucket.max_severity,
                "rule_id_distribution": dict(bucket.rule_id_distribution),
                "severity_distribution": {str(k): v for k, v in bucket.severity_distribution.items()},
                "agent_criticality": bucket.agent_criticality,
                "wazuh_alert_ids": list(bucket.wazuh_alert_ids),
                "mitre_tactics_order": list(bucket.mitre_tactics_order),
                "critical_mitre_present": bucket.critical_mitre_present,
            })

        payload = {
            "schema_version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "meta_id_counter": meta_id_counter,
            "seen_alert_ids": seen_ids,
            "temporal_states": temporal_states_data,
            "active_buckets": active_buckets_data,
            "source_checkpoint": source_checkpoint or {},
            "outbox": outbox or [],
            "finalized_history": finalized_history or [],
        }

        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        tmp_file.replace(self.filepath)

    def restore_state(self, engine: RBTAEngine) -> Dict[str, Any]:
        """Restore internal engine structures from disk into the provided RBTAEngine instance.

        Parameters
        ----------
        engine : RBTAEngine
            Target engine instance to populate.

        Returns
        -------
        Dict[str, Any]
            Restored metadata dictionary containing 'outbox' and 'source_checkpoint'.
        """
        if not self.filepath.exists():
            return {"outbox": [], "source_checkpoint": {}, "finalized_history": []}

        with self.filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Restore Seen IDs and Counter
        engine._seen_alert_ids = set(data.get("seen_alert_ids", []))
        engine._meta_id_counter = int(data.get("meta_id_counter", 1))

        # 2. Restore Temporal States
        engine._temporal_states.clear()
        for agent_id, state_dict in data.get("temporal_states", {}).items():
            from datetime import timedelta
            state = AgentTemporalState(
                agent_id=agent_id,
                base_delta_t=timedelta(seconds=state_dict["base_delta_t_sec"]),
                adaptive=state_dict["adaptive"],
            )
            state.last_timestamp = datetime.fromisoformat(state_dict["last_timestamp"]) if state_dict["last_timestamp"] else None
            state.warmup_event_count = state_dict["warmup_event_count"]
            state.warmup_gaps = list(state_dict["warmup_gaps"])
            state.baseline_gap = state_dict["baseline_gap"]
            state.ema_gap = state_dict["ema_gap"]
            state.current_delta_t = timedelta(seconds=state_dict["current_delta_t_sec"])
            state.is_warmed_up = state_dict["is_warmed_up"]
            state._is_terminal_invalid = state_dict.get("_is_terminal_invalid", False)
            state._invalid_reason = state_dict.get("_invalid_reason")
            engine._temporal_states[agent_id] = state

        # 3. Restore Active Buckets
        engine._active_buckets.clear()
        for b_dict in data.get("active_buckets", []):
            bucket = _ActiveBucket.__new__(_ActiveBucket)
            bucket.meta_id = b_dict["meta_id"]
            bucket.agent_id = b_dict["agent_id"]
            bucket.agent_name = b_dict["agent_name"]
            bucket.rule_group_primary = b_dict["rule_group_primary"]
            bucket.start_time = datetime.fromisoformat(b_dict["start_time"])
            bucket.end_time = datetime.fromisoformat(b_dict["end_time"])
            bucket.alert_count = b_dict["alert_count"]
            bucket.max_severity = b_dict["max_severity"]
            bucket.rule_id_distribution = Counter(b_dict["rule_id_distribution"])
            bucket.severity_distribution = Counter({int(k): v for k, v in b_dict["severity_distribution"].items()})
            bucket.agent_criticality = b_dict["agent_criticality"]
            bucket.wazuh_alert_ids = list(b_dict["wazuh_alert_ids"])
            bucket.mitre_tactics_order = list(b_dict["mitre_tactics_order"])
            bucket._mitre_seen = {t.casefold() for t in bucket.mitre_tactics_order}
            bucket.critical_mitre_present = b_dict["critical_mitre_present"]

            key = (bucket.agent_id, bucket.rule_group_primary)
            engine._active_buckets[key] = bucket

        return {
            "source_checkpoint": data.get("source_checkpoint", {}),
            "outbox": data.get("outbox", []),
            "finalized_history": data.get("finalized_history", []),
        }
