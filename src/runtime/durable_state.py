"""Durable runtime state persistence and crash recovery module."""

from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine, _ActiveBucket
from src.rbta.temporal_state import AgentTemporalState


class DurableStateManager:
    """Manages durable state serialization and crash-recovery for RBTAEngine and runtime."""

    def __init__(self, filepath: Union[str, Path] = "state/runtime_state.json", finalized_history_db_path: Optional[Union[str, Path]] = None) -> None:
        self.filepath: Path = Path(filepath).resolve()
        self.history_db_path: Path = Path(finalized_history_db_path).resolve() if finalized_history_db_path else self.filepath.with_name("finalized_history.sqlite3")
        self._init_db()

    def _init_db(self):
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.history_db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS finalized_history (
                    meta_id INTEGER PRIMARY KEY,
                    scored_data TEXT NOT NULL,
                    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_alert_ids (
                    wazuh_alert_id TEXT PRIMARY KEY,
                    inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

    def append_finalized(self, scored_items: List[Dict[str, Any]]) -> None:
        if not scored_items:
            return
        
        batch = [
            (item["meta_id"], json.dumps(item))
            for item in scored_items
        ]
        
        with sqlite3.connect(self.history_db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO finalized_history (meta_id, scored_data)
                VALUES (?, ?)
                """,
                batch
            )

    def load_finalized_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        if not self.history_db_path.exists():
            return []
        with sqlite3.connect(self.history_db_path) as conn:
            cursor = conn.execute(
                "SELECT scored_data FROM (SELECT meta_id, scored_data FROM finalized_history ORDER BY meta_id DESC LIMIT ?) ORDER BY meta_id ASC",
                (max(1, int(limit)),),
            )
            rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_finalized(self, meta_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute(
                "SELECT scored_data FROM finalized_history WHERE meta_id = ?", (int(meta_id),)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def query_finalized(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        decision: Optional[str] = None,
        action: Optional[str] = None,
        agent_id: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "end_time",
        sort_order: str = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query the append-only history in SQLite without loading it into RAM."""
        sort_expressions = {
            "meta_id": "meta_id",
            "start_time": "json_extract(scored_data, '$.start_time')",
            "end_time": "json_extract(scored_data, '$.end_time')",
            "alert_count": "CAST(json_extract(scored_data, '$.alert_count') AS INTEGER)",
            "max_severity": "CAST(json_extract(scored_data, '$.max_severity') AS INTEGER)",
            "anomaly_score": "CAST(json_extract(scored_data, '$.anomaly_score') AS REAL)",
        }
        order_expr = sort_expressions.get(sort_by, sort_expressions["end_time"])
        direction = "ASC" if sort_order.lower() == "asc" else "DESC"
        clauses: List[str] = []
        params: List[Any] = []
        for field, value in (("decision", decision), ("action", action), ("agent_id", agent_id)):
            if value:
                clauses.append(f"json_extract(scored_data, '$.{field}') = ?")
                params.append(value)
        if search and search.strip():
            clauses.append("(CAST(meta_id AS TEXT) LIKE ? OR lower(json_extract(scored_data, '$.rule_group_primary')) LIKE ? OR lower(json_extract(scored_data, '$.agent_name')) LIKE ? OR lower(json_extract(scored_data, '$.agent_id')) LIKE ?)")
            needle = f"%{search.strip().lower()}%"
            params.extend([needle] * 4)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (max(1, page) - 1) * max(1, page_size)
        with sqlite3.connect(self.history_db_path) as conn:
            total_row = conn.execute(f"SELECT COUNT(*) FROM finalized_history{where}", params).fetchone()
            rows = conn.execute(
                f"SELECT scored_data FROM finalized_history{where} ORDER BY {order_expr} {direction} LIMIT ? OFFSET ?",
                [*params, max(1, page_size), offset],
            ).fetchall()
        return [json.loads(row[0]) for row in rows], int(total_row[0]) if total_row else 0

    def append_seen_alert_ids(self, alert_ids: Set[str]) -> None:
        """Persist only IDs committed since the previous checkpoint."""
        if not alert_ids:
            return
        with sqlite3.connect(self.history_db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO seen_alert_ids (wazuh_alert_id) VALUES (?)",
                ((alert_id,) for alert_id in alert_ids),
            )

    def load_seen_alert_ids(self) -> Set[str]:
        with sqlite3.connect(self.history_db_path) as conn:
            rows = conn.execute("SELECT wazuh_alert_id FROM seen_alert_ids").fetchall()
        return {str(row[0]) for row in rows}

    def count_seen_alert_ids(self) -> int:
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM seen_alert_ids").fetchone()
        return int(row[0]) if row else 0

    def has_seen_alert_id(self, alert_id: str) -> bool:
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM seen_alert_ids WHERE wazuh_alert_id = ? LIMIT 1", (alert_id,)
            ).fetchone()
        return row is not None

    @property
    def state_path(self) -> Path:
        """Return resolved path to the state file."""
        return self.filepath

    def save_state(
        self,
        engine: RBTAEngine,
        outbox: Optional[List[Dict[str, Any]]] = None,
        source_checkpoint: Optional[Dict[str, Any]] = None,
        new_finalized_history: Optional[List[Dict[str, Any]]] = None,
        pending_scoring: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Atomically persist engine state, active buckets, seen alert IDs, pending scoring, and outbox to disk."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if new_finalized_history:
            self.append_finalized(new_finalized_history)

        tmp_file = self.filepath.with_suffix(".tmp")

        # Seen IDs are append-only in SQLite so checkpoint cost is proportional
        # to new events rather than rewriting the full replay history as JSON.
        new_seen_ids = set(getattr(engine, "_new_seen_alert_ids", set()))
        self.append_seen_alert_ids(new_seen_ids)
        if hasattr(engine, "_new_seen_alert_ids"):
            engine._new_seen_alert_ids.difference_update(new_seen_ids)

        # 1. Serialize Meta Counter
        meta_id_counter = engine._meta_id_counter

        # 2. Serialize Agent Temporal States
        temporal_states_data: Dict[str, Dict[str, Any]] = {}
        for agent_id, state in engine._temporal_states.items():
            temporal_states_data[agent_id] = {
                "agent_id": state.agent_id,
                "agent_name": state.agent_name,
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
            "schema_version": "1.1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "meta_id_counter": meta_id_counter,
            "temporal_states": temporal_states_data,
            "active_buckets": active_buckets_data,
            "source_checkpoint": source_checkpoint or {},
            "pending_scoring": pending_scoring or [],
            "outbox": outbox or [],
        }

        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        tmp_file.replace(self.filepath)

    def restore_state(self, engine: RBTAEngine, hydrate_seen_ids: bool = True) -> Dict[str, Any]:
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
            engine._seen_alert_ids = self.load_seen_alert_ids() if hydrate_seen_ids else set()
            engine._new_seen_alert_ids = set()
            return {"outbox": [], "source_checkpoint": {}, "finalized_history": self.load_finalized_history()}

        with self.filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Restore Seen IDs and Counter
        legacy_seen_ids = set(data.get("seen_alert_ids", []))
        if legacy_seen_ids:
            self.append_seen_alert_ids(legacy_seen_ids)
        engine._seen_alert_ids = self.load_seen_alert_ids() if hydrate_seen_ids else set()
        engine._new_seen_alert_ids = set()
        engine._meta_id_counter = int(data.get("meta_id_counter", 1))

        # 2. Restore Temporal States
        engine._temporal_states.clear()
        for agent_id, state_dict in data.get("temporal_states", {}).items():
            from datetime import timedelta
            state = AgentTemporalState(
                agent_id=agent_id,
                agent_name=state_dict.get("agent_name", "unknown"),
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
            # Backward compatibility for state files created before agent_name
            # was persisted with temporal state metadata.
            state = engine._temporal_states.get(bucket.agent_id)
            if state and state.agent_name == "unknown" and bucket.agent_name:
                state.agent_name = bucket.agent_name

        legacy_history = data.get("finalized_history", [])
        if legacy_history:
            self.append_finalized(legacy_history)

        return {
            "source_checkpoint": data.get("source_checkpoint", {}),
            "pending_scoring": data.get("pending_scoring", []),
            "outbox": data.get("outbox", []),
            "finalized_history": self.load_finalized_history(),
        }
