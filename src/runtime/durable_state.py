"""Durable runtime state persistence and crash recovery module."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine, _ActiveBucket
from src.rbta.temporal_state import AgentTemporalState

logger = logging.getLogger(__name__)

#: Version of the Wazuh canonicalizer derivation logic covered by
#: :func:`compute_derivation_hash`. Bump when canonicalizer semantics change
#: so drift detection fails closed instead of silently reusing stale state.
CANONICALIZER_VERSION = "1.0"


def compute_derivation_hash() -> str:
    """Compute a stable hash of the canonicalizer derivation config.

    The canonicalizer derives ``agent_criticality`` and ``rule_group_primary``
    from the domain tables in :mod:`src.config.domain`; any change to those
    tables silently changes fingerprints, buckets, and features.  Hashing their
    JSON-sorted representation (plus :data:`CANONICALIZER_VERSION`) lets the
    live worker detect config drift against the hash stored alongside durable
    state (fail-visible, never silent).

    Fail-closed like model pinning: when the config cannot be read the error
    propagates — there is deliberately no constant fallback, so a broken
    config can never masquerade as a known-good derivation.
    """
    from src.config.domain import (
        AGENT_CRITICALITY,
        CRITICAL_MITRE_TACTICS,
        DEFAULT_AGENT_CRITICALITY,
        DEFAULT_RULE_GROUP_WEIGHT,
        GROUP_SEVERITY_WEIGHT,
    )

    derivation = {
        "agent_criticality": dict(sorted(AGENT_CRITICALITY.items())),
        "group_severity_weight": dict(sorted(GROUP_SEVERITY_WEIGHT.items())),
        "critical_mitre_tactics": sorted(str(t) for t in CRITICAL_MITRE_TACTICS),
        "default_agent_criticality": DEFAULT_AGENT_CRITICALITY,
        "default_rule_group_weight": DEFAULT_RULE_GROUP_WEIGHT,
        "canonicalizer_version": CANONICALIZER_VERSION,
    }
    canonical = json.dumps(derivation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
            conn.execute("CREATE TABLE IF NOT EXISTS runtime_snapshot (id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quarantined_alerts (
                    wazuh_alert_id TEXT PRIMARY KEY,
                    source_index TEXT NOT NULL DEFAULT '',
                    source_document_id TEXT NOT NULL DEFAULT '',
                    error_type TEXT NOT NULL DEFAULT '',
                    count INTEGER NOT NULL DEFAULT 1,
                    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
                    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute("CREATE TABLE IF NOT EXISTS derivation_state (id INTEGER PRIMARY KEY CHECK (id = 1), derivation_hash TEXT NOT NULL)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS suppressed_notifications (
                    meta_id INTEGER PRIMARY KEY,
                    reason TEXT NOT NULL DEFAULT '',
                    suppressed_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meta_id INTEGER,
                    run_id TEXT,
                    verdict TEXT,
                    at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

    def _assert_no_finalized_collision(self, new_finalized_history: List[Dict[str, Any]]) -> None:
        """Fail closed when a batch meta_id already exists in finalized_history.

        N8: silently ignoring (INSERT OR IGNORE) would fork the meta_id stream —
        the batch row is dropped while the in-memory counter has already moved
        on. Checked with a read-only SELECT (chunked at 500) before any write,
        so commit-only-on-success is preserved.

        Idempotency: a colliding row whose stored scored_data is identical to
        the batch item is a retried checkpoint, not a fork — it passes without
        raising. Any divergent content under a committed meta_id still raises.
        """
        ids = [int(item["meta_id"]) for item in (new_finalized_history or [])]
        if not ids:
            return
        batch_by_id: Dict[int, Dict[str, Any]] = {}
        for item in (new_finalized_history or []):
            batch_by_id.setdefault(int(item["meta_id"]), item)
        with sqlite3.connect(self.history_db_path) as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT meta_id, scored_data FROM finalized_history WHERE meta_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                if not rows:
                    continue
                divergent = sorted(
                    int(r[0]) for r in rows
                    if json.loads(r[1]) != batch_by_id.get(int(r[0]))
                )
                if divergent:
                    raise RuntimeError(
                        f"Refusing to checkpoint: batch meta_id(s) already committed "
                        f"in finalized_history with divergent content: {divergent}"
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

    def quarantine_add(
        self,
        wazuh_alert_id: str,
        source_index: str = "",
        source_document_id: str = "",
        error_type: str = "",
    ) -> int:
        """Record a deterministically corrupt alert in durable quarantine (upsert).

        Repeat offenses against the same ID bump ``count``/``last_seen`` while
        ``first_seen`` is preserved, so the quarantine table stays visible in
        status without growing per retry.

        Returns the per-ID ``count`` after this offense (1 on first offense),
        so callers can log the first quarantine as an error and repeats as debug.
        """
        counts = self.quarantine_add_many([{
            "wazuh_alert_id": wazuh_alert_id,
            "source_index": source_index,
            "source_document_id": source_document_id,
            "error_type": error_type,
        }])
        return counts[0]

    def quarantine_add_many(self, entries: List[Dict[str, Any]]) -> List[int]:
        """Record a batch of quarantine entries in a single transaction.

        One connection / one commit for the whole batch (plus one read-back
        for the resulting counts). Returns the per-ID ``count`` after this
        flush, in entry order.
        """
        if not entries:
            return []
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(e.get("wazuh_alert_id", "")),
                str(e.get("source_index", "") or ""),
                str(e.get("source_document_id", "") or ""),
                str(e.get("error_type", "") or ""),
                now,
                now,
            )
            for e in entries
        ]
        with sqlite3.connect(self.history_db_path) as conn:
            conn.executemany(
                """
                INSERT INTO quarantined_alerts
                    (wazuh_alert_id, source_index, source_document_id, error_type, count, first_seen, last_seen)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(wazuh_alert_id) DO UPDATE SET
                    source_index = excluded.source_index,
                    source_document_id = excluded.source_document_id,
                    error_type = excluded.error_type,
                    count = quarantined_alerts.count + 1,
                    last_seen = excluded.last_seen
                """,
                rows,
            )
            conn.commit()
            counts = []
            for (alert_id, *_rest) in rows:
                row = conn.execute(
                    "SELECT count FROM quarantined_alerts WHERE wazuh_alert_id = ?",
                    (alert_id,),
                ).fetchone()
                counts.append(int(row[0]) if row else 0)
        return counts

    def quarantine_release(self, wazuh_alert_id: str) -> int:
        """Delete one quarantined ID; returns affected rows (1 released, 0 absent)."""
        with sqlite3.connect(self.history_db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM quarantined_alerts WHERE wazuh_alert_id = ?",
                (str(wazuh_alert_id),),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def quarantine_count(self) -> int:
        """Return COUNT(*) of quarantined alerts (agent status uses this, not len(list))."""
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM quarantined_alerts").fetchone()
        return int(row[0]) if row else 0

    def suppression_add(self, meta_id: int, reason: str) -> None:
        """Suppress notification for one finalized meta ID (idempotent, first reason wins).

        Called by the dispatcher agent via getattr; never raises on re-suppress.
        """
        with sqlite3.connect(self.history_db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO suppressed_notifications (meta_id, reason) VALUES (?, ?)",
                (int(meta_id), str(reason or "")),
            )
            conn.commit()

    def suppression_count(self) -> int:
        """Return COUNT(*) of suppressed notification meta IDs."""
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM suppressed_notifications").fetchone()
        return int(row[0]) if row else 0

    def notification_add(self, meta_id: int, run_id: str, verdict: str) -> int:
        """Append one dispatcher notification verdict to notification_log.

        Called by the dispatcher agent via getattr; returns the AUTOINCREMENT
        row id of the appended verdict row.
        """
        with sqlite3.connect(self.history_db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO notification_log (meta_id, run_id, verdict) VALUES (?, ?, ?)",
                (int(meta_id), str(run_id), str(verdict)),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def notification_count(self) -> int:
        """Return COUNT(*) of logged dispatcher notification verdicts."""
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM notification_log").fetchone()
        return int(row[0]) if row else 0

    def notification_verdicts(self, limit: int = 100) -> List[str]:
        """Return recent notification verdicts, newest first (shadow-run evidence)."""
        with sqlite3.connect(self.history_db_path) as conn:
            rows = conn.execute(
                "SELECT verdict FROM notification_log ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def quarantine_id_set(self) -> Set[str]:
        """Return just quarantined IDs as a set (cheap per-cycle skip filter).

        Single-column SELECT, unlike :meth:`quarantine_list` which fetches
        display columns newest-first for operator status visibility.
        """
        with sqlite3.connect(self.history_db_path) as conn:
            rows = conn.execute("SELECT wazuh_alert_id FROM quarantined_alerts").fetchall()
        return {str(row[0]) for row in rows}

    def quarantine_list(self) -> List[Dict[str, Any]]:
        """Return quarantined alerts newest-first for operator status visibility."""
        with sqlite3.connect(self.history_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT wazuh_alert_id, source_index, source_document_id, error_type,"
                " count, first_seen, last_seen FROM quarantined_alerts"
                " ORDER BY last_seen DESC, wazuh_alert_id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_derivation_hash(self) -> Optional[str]:
        """Return the stored canonicalizer derivation hash, or None when unset."""
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT derivation_hash FROM derivation_state WHERE id = 1").fetchone()
        return str(row[0]) if row else None

    def set_derivation_hash(self, derivation_hash: str) -> None:
        """Persist the canonicalizer derivation hash for worker drift comparison."""
        with sqlite3.connect(self.history_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO derivation_state (id, derivation_hash) VALUES (1, ?)",
                (str(derivation_hash),),
            )
            conn.commit()

    @property
    def state_path(self) -> Path:
        """Return resolved path to the state file."""
        return self.filepath

    def _backup_legacy_mirror_once(self) -> None:
        """Copy a non-1.2 JSON mirror to .pre-1.2.bak exactly once before overwrite."""
        mirror_backup = self.filepath.with_name(self.filepath.name + ".pre-1.2.bak")
        try:
            if not self.filepath.exists() or mirror_backup.exists():
                return
            raw = self.filepath.read_text(encoding="utf-8")
            try:
                schema = json.loads(raw).get("schema_version") if raw.strip() else None
            except ValueError:
                schema = None
            if schema != "1.2":
                mirror_backup.write_text(raw, encoding="utf-8")
        except OSError:
            logger.warning("Legacy JSON mirror could not be backed up; proceeding with save")

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
        
        tmp_file = self.filepath.with_suffix(".tmp")

        # Seen IDs are append-only in SQLite so checkpoint cost is proportional
        # to new events rather than rewriting the full replay history as JSON.
        new_seen_ids = set(getattr(engine, "_new_seen_alert_ids", set()))

        # N8: fail closed on meta_id collision before any write in this checkpoint.
        self._assert_no_finalized_collision(list(new_finalized_history or []))

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
            "schema_version": "1.2",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "meta_id_counter": meta_id_counter,
            "temporal_states": temporal_states_data,
            "active_buckets": active_buckets_data,
            "source_checkpoint": source_checkpoint or {},
            "pending_scoring": pending_scoring or [],
            "outbox": outbox or [],
        }

        # Serialize before any writes. A single SQLite commit is authoritative
        # for both duplicate protection and the RBTA mutation it protects.
        serialized = json.dumps(payload, indent=2)
        history_rows = [(item["meta_id"], json.dumps(item)) for item in (new_finalized_history or [])]
        with sqlite3.connect(self.history_db_path) as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.executemany(
                "INSERT OR IGNORE INTO finalized_history (meta_id, scored_data) VALUES (?, ?)", history_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO seen_alert_ids (wazuh_alert_id) VALUES (?)",
                ((alert_id,) for alert_id in new_seen_ids),
            )
            conn.execute("INSERT OR REPLACE INTO runtime_snapshot (id, payload) VALUES (1, ?)", (serialized,))
        if hasattr(engine, "_new_seen_alert_ids"):
            engine._new_seen_alert_ids.difference_update(new_seen_ids)

        # F13c: preserve a pre-1.2 mirror exactly once before it is overwritten.
        self._backup_legacy_mirror_once()

        # Compatibility/export mirror only; recovery always prefers SQLite.
        try:
            tmp_file.write_text(serialized, encoding="utf-8")
            tmp_file.replace(self.filepath)
        except OSError:
            logger.warning("Runtime JSON mirror could not be published; committed SQLite snapshot remains authoritative")

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
        # The state directory may have been removed between manager init and
        # restore (e.g. operator cleanup). Recreate it so restore fails only
        # on genuine corruption, not on a missing directory.
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.history_db_path) as conn:
            row = conn.execute("SELECT payload FROM runtime_snapshot WHERE id = 1").fetchone()
            if row is None:
                # F13d fail-closed: a missing snapshot with surviving seen IDs
                # means a partial/corrupt restore — never hydrate dedup into an
                # empty engine, which would silently fork duplicate protection
                # (engine without buckets/temporal state but with seen IDs).
                # History-only state is still restorable: loading finalized
                # history touches no dedup structures.
                seen_count = conn.execute("SELECT COUNT(*) FROM seen_alert_ids").fetchone()[0]
                if int(seen_count or 0) > 0:
                    raise RuntimeError(
                        "Runtime SQLite snapshot is missing while seen_alert_ids"
                        " is non-empty; refusing to hydrate dedup state into an empty engine —"
                        " restore the complete runtime backup instead"
                    )
        if row:
            data = json.loads(row[0])
        elif self.filepath.exists():
            with self.filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") == "1.2":
                raise ValueError("Runtime SQLite snapshot missing; restore the complete runtime backup, not its JSON mirror")
        else:
            engine._seen_alert_ids = self.load_seen_alert_ids() if hydrate_seen_ids else set()
            engine._new_seen_alert_ids = set()
            # N8: history-only restore resumes the meta_id stream at MAX+1 so
            # the next bucket cannot collide with already-committed history.
            with sqlite3.connect(self.history_db_path) as conn:
                max_row = conn.execute("SELECT MAX(meta_id) FROM finalized_history").fetchone()
            engine._meta_id_counter = int(max_row[0]) + 1 if max_row and max_row[0] is not None else 1
            return {"outbox": [], "source_checkpoint": {}, "finalized_history": self.load_finalized_history()}

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
