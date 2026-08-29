import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.json_safe import (
    compute_canonical_fingerprint,
    deterministic_json_dumps,
    redact_sensitive_data,
    to_json_safe,
)


class RawEvidenceConflictError(Exception):
    """Raised when an alert with identical wazuh_alert_id has conflicting canonical evidence."""
    pass


class RawEvidenceIntegrityError(Exception):
    """Raised when persisted raw evidence data violates schema or timestamp integrity."""
    pass


class RawAlertEvidenceStore:
    """RawAlertEvidenceStore — SQLite-backed raw alert evidence persistence with WAL mode.

    Guarantees:
    - Evidence is persisted BEFORE core RBTA state mutation.
    - Idempotent duplicate: identical alert fingerprint -> NO-OP (returns False).
    - Conflicting duplicate: different alert fingerprint -> raises RawEvidenceConflictError (fail-closed).
    - Exact source membership traceability for MetaAlerts (source_total, resolved_total, unresolved_ids).
    - Multi-field search across alert ID, rule ID, description, IP, full log.
    - Optional presentation-level secret redaction on read APIs.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            import os
            db_path = os.environ.get("RBTA_RAW_EVIDENCE_DB", "data/runtime/raw_alert_evidence.sqlite3")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_alert_evidence (
                    wazuh_alert_id TEXT PRIMARY KEY,
                    canonical_fingerprint TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_level INTEGER NOT NULL,
                    rule_description TEXT DEFAULT '',
                    rule_group_primary TEXT NOT NULL,
                    rule_groups_all TEXT DEFAULT '[]',
                    mitre_tactics TEXT DEFAULT '[]',
                    mitre_techniques TEXT DEFAULT '[]',
                    srcip TEXT,
                    location TEXT DEFAULT '',
                    decoder TEXT DEFAULT '',
                    full_log TEXT DEFAULT '',
                    agent_criticality REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    original_source_payload TEXT,
                    source_index TEXT DEFAULT '',
                    source_document_id TEXT DEFAULT '',
                    source_mode TEXT DEFAULT 'LIVE',
                    ingested_at TEXT NOT NULL
                )
                """
            )
            # Automatic schema migration for existing sqlite databases
            cursor = conn.execute("PRAGMA table_info(raw_alert_evidence)")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            column_defs = {
                "canonical_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "rule_description": "TEXT DEFAULT ''",
                "rule_groups_all": "TEXT DEFAULT '[]'",
                "mitre_tactics": "TEXT DEFAULT '[]'",
                "mitre_techniques": "TEXT DEFAULT '[]'",
                "srcip": "TEXT",
                "location": "TEXT DEFAULT ''",
                "decoder": "TEXT DEFAULT ''",
                "full_log": "TEXT DEFAULT ''",
                "metadata": "TEXT DEFAULT '{}'",
                "original_source_payload": "TEXT",
                "source_index": "TEXT DEFAULT ''",
                "source_document_id": "TEXT DEFAULT ''",
                "source_mode": "TEXT DEFAULT 'LIVE'",
                "ingested_at": "TEXT NOT NULL DEFAULT ''",
            }
            for col_name, col_def in column_defs.items():
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE raw_alert_evidence ADD COLUMN {col_name} {col_def}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_alert_evidence (agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_rule ON raw_alert_evidence (rule_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_alert_evidence (timestamp)")

    def store(
        self,
        alert: CanonicalRawAlert,
        source_mode: str = "LIVE",
        original_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a canonical raw alert.

        Returns
        -------
        bool
            True if newly inserted, False if identical duplicate.

        Raises
        ------
        RawEvidenceConflictError
            If wazuh_alert_id already exists with conflicting canonical attributes.
        """
        meta = alert.metadata if isinstance(alert.metadata, dict) else dict(alert.metadata)

        # Extract audit/extended fields safely
        rule_desc = meta.get("rule_description", getattr(alert, "rule_description", ""))
        rule_groups_all = meta.get("rule_groups_all", [alert.rule_group_primary])
        mitre_techs = meta.get("mitre_techniques", ())
        loc = meta.get("location", "")
        dec = meta.get("decoder", "")
        if dec is None:
            dec = ""
        elif not isinstance(dec, str):
            dec = deterministic_json_dumps(dec)
        flog = meta.get("full_log", "")

        # Unify source index / doc id
        source_index = meta.get("source_index", meta.get("opensearch_index", ""))
        source_doc_id = meta.get("source_document_id", meta.get("opensearch_document_id", ""))

        fingerprint = compute_canonical_fingerprint(
            wazuh_alert_id=alert.wazuh_alert_id,
            timestamp=alert.timestamp,
            agent_id=alert.agent_id,
            agent_name=alert.agent_name,
            rule_id=alert.rule_id,
            rule_level=alert.rule_level,
            rule_group_primary=alert.rule_group_primary,
            srcip=alert.srcip or "",
            agent_criticality=alert.agent_criticality,
            mitre_tactics=alert.mitre_tactics,
            metadata=meta,
        )

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT canonical_fingerprint FROM raw_alert_evidence WHERE wazuh_alert_id = ?",
                (alert.wazuh_alert_id,),
            ).fetchone()

            if row is not None:
                existing_fp = row["canonical_fingerprint"]
                if existing_fp == fingerprint:
                    return False  # Identical duplicate -> safe NO-OP
                raise RawEvidenceConflictError(
                    f"Conflicting canonical evidence detected for wazuh_alert_id='{alert.wazuh_alert_id}'. "
                    f"Existing fingerprint '{existing_fp}' != incoming '{fingerprint}'."
                )

            ingested_at = datetime.now(timezone.utc).isoformat()
            ts_str = alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else str(alert.timestamp)
            original_payload_str = deterministic_json_dumps(original_payload) if original_payload is not None else None

            conn.execute(
                """
                INSERT INTO raw_alert_evidence (
                    wazuh_alert_id, canonical_fingerprint, timestamp, agent_id, agent_name,
                    rule_id, rule_level, rule_description, rule_group_primary, rule_groups_all,
                    mitre_tactics, mitre_techniques, srcip, location, decoder,
                    full_log, agent_criticality, metadata, original_source_payload,
                    source_index, source_document_id, source_mode, ingested_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    alert.wazuh_alert_id,
                    fingerprint,
                    ts_str,
                    alert.agent_id,
                    alert.agent_name,
                    alert.rule_id,
                    alert.rule_level,
                    rule_desc,
                    alert.rule_group_primary,
                    json.dumps(to_json_safe(rule_groups_all)),
                    json.dumps(to_json_safe(alert.mitre_tactics)),
                    json.dumps(to_json_safe(mitre_techs)),
                    alert.srcip or "",
                    loc,
                    dec,
                    flog,
                    float(alert.agent_criticality),
                    deterministic_json_dumps(meta),
                    original_payload_str,
                    source_index,
                    source_doc_id,
                    source_mode or "LIVE",
                    ingested_at,
                ),
            )
            return True

    def _row_to_dict(self, row: sqlite3.Row, redact: bool = False) -> Dict[str, Any]:
        d = dict(row)
        d["rule_groups_all"] = json.loads(d.get("rule_groups_all") or "[]")
        d["mitre_tactics"] = json.loads(d.get("mitre_tactics") or "[]")
        d["mitre_techniques"] = json.loads(d.get("mitre_techniques") or "[]")
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except Exception:
            d["metadata"] = {}

        if d.get("original_source_payload"):
            try:
                d["original_source_payload"] = json.loads(d["original_source_payload"])
            except Exception:
                pass

        if redact:
            d = redact_sensitive_data(d)
        return d

    def get(self, wazuh_alert_id: str, redact: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieve a single raw alert evidence record by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM raw_alert_evidence WHERE wazuh_alert_id = ?",
                (wazuh_alert_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_dict(row, redact=redact)

    def get_many(self, wazuh_alert_ids: List[str], redact: bool = False) -> Dict[str, Dict[str, Any]]:
        """Retrieve multiple raw alert evidence records mapped by ID."""
        if not wazuh_alert_ids:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        with self._get_conn() as conn:
            # Batch queries in chunks of 500 to respect SQLite parameter limits
            chunk_size = 500
            for i in range(0, len(wazuh_alert_ids), chunk_size):
                chunk = wazuh_alert_ids[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT * FROM raw_alert_evidence WHERE wazuh_alert_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    rec = self._row_to_dict(row, redact=redact)
                    results[rec["wazuh_alert_id"]] = rec
        return results

    def get_meta_alert_raw_alerts(
        self,
        source_alert_ids: List[str],
        meta_id: Optional[int] = None,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        rule_id: Optional[str] = None,
        level_min: Optional[int] = None,
        level_max: Optional[int] = None,
        srcip: Optional[str] = None,
        mitre_tactic: Optional[str] = None,
        redact: bool = True,
    ) -> Dict[str, Any]:
        """Resolve member raw alerts for a MetaAlert with strict resolution traceability and filtering."""
        source_total = len(source_alert_ids)
        resolved_map = self.get_many(source_alert_ids, redact=redact)
        resolved_total = len(resolved_map)

        # Identify unresolved alert IDs while preserving canonical source order
        unresolved_alert_ids = [aid for aid in source_alert_ids if aid not in resolved_map]

        # Gather resolved records in canonical source order
        resolved_items = [resolved_map[aid] for aid in source_alert_ids if aid in resolved_map]

        # Apply multi-field search and filters
        filtered_items = []
        search_lower = search.strip().lower() if search else None

        for item in resolved_items:
            if rule_id and item.get("rule_id") != rule_id:
                continue
            if level_min is not None and item.get("rule_level", 0) < level_min:
                continue
            if level_max is not None and item.get("rule_level", 0) > level_max:
                continue
            if srcip and item.get("srcip") != srcip:
                continue
            if mitre_tactic and mitre_tactic not in item.get("mitre_tactics", []):
                continue
            if search_lower:
                match = (
                    search_lower in item.get("wazuh_alert_id", "").lower()
                    or search_lower in item.get("rule_id", "").lower()
                    or search_lower in item.get("rule_description", "").lower()
                    or search_lower in item.get("full_log", "").lower()
                    or search_lower in item.get("srcip", "").lower()
                )
                if not match:
                    continue
            filtered_items.append(item)

        filtered_total = len(filtered_items)
        start_idx = max(0, (page - 1) * page_size)
        end_idx = start_idx + page_size
        paginated_items = filtered_items[start_idx:end_idx]

        return {
            "meta_id": meta_id,
            "source_total": source_total,
            "resolved_total": resolved_total,
            "filtered_total": filtered_total,
            "unresolved_alert_ids": unresolved_alert_ids,
            "page": page,
            "page_size": page_size,
            "items": paginated_items,
        }

    def search(
        self,
        query: Optional[str] = None,
        rule_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        srcip: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        redact: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search raw alert evidence across multiple indices."""
        conditions = []
        params: List[Any] = []

        if query:
            conditions.append(
                "(wazuh_alert_id LIKE ? OR rule_id LIKE ? OR rule_description LIKE ? OR full_log LIKE ? OR srcip LIKE ?)"
            )
            like = f"%{query}%"
            params.extend([like, like, like, like, like])
        if rule_id:
            conditions.append("rule_id = ?")
            params.append(rule_id)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if srcip:
            conditions.append("srcip = ?")
            params.append(srcip)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM raw_alert_evidence {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r, redact=redact) for r in rows]

    def count(self) -> int:
        """Return total number of stored raw alert evidence records."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM raw_alert_evidence").fetchone()
            return row["total"] if row else 0

    def count_by_hour(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, int]:
        """Aggregate raw alert counts into UTC hourly buckets within [start_time, end_time].

        Returns
        -------
        Dict[str, int]
            Map of 'YYYY-MM-DD HH:00' -> count of raw alerts with timestamps in that hour.
        """
        st_utc = start_time.astimezone(timezone.utc) if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
        et_utc = end_time.astimezone(timezone.utc) if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)
        start_iso = st_utc.isoformat()
        end_iso = et_utc.isoformat()

        sql = "SELECT wazuh_alert_id, timestamp FROM raw_alert_evidence WHERE timestamp >= ? AND timestamp <= ?"
        hourly_counts: Dict[str, int] = {}
        with self._get_conn() as conn:
            rows = conn.execute(sql, (start_iso, end_iso)).fetchall()
            for r in rows:
                ts_raw = r["timestamp"]
                try:
                    if isinstance(ts_raw, str):
                        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
                    else:
                        dt = ts_raw.astimezone(timezone.utc)
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
                except Exception as exc:
                    raise RawEvidenceIntegrityError(
                        f"Corrupt timestamp '{ts_raw}' encountered in raw evidence record '{r['wazuh_alert_id']}': {exc}"
                    ) from exc
        return hourly_counts
