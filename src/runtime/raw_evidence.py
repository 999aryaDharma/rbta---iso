import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.contracts.raw_alert import CanonicalRawAlert

class RawAlertEvidenceStore:
    """RawAlertEvidenceStore — SQLite-backed raw alert evidence persistence.

    Evidence is written BEFORE core RBTA mutation during alert ingestion.
    This is an audit layer, NOT a research feature source.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            import os
            db_path = os.environ.get("RBTA_RAW_EVIDENCE_DB", "data/runtime/raw_alert_evidence.sqlite3")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_alert_evidence (
                    wazuh_alert_id TEXT PRIMARY KEY,
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
                    agent_criticality INTEGER NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    original_source_payload TEXT,
                    opensearch_index TEXT DEFAULT '',
                    opensearch_document_id TEXT DEFAULT '',
                    source_mode TEXT DEFAULT '',
                    ingested_at TEXT NOT NULL
                )
                """
            )
            # Create some indexes for search performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_alert_evidence (agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_rule ON raw_alert_evidence (rule_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_alert_evidence (timestamp)")

    def store(self, alert: CanonicalRawAlert, source_mode: str = '', original_payload: dict | None = None) -> bool:
        """Store a raw alert idempotently. Returns True if newly inserted, False if it already existed."""
        ingested_at = datetime.now(timezone.utc).isoformat()
        original_payload_str = json.dumps(original_payload) if original_payload is not None else None

        query = """
            INSERT OR IGNORE INTO raw_alert_evidence (
                wazuh_alert_id, timestamp, agent_id, agent_name, rule_id, rule_level,
                rule_description, rule_group_primary, rule_groups_all, mitre_tactics,
                mitre_techniques, srcip, location, decoder, full_log, agent_criticality,
                metadata, original_source_payload, opensearch_index, opensearch_document_id,
                source_mode, ingested_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        """

        meta = alert.metadata if isinstance(alert.metadata, dict) else dict(alert.metadata)
        rule_desc = meta.get("rule_description", getattr(alert, "rule_description", ""))
        rule_groups_all = meta.get("rule_groups_all", getattr(alert, "rule_groups_all", [alert.rule_group_primary]))
        mitre_techs = meta.get("mitre_techniques", getattr(alert, "mitre_techniques", ()))
        loc = meta.get("location", getattr(alert, "location", ""))
        dec = meta.get("decoder", getattr(alert, "decoder", ""))
        flog = meta.get("full_log", getattr(alert, "full_log", ""))
        os_idx = meta.get("opensearch_index", getattr(alert, "opensearch_index", ""))
        os_doc_id = meta.get("opensearch_document_id", getattr(alert, "opensearch_document_id", ""))

        params = (
            alert.wazuh_alert_id,
            alert.timestamp.isoformat(),
            alert.agent_id,
            alert.agent_name,
            alert.rule_id,
            alert.rule_level,
            str(rule_desc or ""),
            alert.rule_group_primary,
            json.dumps(list(rule_groups_all)),
            json.dumps(list(alert.mitre_tactics)),
            json.dumps(list(mitre_techs)),
            alert.srcip,
            str(loc or ""),
            str(dec or ""),
            str(flog or ""),
            alert.agent_criticality,
            json.dumps(meta),
            original_payload_str,
            str(os_idx or ""),
            str(os_doc_id or ""),
            source_mode,
            ingested_at
        )

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for json_field in ['rule_groups_all', 'mitre_tactics', 'mitre_techniques', 'metadata']:
            if d.get(json_field):
                try:
                    d[json_field] = json.loads(d[json_field])
                except json.JSONDecodeError:
                    pass
        if d.get('original_source_payload'):
            try:
                d['original_source_payload'] = json.loads(d['original_source_payload'])
            except json.JSONDecodeError:
                pass
        return d

    def get(self, wazuh_alert_id: str) -> dict | None:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM raw_alert_evidence WHERE wazuh_alert_id = ?", (wazuh_alert_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def get_many(self, alert_ids: list[str]) -> list[dict]:
        if not alert_ids:
            return []

        placeholders = ",".join("?" * len(alert_ids))
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"SELECT * FROM raw_alert_evidence WHERE wazuh_alert_id IN ({placeholders})",
                tuple(alert_ids)
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search(
        self,
        meta_id_alert_ids: list[str],
        page: int = 1,
        page_size: int = 50,
        search: str | None = None,
        rule_id: str | None = None,
        level_min: int | None = None,
        level_max: int | None = None,
        srcip: str | None = None,
        mitre_tactic: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None
    ) -> tuple[list[dict], int]:
        if not meta_id_alert_ids:
            return [], 0

        where_clauses = []
        params = []

        placeholders = ",".join("?" * len(meta_id_alert_ids))
        where_clauses.append(f"wazuh_alert_id IN ({placeholders})")
        params.extend(meta_id_alert_ids)

        if search:
            where_clauses.append("(rule_description LIKE ? OR full_log LIKE ?)")
            like_val = f"%{search}%"
            params.extend([like_val, like_val])

        if rule_id:
            where_clauses.append("rule_id = ?")
            params.append(rule_id)

        if level_min is not None:
            where_clauses.append("rule_level >= ?")
            params.append(level_min)

        if level_max is not None:
            where_clauses.append("rule_level <= ?")
            params.append(level_max)

        if srcip:
            where_clauses.append("srcip = ?")
            params.append(srcip)

        if mitre_tactic:
            where_clauses.append("mitre_tactics LIKE ?")
            params.append(f"%\"{mitre_tactic}\"%")

        if from_ts:
            where_clauses.append("timestamp >= ?")
            params.append(from_ts)

        if to_ts:
            where_clauses.append("timestamp <= ?")
            params.append(to_ts)

        where_sql = " AND ".join(where_clauses)

        count_query = f"SELECT COUNT(*) FROM raw_alert_evidence WHERE {where_sql}"

        offset = (page - 1) * page_size

        data_query = f"""
            SELECT * FROM raw_alert_evidence
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """

        data_params = list(params)
        data_params.extend([page_size, offset])

        with self._get_conn() as conn:
            total = conn.execute(count_query, params).fetchone()[0]
            cursor = conn.execute(data_query, data_params)
            items = [self._row_to_dict(row) for row in cursor.fetchall()]

        return items, total

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM raw_alert_evidence").fetchone()[0]
