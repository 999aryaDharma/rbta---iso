"""Best-effort live alert archiver: raw JSONL replay dataset writer (Design A).

Every successfully ingested live alert is recorded as one compact JSON object
per line in ``data/archive/live-YYYYMMDD.jsonl`` (daily rotation by event-time
UTC date), so the live stream can later be replayed as a deterministic replay
dataset: each line is accepted by
:func:`src.etl.wazuh_canonicalizer.canonicalize_wazuh_alert`.

The live poller discards the original OpenSearch hit after canonicalization,
so the archive record is reconstructed from :class:`CanonicalRawAlert` plus
its preserved metadata (rule description/groups, location, full_log, decoder,
manager, data, source index/document id). Reconstruction is canonical-preserving
(the re-canonicalized alert is field-equivalent), not byte-identical.

Failure policy: archiving must never break live ingestion. All public entry
points swallow I/O errors, log at warning level, bump the ``failed`` counter,
and return a degraded result (``None`` / ``0``).
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping, Optional, Union

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.json_safe import deterministic_json_dumps, to_json_safe

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = "data/archive"

_stats_lock = threading.Lock()
_stats: Dict[str, Any] = {"archived": 0, "failed": 0, "last_error": None}


def get_stats() -> Dict[str, Any]:
    """Return a snapshot of archive counters (archived, failed, last_error)."""
    with _stats_lock:
        return dict(_stats)


def _bump(archived: int = 0, failed: int = 0, error: Optional[str] = None) -> None:
    with _stats_lock:
        _stats["archived"] += archived
        _stats["failed"] += failed
        if error is not None:
            _stats["last_error"] = error


def archive_enabled_from_env() -> bool:
    """Read the archive kill-switch (default False so replay/demo are unaffected)."""
    return os.environ.get("RBTA_LIVE_ARCHIVE_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def archive_dir_from_env(default: Union[str, Path] = DEFAULT_ARCHIVE_DIR) -> Path:
    """Read the archive directory override (default ``data/archive``)."""
    raw = os.environ.get("RBTA_LIVE_ARCHIVE_DIR", "")
    return Path(raw.strip()) if raw.strip() else Path(default)


def _meta(alert: CanonicalRawAlert) -> Dict[str, Any]:
    meta = alert.metadata
    return dict(meta) if isinstance(meta, Mapping) else {}


def build_archive_record(alert: CanonicalRawAlert) -> Dict[str, Any]:
    """Reconstruct a replay-compatible raw Wazuh alert dict from a canonical alert.

    Returns an OpenSearch-hit envelope (``{_index, _id, _source[, sort]}``)
    when the canonical metadata carries source lineage, otherwise the plain
    alert body. Both shapes are accepted by ``canonicalize_wazuh_alert``.
    """
    meta = _meta(alert)
    safe = to_json_safe(meta)

    rule: Dict[str, Any] = {"id": alert.rule_id, "level": alert.rule_level}
    description = safe.get("rule_description")
    if description:
        rule["description"] = description
    groups = safe.get("rule_groups_all")
    if isinstance(groups, (list, tuple)) and groups:
        rule["groups"] = list(groups)
    if alert.mitre_tactics:
        rule.setdefault("mitre", {})["tactic"] = list(alert.mitre_tactics)

    body: Dict[str, Any] = {
        "id": alert.wazuh_alert_id,
        "timestamp": alert.timestamp.isoformat(),
        "rule": rule,
        "agent": {"id": alert.agent_id, "name": alert.agent_name},
    }

    data_block = safe.get("data")
    if isinstance(data_block, dict):
        body["data"] = dict(data_block)
    if alert.srcip:
        data = body.setdefault("data", {})
        if isinstance(data, dict):
            data.setdefault("srcip", alert.srcip)
        body.setdefault("srcip", alert.srcip)

    for key in ("location", "full_log", "decoder", "manager"):
        value = safe.get(key)
        if value not in (None, ""):
            body[key] = value

    source_index = safe.get("source_index")
    source_doc_id = safe.get("source_document_id")
    if source_index or source_doc_id:
        envelope: Dict[str, Any] = {"_source": body}
        if source_index:
            envelope["_index"] = source_index
        if source_doc_id:
            envelope["_id"] = source_doc_id
        if safe.get("source_sort") is not None:
            envelope["sort"] = safe.get("source_sort")
        return envelope
    return body


def _event_time(alert: CanonicalRawAlert) -> datetime:
    ts = alert.timestamp
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _daily_path(archive_dir: Union[str, Path], event_time: datetime) -> Path:
    day = event_time.astimezone(timezone.utc).strftime("%Y%m%d")
    return Path(archive_dir) / f"live-{day}.jsonl"


def _append_line(path: Path, line: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                pass
        return True
    except Exception as exc:  # never propagate to the live cycle
        logger.warning("Live archive write failed for '%s': %s", path, exc)
        _bump(failed=1, error=str(exc))
        return False


def append_alert(
    alert: CanonicalRawAlert,
    archive_dir: Union[str, Path],
    event_time: Optional[datetime] = None,
) -> Optional[Path]:
    """Append one alert to the daily archive file. Never raises; None on failure."""
    try:
        record = build_archive_record(alert)
        line = deterministic_json_dumps(record)
        path = _daily_path(archive_dir, event_time or _event_time(alert))
        if _append_line(path, line):
            _bump(archived=1)
            return path
        return None
    except Exception as exc:  # defensive: build must not break ingestion either
        logger.warning("Live archive record build failed for '%s': %s", alert.wazuh_alert_id, exc)
        _bump(failed=1, error=str(exc))
        return None


def append_alerts(
    alerts: List[CanonicalRawAlert],
    archive_dir: Union[str, Path],
) -> int:
    """Append a batch in (event_time, alert_id) order, rotating per UTC day. Never raises."""
    try:
        ordered = sorted(alerts, key=lambda a: (_event_time(a), a.wazuh_alert_id))
    except Exception as exc:
        logger.warning("Live archive batch sort failed: %s", exc)
        _bump(failed=1, error=str(exc))
        return 0
    written = 0
    for alert in ordered:
        if append_alert(alert, archive_dir) is not None:
            written += 1
    return written
