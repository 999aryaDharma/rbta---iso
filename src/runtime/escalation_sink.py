"""Escalation dispatch sink interface and deferred Telegram file output adapter."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from collections import deque
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set

from src.contracts.scored_meta_alert import ScoredMetaAlert

logger = logging.getLogger(__name__)


class EscalationSink(ABC):
    """Abstract base class for escalation alert dispatch sinks."""

    @abstractmethod
    def emit(self, scored: ScoredMetaAlert, run_id: str) -> bool:
        """Emit an escalation payload. Returns True if emitted, False if skipped/deduplicated."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_payloads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve latest emitted payloads for auditing/inspection."""
        raise NotImplementedError

    @abstractmethod
    def get_total_count(self) -> int:
        """Retrieve total count of emitted payloads."""
        raise NotImplementedError


class DeferredTelegramFileSink(EscalationSink):
    """Append-only thread-safe file sink for deferred Telegram escalation payloads.

    Writes one JSON object per line (JSONL format) to a local audit file with
    idempotency enforcement on (run_id, meta_id).
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen_idempotency_keys: Set[str] = set()
        self._recent_payloads: deque = deque(maxlen=100)
        self._load_existing_keys()

    def _load_existing_keys(self) -> None:
        """Scan existing file if present to populate seen idempotency keys and recent payloads."""
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        obj = json.loads(line_str)
                        key = obj.get("idempotency_key")
                        if key:
                            self._seen_idempotency_keys.add(key)
                        self._recent_payloads.append(obj)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Could not read existing Telegram payload sink keys: %s", e)

    def emit(self, scored: ScoredMetaAlert, run_id: str) -> bool:
        """Emit scored meta-alert to deferred Telegram file sink if action == ESCALATE."""
        if scored.action != "ESCALATE":
            return False

        idempotency_key = f"{run_id}:{scored.meta_id}"

        with self._lock:
            if idempotency_key in self._seen_idempotency_keys:
                logger.debug("Skipping duplicate escalation payload for key '%s'", idempotency_key)
                return False

            now_iso = datetime.now(timezone.utc).isoformat()
            threshold_val = float(scored.threshold_used)
            score_val = float(scored.anomaly_score)

            message = (
                f"[{scored.decision}] MetaAlert #{scored.meta_id} | "
                f"{scored.rule_group_primary} | {scored.alert_count} alerts | "
                f"severity {scored.max_severity} | "
                f"anomaly {score_val:.6f} > threshold {threshold_val:.6f}"
            )

            payload = {
                "timestamp": now_iso,
                "run_id": run_id,
                "meta_id": scored.meta_id,
                "idempotency_key": idempotency_key,
                "decision": scored.decision,
                "action": scored.action,
                "anomaly_score": round(score_val, 6),
                "threshold": round(threshold_val, 6),
                "model_version": scored.model_version,
                "agent_id": scored.agent_id,
                "agent_name": scored.agent_name,
                "rule_group_primary": scored.rule_group_primary,
                "alert_count": scored.alert_count,
                "max_severity": scored.max_severity,
                "message": message,
            }

            try:
                line = json.dumps(payload, ensure_ascii=False) + "\n"
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                self._seen_idempotency_keys.add(idempotency_key)
                self._recent_payloads.append(payload)
                logger.info("Emitted deferred Telegram payload for %s", idempotency_key)
                return True
            except Exception as e:
                logger.error("Failed to append deferred Telegram payload for %s: %s", idempotency_key, e)
                raise RuntimeError(f"Deferred Telegram sink write error: {e}") from e

    def get_latest_payloads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the latest N payloads quickly from memory cache."""
        with self._lock:
            if not self._recent_payloads and self.file_path.exists():
                self._load_existing_keys()
            items = list(self._recent_payloads)
            return items[-limit:] if limit < len(items) else items

    def get_total_count(self) -> int:
        """Return total number of recorded escalation payloads."""
        with self._lock:
            return len(self._seen_idempotency_keys)
