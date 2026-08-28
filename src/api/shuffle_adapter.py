"""Shuffle SOAR webhook forwarder with retry and event idempotency header."""

import logging
from typing import Any, Dict, Optional, Tuple
import requests

from src.contracts.scored_meta_alert import ScoredMetaAlert

logger = logging.getLogger(__name__)


class ShuffleForwarderError(RuntimeError):
    """Raised when webhook dispatch to Shuffle fails."""
    pass


class ShuffleWebhookForwarder:
    """Dispatches finalized and scored MetaAlerts to Shuffle SOAR workflow webhooks.

    Parameters
    ----------
    webhook_url : str
        Target Shuffle webhook execution URL.
    api_key : str | None
        Optional webhook bearer token.
    timeout : Tuple[float, float]
        (connect_timeout, read_timeout) in seconds.
    max_retries : int
        Max retry attempts on transient network errors.
    """

    def __init__(
        self,
        webhook_url: str,
        api_key: Optional[str] = None,
        timeout: Tuple[float, float] = (5.0, 15.0),
        max_retries: int = 3,
    ) -> None:
        self.webhook_url: str = webhook_url
        self.api_key: Optional[str] = api_key
        self.timeout: Tuple[float, float] = timeout
        self.max_retries: int = max_retries
        self._session = requests.Session()

    def forward(self, scored_meta: ScoredMetaAlert) -> bool:
        """Post scored meta-alert payload to Shuffle webhook with idempotent X-Event-ID header.

        Parameters
        ----------
        scored_meta : ScoredMetaAlert
            Scored meta-alert to dispatch.

        Returns
        -------
        bool
            True if delivery succeeded (HTTP 200/202).
        """
        payload: Dict[str, Any] = {
            "meta_id": scored_meta.meta_id,
            "agent_id": scored_meta.agent_id,
            "agent_name": scored_meta.agent_name,
            "rule_group_primary": scored_meta.rule_group_primary,
            "start_time": scored_meta.start_time.isoformat(),
            "end_time": scored_meta.end_time.isoformat(),
            "alert_count": scored_meta.alert_count,
            "max_severity": scored_meta.max_severity,
            "mitre_tactics": list(scored_meta.mitre_tactics),
            "seven_features": dict(scored_meta.seven_features),
            "raw_model_score": scored_meta.raw_model_score,
            "anomaly_score": scored_meta.anomaly_score,
            "threshold_used": scored_meta.threshold_used,
            "decision": scored_meta.decision,
            "action": scored_meta.action,
            "escalate": scored_meta.escalate,
            "model_version": scored_meta.model_version,
            "source_alert_ids": list(scored_meta.source_alert_ids),
        }

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Event-ID": f"rbta-meta-{scored_meta.meta_id}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    self.webhook_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=True,
                )
                if resp.status_code in (200, 201, 202, 204):
                    return True
                logger.warning("Shuffle webhook returned status %s: %s", resp.status_code, resp.text)
            except Exception as exc:
                logger.warning("Error dispatching to Shuffle webhook (attempt %s/%s): %s", attempt, self.max_retries, exc)

        return False
