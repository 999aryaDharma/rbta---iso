"""Live Wazuh Indexer polling source with overlap window and deduplication."""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Set

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.ingestion.wazuh_client import WazuhIndexerClient

logger = logging.getLogger(__name__)


class WazuhIndexerLivePoller:
    """Polls Wazuh Indexer for newly indexed alerts within an overlap time window.

    Parameters
    ----------
    client : WazuhIndexerClient | None
        Wazuh client instance.
    overlap_window : timedelta
        Time window to look back on each poll (default 5 minutes).
    poll_interval : timedelta
        Interval between polls (default 5 seconds).
    page_size : int
        Max hits per poll request (default 500).
    """

    def __init__(
        self,
        client: Optional[WazuhIndexerClient] = None,
        overlap_window: timedelta = timedelta(minutes=5),
        poll_interval: timedelta = timedelta(seconds=5),
        page_size: int = 500,
    ) -> None:
        self.client: WazuhIndexerClient = client or WazuhIndexerClient()
        self.overlap_window: timedelta = overlap_window
        self.poll_interval: timedelta = poll_interval
        self.page_size: int = page_size

    def poll_once(
        self,
        current_time: Optional[datetime] = None,
        high_watermark: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Execute a single live polling cycle.

        Parameters
        ----------
        current_time : datetime | None
            Reference time for poll cycle (defaults to UTC now).
        high_watermark : datetime | None
            Latest processed event timestamp to anchor the overlap window.

        Returns
        -------
        List[CanonicalRawAlert]
            List of newly discovered canonical alerts in chronological order.
        """
        now = current_time or datetime.now(timezone.utc)
        start_time = (high_watermark or now) - self.overlap_window

        new_alerts: List[CanonicalRawAlert] = []
        search_after_cursor = None
        sort_spec = [{"@timestamp": "asc"}, {"id": "asc"}]

        # OpenSearch index pattern covering all daily indices (supports midnight rollover)
        target_endpoint = "/wazuh-alerts-*/_search"

        while True:
            query_body = {
                "size": self.page_size,
                "sort": sort_spec,
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": now.isoformat(),
                        }
                    }
                },
            }
            if search_after_cursor is not None:
                query_body["search_after"] = search_after_cursor

            resp = self.client._request("POST", target_endpoint, json_data=query_body)
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits:
                try:
                    alert = canonicalize_wazuh_alert(hit)
                    new_alerts.append(alert)
                except Exception as exc:
                    logger.warning("Failed to canonicalize hit in live poller: %s", exc)

            if len(hits) < self.page_size:
                break

            search_after_cursor = hits[-1].get("sort")
            if search_after_cursor is None:
                break

        return new_alerts
