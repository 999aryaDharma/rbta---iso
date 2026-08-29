"""Live Wazuh Indexer polling source with fast recent polling and lossless reconciliation scanning."""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.ingestion.wazuh_client import WazuhIndexerClient

logger = logging.getLogger(__name__)


def derive_daily_indices(
    start_time: datetime,
    end_time: datetime,
    prefix: str = "wazuh-alerts-4.x-",
) -> List[str]:
    """Derive exact list of daily index names for all UTC days between start_time and end_time (inclusive).

    Parameters
    ----------
    start_time : datetime
        Start boundary timestamp.
    end_time : datetime
        End boundary timestamp.
    prefix : str
        Wazuh daily index prefix (default 'wazuh-alerts-4.x-').

    Returns
    -------
    List[str]
        Chronologically sorted list of exact daily index names.
    """
    start_date = start_time.astimezone(timezone.utc).date()
    end_date = end_time.astimezone(timezone.utc).date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    indices = []
    curr = start_date
    while curr <= end_date:
        indices.append(f"{prefix}{curr.strftime('%Y.%m.%d')}")
        curr += timedelta(days=1)
    return indices


class WazuhIndexerLivePoller:
    """Polls Wazuh Indexer for raw security alerts using fast recent polling and reconciliation scanning.

    Parameters
    ----------
    client : WazuhIndexerClient | None
        Wazuh client instance.
    overlap_window : timedelta
        Lookback overlap window for recent polling (default 5 minutes).
    poll_interval : timedelta
        Interval between fast polls (default 5 seconds).
    page_size : int
        Max hits per search request (default 500).
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

    def _paginate_search(
        self,
        indices: Sequence[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Execute deterministic paginated search using @timestamp ASC and id ASC."""
        if not indices:
            return []

        target_endpoint = f"/{','.join(indices)}/_search"
        sort_spec = [{"@timestamp": "asc"}, {"id": "asc"}]

        query_range: Dict[str, Any] = {}
        if start_time is not None:
            query_range["gte"] = start_time.isoformat()
        if end_time is not None:
            query_range["lte"] = end_time.isoformat()

        query_body: Dict[str, Any] = {
            "size": self.page_size,
            "sort": sort_spec,
        }
        if query_range:
            query_body["query"] = {"range": {"@timestamp": query_range}}

        new_alerts: List[CanonicalRawAlert] = []
        search_after_cursor = None

        while True:
            current_body = dict(query_body)
            if search_after_cursor is not None:
                current_body["search_after"] = search_after_cursor

            resp = self.client._request("POST", target_endpoint, json_data=current_body)
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

    def poll_recent(
        self,
        current_time: Optional[datetime] = None,
        recent_poll_cursor: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Fast recent polling path for low-latency alert discovery.

        Parameters
        ----------
        current_time : datetime | None
            Reference time for poll cycle (defaults to UTC now).
        recent_poll_cursor : datetime | None
            Starting hint timestamp for fast polling.

        Returns
        -------
        List[CanonicalRawAlert]
            List of raw alerts found within the recent polling window.
        """
        now = current_time or datetime.now(timezone.utc)
        start_time = (recent_poll_cursor or now) - self.overlap_window
        indices = derive_daily_indices(start_time, now)
        return self._paginate_search(indices=indices, start_time=start_time, end_time=now)

    def poll_reconciliation(
        self,
        current_time: Optional[datetime] = None,
        reconciliation_days: int = 2,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Lossless reconciliation scan path over retained daily indices.

        Parameters
        ----------
        current_time : datetime | None
            Reference time for the scan.
        reconciliation_days : int
            Number of recent daily indices to scan (default 2: today and yesterday).
        start_time : datetime | None
            Optional explicit start time override.
        end_time : datetime | None
            Optional explicit end time override.

        Returns
        -------
        List[CanonicalRawAlert]
            All canonical alerts retrievable across the scanned daily indices.
        """
        now = end_time or current_time or datetime.now(timezone.utc)
        scan_start = start_time or (now - timedelta(days=max(0, reconciliation_days - 1)))
        indices = derive_daily_indices(scan_start, now)
        return self._paginate_search(indices=indices, start_time=start_time, end_time=end_time)

    def poll_once(
        self,
        current_time: Optional[datetime] = None,
        recent_poll_cursor: Optional[datetime] = None,
        high_watermark: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Convenience method for fast recent polling (accepts legacy high_watermark alias)."""
        cursor = recent_poll_cursor or high_watermark
        return self.poll_recent(current_time=current_time, recent_poll_cursor=cursor)
