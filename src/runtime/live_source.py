"""Live Wazuh Indexer polling source with fast recent polling and lossless reconciliation scanning."""

from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.ingestion.wazuh_client import WazuhIndexerClient

logger = logging.getLogger(__name__)


class LiveSourceError(RuntimeError):
    """Base exception for live source errors."""
    pass


class LiveSourceIntegrityError(LiveSourceError):
    """Raised when source response, pagination, or integrity contract is violated."""
    pass


class LiveCanonicalizationError(LiveSourceIntegrityError):
    """Raised when a source document fails canonicalization."""
    pass


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
        """Execute deterministic paginated search using @timestamp ASC and id ASC.

        Enforces strict fail-closed response validation and canonicalization integrity.
        """
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
            try:
                data = resp.json()
            except Exception as exc:
                raise LiveSourceIntegrityError(f"Indexer response is not valid JSON: {exc}") from exc

            # Strict response validation
            if not isinstance(data, dict):
                raise LiveSourceIntegrityError(
                    f"Malformed Indexer response: expected JSON object, got {type(data).__name__}"
                )
            if "hits" not in data or not isinstance(data["hits"], dict):
                raise LiveSourceIntegrityError(
                    "Malformed Indexer response: missing or invalid 'hits' dictionary"
                )
            if "hits" not in data["hits"] or not isinstance(data["hits"]["hits"], list):
                raise LiveSourceIntegrityError(
                    "Malformed Indexer response: missing or invalid 'hits.hits' list"
                )

            hits = data["hits"]["hits"]

            # Fail-closed canonicalization loop
            for pos, hit in enumerate(hits):
                try:
                    alert = canonicalize_wazuh_alert(hit)
                    new_alerts.append(alert)
                except Exception as exc:
                    doc_id = hit.get("_id") if isinstance(hit, dict) else None
                    idx_name = hit.get("_index") if isinstance(hit, dict) else None
                    raise LiveCanonicalizationError(
                        f"Failed to canonicalize document in live source (index={idx_name}, doc_id={doc_id}, page_pos={pos}): {exc}"
                    ) from exc

            # Termination condition
            if len(hits) < self.page_size:
                break

            # Pagination cursor validation on full pages
            last_hit = hits[-1]
            if not isinstance(last_hit, dict) or "sort" not in last_hit:
                raise LiveSourceIntegrityError(
                    f"Full page ({len(hits)} items) missing 'sort' field in final hit for search_after pagination"
                )

            search_after_cursor = last_hit["sort"]
            if not isinstance(search_after_cursor, (list, tuple)) or len(search_after_cursor) < 2:
                raise LiveSourceIntegrityError(
                    f"Full page final hit has invalid 'sort' cursor: {search_after_cursor}"
                )

        return new_alerts

    def discover_retained_daily_alert_indices(
        self,
        prefix: str = "wazuh-alerts-4.x-",
    ) -> List[str]:
        """Discover all retained Wazuh daily alert indices currently present on the Indexer.

        Parameters
        ----------
        prefix : str
            Index name prefix for daily alert indices.

        Returns
        -------
        List[str]
            Chronologically sorted list of discovered daily alert index names.
        """
        raw_indices = self.client.list_indices("wazuh-alerts-*")
        daily_indices: List[str] = []

        date_regex = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")

        for idx_name in raw_indices:
            if not isinstance(idx_name, str):
                continue
            if idx_name.startswith(prefix):
                suffix = idx_name[len(prefix):]
                if date_regex.match(suffix):
                    daily_indices.append(idx_name)
            elif re.match(r"^wazuh-alerts-(?:\d+\.x-)?\d{4}\.\d{2}\.\d{2}$", idx_name):
                daily_indices.append(idx_name)

        return sorted(list(set(daily_indices)))

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
        """Recent reconciliation scan path over retained daily indices (default 2 days: today and yesterday).

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
            All canonical alerts retrievable across the scanned recent daily indices.
        """
        now = end_time or current_time or datetime.now(timezone.utc)
        scan_start = start_time or (now - timedelta(days=max(0, reconciliation_days - 1)))
        indices = derive_daily_indices(scan_start, now)
        return self._paginate_search(indices=indices, start_time=start_time, end_time=end_time)

    def poll_full_reconciliation(
        self,
        prefix: str = "wazuh-alerts-4.x-",
    ) -> List[CanonicalRawAlert]:
        """Lossless full-retention reconciliation scan across all retained Wazuh daily alert indices.

        Discovers every daily alert index currently present in the Indexer and paginates all of them
        completely with zero timestamp-based cutoffs.

        Parameters
        ----------
        prefix : str
            Index name prefix to discover.

        Returns
        -------
        List[CanonicalRawAlert]
            All canonical alerts retrievable across all retained daily indices.
        """
        indices = self.discover_retained_daily_alert_indices(prefix=prefix)
        if not indices:
            return []
        return self._paginate_search(indices=indices, start_time=None, end_time=None)

    def poll_once(
        self,
        current_time: Optional[datetime] = None,
        recent_poll_cursor: Optional[datetime] = None,
        high_watermark: Optional[datetime] = None,
    ) -> List[CanonicalRawAlert]:
        """Convenience method for fast recent polling (deprecated high_watermark alias)."""
        cursor = recent_poll_cursor or high_watermark
        return self.poll_recent(current_time=current_time, recent_poll_cursor=cursor)
