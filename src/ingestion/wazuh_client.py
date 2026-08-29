"""Authenticated HTTPS client for Wazuh Indexer (OpenSearch-compatible API)."""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


class WazuhClientError(RuntimeError):
    """Base error for Wazuh Indexer client operations."""
    pass


class WazuhAuthError(WazuhClientError):
    """Raised when authentication (401/403) to Wazuh Indexer fails."""
    pass


class WazuhIndexerClient:
    """Secure client for communicating with Wazuh Indexer via OpenSearch REST API.

    Parameters
    ----------
    base_url : str
        Wazuh Indexer base URL (e.g. "https://172.16.83.180:9200").
    username : str | None
        HTTP Basic Auth username (defaults to WAZUH_INDEXER_USERNAME env var).
    password : str | None
        HTTP Basic Auth password (defaults to WAZUH_INDEXER_PASSWORD env var).
    verify_tls : bool | str
        SSL verification setting. True (default), False, or path to CA bundle.
    timeout : Tuple[float, float]
        (connect_timeout, read_timeout) in seconds.
    max_retries : int
        Max retries on transient network/5xx errors.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_tls: Union[bool, str] = True,
        timeout: Tuple[float, float] = (5.0, 30.0),
        max_retries: int = 3,
        sleep_fn=None,
        random_fn=None,
    ) -> None:
        self.base_url: str = (base_url or os.getenv("WAZUH_INDEXER_URL", "https://localhost:9200")).rstrip("/")
        self.username: Optional[str] = username or os.getenv("WAZUH_INDEXER_USERNAME")
        self.password: Optional[str] = password or os.getenv("WAZUH_INDEXER_PASSWORD")
        self.verify_tls: Union[bool, str] = verify_tls
        self.timeout: Tuple[float, float] = timeout
        self.max_retries: int = max_retries

        self._sleep_fn = sleep_fn or time.sleep
        self._random_fn = random_fn or (lambda: __import__('random').random())

        self._session = requests.Session()
        if self.username and self.password:
            self._session.auth = HTTPBasicAuth(self.username, self.password)

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Execute authenticated request with transient error retry and 401/403 fail-fast."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    params=params,
                    verify=self.verify_tls,
                    timeout=self.timeout,
                )

                # 401 / 403 Fail Fast
                if resp.status_code in (401, 403):
                    raise WazuhAuthError(
                        f"Authentication failed ({resp.status_code}) against Wazuh Indexer at '{self.base_url}': {resp.text}"
                    )

                # Transient server errors (429, 502, 503, 504) -> retry
                if resp.status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                    delay = min(30.0, (2 ** (attempt - 1)) * 0.5) + self._random_fn() * 0.5
                    self._sleep_fn(delay)
                    continue

                resp.raise_for_status()
                return resp

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise WazuhClientError(f"Network failure calling '{url}' after {self.max_retries} attempts: {exc}") from exc
                delay = min(30.0, (2 ** (attempt - 1)) * 0.5) + self._random_fn() * 0.5
                self._sleep_fn(delay)
            except WazuhAuthError:
                raise
            except requests.exceptions.HTTPError as exc:
                raise WazuhClientError(f"HTTP error ({resp.status_code}) calling '{url}': {resp.text}") from exc

        raise WazuhClientError(f"Exceeded max retries calling '{url}'")

    def list_indices(self, pattern: str = "wazuh-alerts-*") -> List[str]:
        """List cluster index names matching the pattern.

        Parameters
        ----------
        pattern : str
            Index pattern wildcard (e.g. 'wazuh-alerts-*').

        Returns
        -------
        List[str]
            List of matching index names.
        """
        resp = self._request("GET", f"/_cat/indices/{pattern}", params={"format": "json"})
        indices_data = resp.json()
        return [str(item["index"]) for item in indices_data if "index" in item]

    def create_point_in_time(self, index_name: str, keep_alive: str = "5m") -> str:
        """Open a Point-In-Time (PIT) snapshot on a single daily index.

        Parameters
        ----------
        index_name : str
            Target daily index name.
        keep_alive : str
            PIT retention duration string (e.g. '5m').

        Returns
        -------
        str
            Unique Point-In-Time ID.

        Raises
        ------
        WazuhClientError
            If PIT creation fails or contains partial shard failures.
        """
        resp = self._request("POST", f"/{index_name}/_search/point_in_time", params={"keep_alive": keep_alive})
        data = resp.json()

        shards = data.get("_shards", {})
        if shards.get("failed", 0) > 0:
            raise WazuhClientError(
                f"Partial PIT creation rejected for index '{index_name}': {shards.get('failed')} shards failed out of {shards.get('total')}"
            )

        pit_id = data.get("pit_id")
        if not pit_id:
            raise WazuhClientError(f"No pit_id returned in response for index '{index_name}'")

        return str(pit_id)

    def close_point_in_time(self, pit_id: str) -> bool:
        """Close an active Point-In-Time snapshot context.

        Parameters
        ----------
        pit_id : str
            Point-In-Time ID to close.

        Returns
        -------
        bool
            True if closure succeeded.
        """
        try:
            resp = self._request("DELETE", "/_search/point_in_time", json_data={"pit_id": [pit_id]})
            data = resp.json()
            return bool(data.get("succeeded", True))
        except Exception as exc:
            logger.warning("Error closing PIT '%s': %s", pit_id, exc)
            return False

    def search_page(
        self,
        pit_id: str,
        page_size: int = 500,
        search_after: Optional[Sequence[Any]] = None,
        sort: Optional[Sequence[Dict[str, str]]] = None,
        keep_alive: str = "5m",
    ) -> List[Dict[str, Any]]:
        """Fetch a page of hits from a Point-In-Time using search_after.

        Parameters
        ----------
        pit_id : str
            Active Point-In-Time ID.
        page_size : int
            Number of hits per page.
        search_after : Sequence[Any] | None
            Cursor values from the last hit of the preceding page.
        sort : Sequence[Dict[str, str]] | None
            Exact sort criteria (default: [{"@timestamp": "asc"}, {"id": "asc"}]).
        keep_alive : str
            PIT keep_alive extension duration.

        Returns
        -------
        List[Dict[str, Any]]
            List of raw OpenSearch hit objects.
        """
        sort_criteria = sort or [{"@timestamp": "asc"}, {"id": "asc"}]
        body: Dict[str, Any] = {
            "size": page_size,
            "pit": {"id": pit_id, "keep_alive": keep_alive},
            "sort": list(sort_criteria),
            "track_total_hits": False,
        }
        if search_after:
            body["search_after"] = list(search_after)

        resp = self._request("POST", "/_search", json_data=body)
        data = resp.json()
        hits_container = data.get("hits", {})
        return list(hits_container.get("hits", []))
