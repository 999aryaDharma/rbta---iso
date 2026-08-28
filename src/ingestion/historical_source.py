"""Historical Wazuh Indexer acquisition source with PIT and checkpointing."""

import fnmatch
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import canonicalize_wazuh_alert
from src.ingestion.checkpoint import CheckpointManager, HistoricalCheckpoint
from src.ingestion.wazuh_client import WazuhIndexerClient

logger = logging.getLogger(__name__)


class WazuhIndexerHistoricalSource:
    """Historical data source pulling alerts from Wazuh Indexer via daily PIT snapshots.

    Parameters
    ----------
    client : WazuhIndexerClient
        Authenticated Wazuh Indexer client instance.
    checkpoint_manager : CheckpointManager | None
        Persistence manager for resuming interrupted exports.
    page_size : int
        Number of hits per search page (default 500).
    keep_alive : str
        PIT keep_alive duration (default '5m').
    """

    def __init__(
        self,
        client: Optional[WazuhIndexerClient] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        page_size: int = 500,
        keep_alive: str = "5m",
    ) -> None:
        self.client: WazuhIndexerClient = client or WazuhIndexerClient()
        self.checkpoint_manager: CheckpointManager = checkpoint_manager or CheckpointManager()
        self.page_size: int = page_size
        self.keep_alive: str = keep_alive

    def discover_indices(self, pattern: str = "wazuh-alerts-4.x-*") -> List[str]:
        """Discover and sort available daily indices matching pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern to match daily indices against.

        Returns
        -------
        List[str]
            Ascending sorted list of daily index names.
        """
        all_indices = self.client.list_indices(pattern=pattern)
        filtered = [idx for idx in all_indices if fnmatch.fnmatch(idx, pattern)]
        filtered.sort()
        return filtered

    def stream_canonical_alerts(
        self,
        index_pattern: str = "wazuh-alerts-4.x-*",
    ) -> Iterator[CanonicalRawAlert]:
        """Yield CanonicalRawAlert objects across discovered daily indices with checkpointing.

        Parameters
        ----------
        index_pattern : str
            Daily index discovery pattern.

        Yields
        ------
        CanonicalRawAlert
            Normalized canonical alerts.
        """
        checkpoint: HistoricalCheckpoint = self.checkpoint_manager.load()
        indices = self.discover_indices(pattern=index_pattern)

        for index_name in indices:
            if index_name in checkpoint.completed_indices:
                logger.info("Skipping already completed index: %s", index_name)
                continue

            logger.info("Starting historical export for index: %s", index_name)
            pit_id = self.client.create_point_in_time(index_name, keep_alive=self.keep_alive)

            # Determine initial search_after cursor
            search_after = None
            if checkpoint.current_index == index_name and checkpoint.last_sort:
                search_after = checkpoint.last_sort
                logger.info("Resuming index '%s' after sort cursor: %s", index_name, search_after)

            try:
                while True:
                    hits = self.client.search_page(
                        pit_id=pit_id,
                        page_size=self.page_size,
                        search_after=search_after,
                        keep_alive=self.keep_alive,
                    )

                    if not hits:
                        # Index exhausted
                        checkpoint.mark_index_completed(index_name)
                        self.checkpoint_manager.save(checkpoint)
                        logger.info("Completed historical export for index: %s", index_name)
                        break

                    for hit in hits:
                        canonical_alert = canonicalize_wazuh_alert(hit)
                        yield canonical_alert

                    # Update cursor to last hit's sort array
                    last_hit = hits[-1]
                    last_sort = last_hit.get("sort")
                    last_id = (
                        last_hit.get("_source", {}).get("id")
                        or last_hit.get("id")
                        or last_hit.get("_id")
                    )

                    if last_sort and last_id:
                        search_after = last_sort
                        checkpoint.current_index = index_name
                        checkpoint.last_sort = list(last_sort)
                        checkpoint.last_wazuh_alert_id = str(last_id)
                        checkpoint.processed_count += len(hits)
                        self.checkpoint_manager.save(checkpoint)

            finally:
                self.client.close_point_in_time(pit_id)
