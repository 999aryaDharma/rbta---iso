"""Unit tests for WazuhIndexerLivePoller with overlap and deduplication (Sprint 7)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.wazuh_client import WazuhIndexerClient
from src.runtime.live_source import WazuhIndexerLivePoller


def make_hit(idx: int, ts_str: str = "2026-08-28T10:00:00.000+0000") -> dict:
    return {
        "_index": "wazuh-alerts-4.x-2026.08.28",
        "_id": f"doc_{idx}",
        "sort": [1775118765000 + idx, f"alert_{idx}"],
        "_source": {
            "id": f"alert_{idx}",
            "timestamp": ts_str,
            "agent": {"id": "001", "name": "soc-1"},
            "rule": {"id": "5501", "level": 3, "groups": ["pam"]},
            "rule_group_primary": "pam",
            "agent_criticality": 1,
        },
    }


def test_live_poller_queries_overlap_range_and_deduplicates():
    """Live poller queries time window [now - overlap, now] and filters out previously seen alert IDs."""
    client = MagicMock(spec=WazuhIndexerClient)
    client.list_indices.return_value = ["wazuh-alerts-4.x-2026.08.28"]

    # First poll returns alerts 1, 2
    # Second poll (with overlap) returns alerts 2, 3
    client._request.side_effect = [
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(2), make_hit(3)]}}),
    ]

    poller = WazuhIndexerLivePoller(
        client=client,
        overlap_window=timedelta(minutes=5),
        poll_interval=timedelta(seconds=5),
    )

    # First poll cycle
    alerts1 = poller.poll_once()
    assert len(alerts1) == 2
    assert [a.wazuh_alert_id for a in alerts1] == ["alert_1", "alert_2"]

    # Second poll cycle (alert_2 is duplicate in overlap window)
    alerts2 = poller.poll_once()
    assert len(alerts2) == 1
    assert [a.wazuh_alert_id for a in alerts2] == ["alert_3"]
