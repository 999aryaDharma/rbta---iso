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


def test_live_poller_queries_overlap_range_with_high_watermark():
    """Live poller queries time window [watermark - overlap, current_time] without pre-commit deduplication."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2)]}},
    )

    poller = WazuhIndexerLivePoller(
        client=client,
        overlap_window=timedelta(minutes=5),
        poll_interval=timedelta(seconds=5),
    )

    t_watermark = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    t_now = datetime(2026, 8, 28, 10, 5, 0, tzinfo=timezone.utc)

    alerts = poller.poll_once(current_time=t_now, high_watermark=t_watermark)
    assert len(alerts) == 2
    assert [a.wazuh_alert_id for a in alerts] == ["alert_1", "alert_2"]

    # Verify query body used watermark - overlap (09:55:00) to now (10:05:00)
    args, kwargs = client._request.call_args
    range_query = kwargs["json_data"]["query"]["range"]["@timestamp"]
    assert range_query["gte"] == "2026-08-28T09:55:00+00:00"
    assert range_query["lte"] == "2026-08-28T10:05:00+00:00"


def test_live_poller_midnight_spanning_query():
    """Live poller derive daily indices covering window spanning UTC midnight."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": []}},
    )

    poller = WazuhIndexerLivePoller(client=client, overlap_window=timedelta(minutes=10))

    t_start = datetime(2026, 8, 28, 23, 55, 0, tzinfo=timezone.utc)
    t_now = datetime(2026, 8, 29, 0, 5, 0, tzinfo=timezone.utc)

    poller.poll_once(current_time=t_now, high_watermark=t_start)

    args, kwargs = client._request.call_args
    # Target endpoint should query both days or pattern covering both
    endpoint = args[1]
    assert "2026.08.28" in endpoint and "2026.08.29" in endpoint or "wazuh-alerts-*" in endpoint


def test_live_poller_pagination():
    """Live poller requests pages until result length < page_size."""
    client = MagicMock(spec=WazuhIndexerClient)
    
    # 5 alerts total, page_size = 2 -> 3 pages needed
    client._request.side_effect = [
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(3), make_hit(4)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(5)]}}),
    ]

    poller = WazuhIndexerLivePoller(
        client=client,
        page_size=2,
    )

    alerts = poller.poll_once()
    assert len(alerts) == 5
    assert [a.wazuh_alert_id for a in alerts] == ["alert_1", "alert_2", "alert_3", "alert_4", "alert_5"]
    assert client._request.call_count == 3


def test_live_poller_propagates_transport_error():
    """Live poller does not silently swallow network exceptions."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.side_effect = Exception("Connection Refused")

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(Exception, match="Connection Refused"):
        poller.poll_once()
