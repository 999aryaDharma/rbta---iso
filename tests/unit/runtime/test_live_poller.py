"""Unit tests for WazuhIndexerLivePoller with fast recent poll, reconciliation scan, and exact index derivation."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.wazuh_client import WazuhIndexerClient
from src.runtime.live_source import WazuhIndexerLivePoller, derive_daily_indices


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


def test_derive_daily_indices_midnight_spanning():
    """Deriving daily indices across midnight spans exactly the two UTC dates without wildcard."""
    t_start = datetime(2026, 8, 28, 23, 55, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 8, 29, 0, 5, 0, tzinfo=timezone.utc)

    indices = derive_daily_indices(t_start, t_end)
    assert indices == ["wazuh-alerts-4.x-2026.08.28", "wazuh-alerts-4.x-2026.08.29"]
    assert "wazuh-alerts-*" not in indices


def test_live_poller_queries_overlap_range_with_recent_cursor():
    """Fast recent poll queries time window [cursor - overlap, current_time] without dropping duplicates."""
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

    t_cursor = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    t_now = datetime(2026, 8, 28, 10, 5, 0, tzinfo=timezone.utc)

    alerts = poller.poll_recent(current_time=t_now, recent_poll_cursor=t_cursor)
    assert len(alerts) == 2
    assert [a.wazuh_alert_id for a in alerts] == ["alert_1", "alert_2"]

    # Verify query body used cursor - overlap (09:55:00) to now (10:05:00)
    args, kwargs = client._request.call_args
    range_query = kwargs["json_data"]["query"]["range"]["@timestamp"]
    assert range_query["gte"] == "2026-08-28T09:55:00+00:00"
    assert range_query["lte"] == "2026-08-28T10:05:00+00:00"


def test_live_poller_midnight_spanning_query():
    """Live poller targets exact daily indices when window crosses UTC midnight (not wildcard)."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": []}},
    )

    poller = WazuhIndexerLivePoller(client=client, overlap_window=timedelta(minutes=10))

    t_start = datetime(2026, 8, 28, 23, 55, 0, tzinfo=timezone.utc)
    t_now = datetime(2026, 8, 29, 0, 5, 0, tzinfo=timezone.utc)

    poller.poll_recent(current_time=t_now, recent_poll_cursor=t_start)

    args, kwargs = client._request.call_args
    endpoint = args[1]
    # Must explicitly target the two daily indices, not a blind wildcard
    assert endpoint == "/wazuh-alerts-4.x-2026.08.28,wazuh-alerts-4.x-2026.08.29/_search"
    assert "wazuh-alerts-*" not in endpoint


def test_live_poller_reconciliation_scan_queries_retained_days():
    """Reconciliation scan queries exact retained daily indices across configured days."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [make_hit(100, "2026-08-28T02:00:00.000+0000")]}},
    )

    poller = WazuhIndexerLivePoller(client=client)
    t_now = datetime(2026, 8, 29, 14, 0, 0, tzinfo=timezone.utc)

    recon_alerts = poller.poll_reconciliation(current_time=t_now, reconciliation_days=2)
    assert len(recon_alerts) == 1
    assert recon_alerts[0].wazuh_alert_id == "alert_100"

    args, kwargs = client._request.call_args
    endpoint = args[1]
    assert endpoint == "/wazuh-alerts-4.x-2026.08.28,wazuh-alerts-4.x-2026.08.29/_search"


def test_live_poller_pagination():
    """Live poller requests pages with search_after until result length < page_size."""
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

    alerts = poller.poll_recent(current_time=datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc))
    assert len(alerts) == 5
    assert [a.wazuh_alert_id for a in alerts] == ["alert_1", "alert_2", "alert_3", "alert_4", "alert_5"]
    assert client._request.call_count == 3


def test_live_poller_no_permanent_seen_registry():
    """Poller does not swallow duplicate IDs on successive calls (core owns dedup)."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.side_effect = [
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2), make_hit(3)]}}),
    ]

    poller = WazuhIndexerLivePoller(client=client)
    p1 = poller.poll_recent(current_time=datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc))
    assert len(p1) == 2

    p2 = poller.poll_recent(current_time=datetime(2026, 8, 28, 10, 1, 0, tzinfo=timezone.utc))
    assert len(p2) == 3  # alerts 1 and 2 are returned again without being dropped by poller


def test_live_poller_propagates_transport_error():
    """Live poller does not silently swallow network exceptions."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.side_effect = Exception("Connection Refused")

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(Exception, match="Connection Refused"):
        poller.poll_recent()
