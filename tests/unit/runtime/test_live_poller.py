"""Unit tests for WazuhIndexerLivePoller with fast poll, reconciliation, and fail-closed integrity."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.wazuh_client import WazuhIndexerClient
from src.runtime.live_source import (
    LiveCanonicalizationError,
    LiveSourceIntegrityError,
    WazuhIndexerLivePoller,
    derive_daily_indices,
)


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


def test_live_poller_canonicalization_failure_raises_fail_closed():
    """When a document fails canonicalization, poller raises LiveCanonicalizationError instead of silent continue."""
    client = MagicMock(spec=WazuhIndexerClient)
    malformed_hit = {
        "_index": "wazuh-alerts-4.x-2026.08.28",
        "_id": "doc_bad",
        "sort": [1775118765000, "alert_bad"],
        "_source": {
            "id": "alert_bad",
            # missing timestamp and agent
        },
    }
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [make_hit(1), malformed_hit, make_hit(2)]}},
    )

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(LiveCanonicalizationError) as exc_info:
        poller.poll_recent(current_time=datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc))

    assert "doc_bad" in str(exc_info.value)
    assert "wazuh-alerts-4.x-2026.08.28" in str(exc_info.value)


def test_live_poller_malformed_response_empty_dict_fails():
    """Empty dictionary response from Indexer raises LiveSourceIntegrityError."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(status_code=200, json=lambda: {})

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(LiveSourceIntegrityError, match="missing or invalid 'hits'"):
        poller.poll_recent()


def test_live_poller_malformed_response_hits_not_dict_fails():
    """Response with hits: None or non-dict raises LiveSourceIntegrityError."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(status_code=200, json=lambda: {"hits": None})

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(LiveSourceIntegrityError, match="missing or invalid 'hits'"):
        poller.poll_recent()


def test_live_poller_malformed_response_hits_hits_not_list_fails():
    """Response with hits.hits not a list raises LiveSourceIntegrityError."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(status_code=200, json=lambda: {"hits": {"hits": "not-a-list"}})

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(LiveSourceIntegrityError, match="missing or invalid 'hits.hits'"):
        poller.poll_recent()


def test_live_poller_valid_empty_hits_returns_empty_list():
    """Valid empty response returns empty list normally."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.return_value = MagicMock(status_code=200, json=lambda: {"hits": {"hits": []}})

    poller = WazuhIndexerLivePoller(client=client)
    alerts = poller.poll_recent()
    assert alerts == []


def test_live_poller_full_page_missing_sort_cursor_fails():
    """When a page has page_size items but missing 'sort' cursor, LiveSourceIntegrityError is raised."""
    client = MagicMock(spec=WazuhIndexerClient)
    hit1 = make_hit(1)
    hit2 = make_hit(2)
    del hit2["sort"]  # remove sort on final hit of full page

    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [hit1, hit2]}},
    )

    poller = WazuhIndexerLivePoller(client=client, page_size=2)

    with pytest.raises(LiveSourceIntegrityError, match="missing 'sort' field in final hit"):
        poller.poll_recent()


def test_live_poller_full_page_invalid_sort_cursor_fails():
    """When a page has page_size items but invalid 'sort' cursor, LiveSourceIntegrityError is raised."""
    client = MagicMock(spec=WazuhIndexerClient)
    hit1 = make_hit(1)
    hit2 = make_hit(2)
    hit2["sort"] = "not-a-sequence"

    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [hit1, hit2]}},
    )

    poller = WazuhIndexerLivePoller(client=client, page_size=2)

    with pytest.raises(LiveSourceIntegrityError, match="invalid 'sort' cursor"):
        poller.poll_recent()


def test_live_poller_discover_retained_daily_alert_indices():
    """Poller filters and sorts only valid Wazuh daily alert indices from Indexer list."""
    client = MagicMock(spec=WazuhIndexerClient)
    client.list_indices.return_value = [
        "wazuh-alerts-4.x-2026.08.29",
        ".kibana_1",
        "wazuh-alerts-4.x-2026.08.20",
        "wazuh-monitoring-2026.08.29",
        "wazuh-alerts-4.x-2026.08.25",
        "wazuh-alerts-4.x-invalid-name",
    ]

    poller = WazuhIndexerLivePoller(client=client)
    indices = poller.discover_retained_daily_alert_indices()

    assert indices == [
        "wazuh-alerts-4.x-2026.08.20",
        "wazuh-alerts-4.x-2026.08.25",
        "wazuh-alerts-4.x-2026.08.29",
    ]


def test_live_poller_full_reconciliation_scans_all_retained_indices():
    """Full-retention reconciliation scans every discovered daily index without timestamp cutoffs."""
    client = MagicMock(spec=WazuhIndexerClient)
    client.list_indices.return_value = [
        "wazuh-alerts-4.x-2026.08.20",
        "wazuh-alerts-4.x-2026.08.29",
    ]
    client._request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"hits": {"hits": [make_hit(100, "2026-08-20T05:00:00.000+0000")]}},
    )

    poller = WazuhIndexerLivePoller(client=client)
    alerts = poller.poll_full_reconciliation()

    assert len(alerts) == 1
    assert alerts[0].wazuh_alert_id == "alert_100"

    args, kwargs = client._request.call_args
    assert args[1] == "/wazuh-alerts-4.x-2026.08.20,wazuh-alerts-4.x-2026.08.29/_search"
    # No range filter in full retention query
    assert "query" not in kwargs["json_data"]


def test_live_poller_pagination_normal_3_pages():
    """Live poller requests pages with search_after until result length < page_size."""
    client = MagicMock(spec=WazuhIndexerClient)

    client._request.side_effect = [
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(1), make_hit(2)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(3), make_hit(4)]}}),
        MagicMock(status_code=200, json=lambda: {"hits": {"hits": [make_hit(5)]}}),
    ]

    poller = WazuhIndexerLivePoller(client=client, page_size=2)
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
    assert len(p2) == 3


def test_live_poller_propagates_transport_error():
    """Live poller does not silently swallow network exceptions."""
    client = MagicMock(spec=WazuhIndexerClient)
    client._request.side_effect = Exception("Connection Refused")

    poller = WazuhIndexerLivePoller(client=client)

    with pytest.raises(Exception, match="Connection Refused"):
        poller.poll_recent()
