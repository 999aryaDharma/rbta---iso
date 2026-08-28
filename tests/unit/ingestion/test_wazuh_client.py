"""Unit tests for WazuhIndexerClient (Sprint 5)."""
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ingestion.wazuh_client import WazuhClientError, WazuhIndexerClient, WazuhAuthError


def test_wazuh_client_init_and_secure_tls_defaults():
    """Client defaults to verify_tls=True and requires explicit url."""
    client = WazuhIndexerClient(
        base_url="https://172.16.83.180:9200",
        username="admin",
        password="secretpassword",
        verify_tls=True,
    )
    assert client.base_url == "https://172.16.83.180:9200"
    assert client.verify_tls is True
    assert client.timeout == (5.0, 30.0)


def test_wazuh_client_401_403_fails_fast_without_retry():
    """HTTP 401/403 status codes raise WazuhAuthError immediately without retrying."""
    client = WazuhIndexerClient(base_url="https://wazuh-indexer:9200", username="user", password="bad")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
        with pytest.raises(WazuhAuthError, match="Authentication failed"):
            client.create_point_in_time("wazuh-alerts-4.x-2026.04.02")

        assert mock_req.call_count == 1  # No retry on 401


def test_wazuh_client_pit_creation_and_close():
    """Client creates Point-In-Time and closes it properly."""
    client = WazuhIndexerClient(base_url="https://wazuh-indexer:9200")

    # Mock create PIT response
    mock_pit_resp = MagicMock()
    mock_pit_resp.status_code = 200
    mock_pit_resp.json.return_value = {"pit_id": "pit_xyz123", "_shards": {"total": 5, "successful": 5, "failed": 0}}

    # Mock close PIT response
    mock_close_resp = MagicMock()
    mock_close_resp.status_code = 200
    mock_close_resp.json.return_value = {"succeeded": True, "num_freed": 1}

    with patch.object(client._session, "request", side_effect=[mock_pit_resp, mock_close_resp]):
        pit_id = client.create_point_in_time("wazuh-alerts-4.x-2026.04.02")
        assert pit_id == "pit_xyz123"

        closed = client.close_point_in_time(pit_id)
        assert closed is True


def test_wazuh_client_partial_pit_rejected():
    """If PIT response contains failed shards, client raises WazuhClientError."""
    client = WazuhIndexerClient(base_url="https://wazuh-indexer:9200")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"pit_id": "pit_partial", "_shards": {"total": 5, "successful": 4, "failed": 1}}

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(WazuhClientError, match="Partial PIT creation rejected"):
            client.create_point_in_time("wazuh-alerts-4.x-2026.04.02")
