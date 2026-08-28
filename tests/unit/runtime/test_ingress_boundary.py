"""Unit tests for CollectorIngressBoundary (Sprint 7)."""
from datetime import datetime, timezone
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.ingress import CollectorIngressBoundary, IngressPayloadError


def make_valid_raw_event(alert_id: str) -> dict:
    return {
        "id": alert_id,
        "timestamp": "2026-08-28T10:00:00.000+0000",
        "agent": {"id": "001", "name": "soc-1"},
        "rule": {"id": "5501", "level": 3, "groups": ["pam"]},
    }


def test_ingress_boundary_accepts_valid_payload_and_detects_duplicates():
    """Ingress accepts valid payload and flags duplicate without failing."""
    ingress = CollectorIngressBoundary(api_key="secret-key")

    # Valid payload with auth
    event = make_valid_raw_event("alert_100")
    res1 = ingress.process_incoming(event, auth_header="Bearer secret-key")
    assert res1.status == "accepted"
    assert res1.is_duplicate is False
    assert isinstance(res1.canonical_alert, CanonicalRawAlert)
    assert res1.canonical_alert.wazuh_alert_id == "alert_100"

    # Same alert sent again -> accepted as duplicate
    res2 = ingress.process_incoming(event, auth_header="Bearer secret-key")
    assert res2.status == "accepted"
    assert res2.is_duplicate is True
    assert res2.canonical_alert is None


def test_ingress_boundary_rejects_unauthorized():
    """Ingress rejects unauthorized request."""
    ingress = CollectorIngressBoundary(api_key="secret-key")
    event = make_valid_raw_event("alert_100")

    with pytest.raises(IngressPayloadError, match="Unauthorized"):
        ingress.process_incoming(event, auth_header="Bearer bad-key")


def test_ingress_boundary_rejects_malformed_schema():
    """Ingress rejects malformed payload with clear IngressPayloadError."""
    ingress = CollectorIngressBoundary(api_key="secret-key")

    bad_event = {"agent": {"id": "001"}}  # Missing id, timestamp, rule
    with pytest.raises(IngressPayloadError, match="Malformed raw alert"):
        ingress.process_incoming(bad_event, auth_header="Bearer secret-key")
