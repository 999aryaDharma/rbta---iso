"""Unit tests for Wazuh alert canonicalizer."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import (
    CanonicalizationError,
    canonicalize_wazuh_alert,
    parse_wazuh_timestamp,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "wazuh"


def load_fixture(filename: str) -> dict:
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def test_parse_wazuh_timestamp_iso_formats():
    """Test parsing various valid Wazuh timestamp representations into timezone-aware UTC datetime."""
    # With offset +0000
    dt1 = parse_wazuh_timestamp("2026-08-28T05:38:45.712+0000")
    assert dt1.tzinfo is not None
    assert dt1.year == 2026 and dt1.month == 8 and dt1.day == 28
    assert dt1.hour == 5 and dt1.minute == 38 and dt1.second == 45
    assert dt1.microsecond == 712000

    # With Z
    dt2 = parse_wazuh_timestamp("2026-08-28T05:38:45.712Z")
    assert dt2.tzinfo is not None
    assert dt2 == dt1

    # With colon offset +08:00 (converted to UTC)
    dt3 = parse_wazuh_timestamp("2026-08-28T13:38:45.712+08:00")
    assert dt3 == dt1

    # Epoch timestamp in seconds / milliseconds
    dt4 = parse_wazuh_timestamp(1787895525.712)
    assert dt4.year == 2026


def test_parse_wazuh_timestamp_invalid():
    """Test that invalid timestamp formats raise CanonicalizationError."""
    with pytest.raises(CanonicalizationError, match="timestamp"):
        parse_wazuh_timestamp("invalid-date-string")

    with pytest.raises(CanonicalizationError, match="timestamp"):
        parse_wazuh_timestamp(None)


def test_canonicalize_standard_raw_alert():
    """Test canonicalization of standard raw Wazuh alert fixture."""
    data = load_fixture("raw_alert_standard.json")
    alert = canonicalize_wazuh_alert(data)

    assert isinstance(alert, CanonicalRawAlert)
    assert alert.wazuh_alert_id == "1787895525.48425"
    assert alert.agent_id == "000"
    assert alert.agent_name == "wazuh-soc"
    assert alert.rule_id == "5501"
    assert alert.rule_level == 3
    assert alert.rule_group_primary == "pam"
    assert "Defense Evasion" in alert.mitre_tactics
    assert "Initial Access" in alert.mitre_tactics
    assert alert.agent_criticality == 1
    assert alert.metadata["location"] == "journald"


def test_canonicalize_opensearch_hit_envelope():
    """Test canonicalization of OpenSearch search hit envelope unwrapping."""
    data = load_fixture("opensearch_hit.json")
    alert = canonicalize_wazuh_alert(data)

    assert alert.wazuh_alert_id == "1787895526.48426"  # from _source.id
    assert alert.agent_id == "001"
    assert alert.agent_name == "soc-1"
    assert alert.rule_id == "5710"
    assert alert.rule_level == 7
    assert alert.rule_group_primary == "authentication_failed"
    assert alert.mitre_tactics == ("Credential Access",)
    assert alert.srcip == "192.168.1.50"
    assert alert.agent_criticality == 1

    # Metadata preserves envelope
    assert alert.metadata["source_index"] == "wazuh-alerts-4.x-2026.08.28"
    assert alert.metadata["source_document_id"] == "doc-hit-sample-001"
    assert alert.metadata["source_sort"] == [1787895526000, "1787895526.48426"]


def test_canonicalize_alert_no_mitre():
    """Test canonicalization of alert without MITRE section."""
    data = load_fixture("raw_alert_no_mitre.json")
    alert = canonicalize_wazuh_alert(data)

    assert alert.wazuh_alert_id == "1787897412.1001"
    assert alert.agent_name == "pusatkarir"
    assert alert.agent_criticality == 3  # pusatkarir is High criticality (3)
    assert alert.rule_group_primary == "syslog"  # syslog > ossec
    assert alert.mitre_tactics == ()
    assert alert.srcip is None


def test_canonicalize_alert_flattened_mitre():
    """Test canonicalization of alert with flattened MITRE tactics."""
    data = load_fixture("raw_alert_flattened_mitre.json")
    alert = canonicalize_wazuh_alert(data)

    assert alert.wazuh_alert_id == "1787901330.2002"
    assert alert.agent_name == "dfir-iris"
    assert alert.agent_criticality == 4  # dfir-iris is Critical (4)
    assert alert.rule_group_primary in ("sql_injection", "attack")
    assert alert.rule_group_primary == "sql_injection"  # priority tie-break
    assert "Initial Access" in alert.mitre_tactics
    assert "Defense Evasion" in alert.mitre_tactics
    assert alert.srcip == "203.0.113.19"


def test_canonicalize_parity_raw_vs_opensearch_hit():
    """Test that raw source and OpenSearch hit containing identical _source produce identical core domain fields."""
    hit = load_fixture("opensearch_hit.json")
    raw = hit["_source"]

    alert_from_hit = canonicalize_wazuh_alert(hit)
    alert_from_raw = canonicalize_wazuh_alert(raw)

    assert alert_from_hit.wazuh_alert_id == alert_from_raw.wazuh_alert_id
    assert alert_from_hit.timestamp == alert_from_raw.timestamp
    assert alert_from_hit.agent_id == alert_from_raw.agent_id
    assert alert_from_hit.agent_name == alert_from_raw.agent_name
    assert alert_from_hit.rule_group_primary == alert_from_raw.rule_group_primary
    assert alert_from_hit.rule_level == alert_from_raw.rule_level
    assert alert_from_hit.rule_id == alert_from_raw.rule_id
    assert alert_from_hit.mitre_tactics == alert_from_raw.mitre_tactics
    assert alert_from_hit.srcip == alert_from_raw.srcip
    assert alert_from_hit.agent_criticality == alert_from_raw.agent_criticality


def test_canonicalize_missing_mandatory_fields_fails():
    """Test that missing mandatory fields raise CanonicalizationError."""
    # Missing ID
    with pytest.raises(CanonicalizationError, match="id"):
        canonicalize_wazuh_alert({
            "timestamp": "2026-08-28T05:00:00Z",
            "rule": {"id": "100", "level": 3, "groups": ["syslog"]},
        })

    # Missing Rule ID
    with pytest.raises(CanonicalizationError, match="rule.id"):
        canonicalize_wazuh_alert({
            "id": "123.456",
            "timestamp": "2026-08-28T05:00:00Z",
            "rule": {"level": 3, "groups": ["syslog"]},
        })

    # Missing Timestamp
    with pytest.raises(CanonicalizationError, match="timestamp"):
        canonicalize_wazuh_alert({
            "id": "123.456",
            "rule": {"id": "100", "level": 3, "groups": ["syslog"]},
        })


def test_canonicalize_deterministic_output():
    """Test that repeated canonicalization on same object returns equal CanonicalRawAlert."""
    data = load_fixture("raw_alert_standard.json")
    a1 = canonicalize_wazuh_alert(data)
    a2 = canonicalize_wazuh_alert(data)
    assert a1 == a2
