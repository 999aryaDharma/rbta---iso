"""Unit tests for Wazuh alert canonicalizer."""
import json
from datetime import datetime, timezone, timedelta
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


# ── Timestamp Tests (FIX 6) ───────────────────────────────────────────────────

def test_parse_wazuh_timestamp_iso_formats():
    """Test parsing various valid timezone-aware Wazuh timestamps into UTC."""
    # With offset +0000
    dt1 = parse_wazuh_timestamp("2026-08-28T05:38:45.712+0000")
    assert dt1.tzinfo is not None
    assert dt1.utcoffset() == timedelta(0)
    assert dt1.year == 2026 and dt1.month == 8 and dt1.day == 28
    assert dt1.hour == 5 and dt1.minute == 38 and dt1.second == 45
    assert dt1.microsecond == 712000

    # With Z
    dt2 = parse_wazuh_timestamp("2026-08-28T05:38:45.712Z")
    assert dt2.tzinfo is not None
    assert dt2 == dt1

    # With explicit offset +08:00 (converted to UTC)
    dt3 = parse_wazuh_timestamp("2026-08-28T13:38:45.712+08:00")
    assert dt3.tzinfo is not None
    assert dt3 == dt1

    # Timezone-aware datetime object
    aware_dt = datetime(2026, 8, 28, 5, 38, 45, 712000, tzinfo=timezone.utc)
    assert parse_wazuh_timestamp(aware_dt) == aware_dt

    # Epoch in seconds / milliseconds
    dt_epoch = parse_wazuh_timestamp(1787895525.712)
    assert dt_epoch.tzinfo is not None


def test_parse_wazuh_timestamp_rejects_naive():
    """Test that naive timestamps (strings without offset and naive datetime objects) are strictly rejected."""
    # Naive ISO string without offset or Z
    with pytest.raises(CanonicalizationError, match="naive|offset"):
        parse_wazuh_timestamp("2026-08-28T05:38:45.712")

    with pytest.raises(CanonicalizationError, match="naive|offset"):
        parse_wazuh_timestamp("2026-08-28 05:38:45")

    # Naive datetime object
    naive_dt = datetime(2026, 8, 28, 5, 38, 45)
    with pytest.raises(CanonicalizationError, match="naive|timezone-aware"):
        parse_wazuh_timestamp(naive_dt)


def test_parse_wazuh_timestamp_invalid():
    """Test that invalid timestamp formats raise CanonicalizationError."""
    with pytest.raises(CanonicalizationError, match="[Tt]imestamp|naive"):
        parse_wazuh_timestamp("invalid-date-string")

    with pytest.raises(CanonicalizationError, match="[Tt]imestamp"):
        parse_wazuh_timestamp(None)

    with pytest.raises(CanonicalizationError, match="[Tt]imestamp"):
        parse_wazuh_timestamp("")


# ── Rule Level Validation Tests (FIX 4) ───────────────────────────────────────

def test_canonicalize_rule_level_validation():
    """Test strict validation of rule.level in range 0..15."""
    base_alert = {
        "id": "1001",
        "timestamp": "2026-08-28T05:00:00Z",
        "rule": {"id": "500", "groups": ["syslog"]},
    }

    # Missing rule.level
    with pytest.raises(CanonicalizationError, match="rule.level"):
        canonicalize_wazuh_alert(base_alert)

    # Null rule.level
    with pytest.raises(CanonicalizationError, match="rule.level"):
        canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": None, "groups": ["syslog"]}})

    # Non-numeric rule.level
    with pytest.raises(CanonicalizationError, match="rule.level"):
        canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": "high", "groups": ["syslog"]}})

    # Negative rule.level
    with pytest.raises(CanonicalizationError, match="Rule level"):
        canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": -1, "groups": ["syslog"]}})

    # Out-of-range rule.level
    with pytest.raises(CanonicalizationError, match="Rule level"):
        canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 16, "groups": ["syslog"]}})

    # Valid boundary 0
    a0 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 0, "groups": ["syslog"]}})
    assert a0.rule_level == 0

    # Valid boundary 15
    a15 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 15, "groups": ["syslog"]}})
    assert a15.rule_level == 15


# ── Agent Normalization Tests (FIX 5) ─────────────────────────────────────────

def test_canonicalize_agent_null_and_fallback_normalization():
    """Test that null, missing, or whitespace agent fields normalize cleanly to '000' and 'unknown' without creating 'None' strings."""
    base_alert = {
        "id": "1001",
        "timestamp": "2026-08-28T05:00:00Z",
        "rule": {"id": "500", "level": 3, "groups": ["syslog"]},
    }

    # 1. Missing agent block
    a1 = canonicalize_wazuh_alert(base_alert)
    assert a1.agent_id == "000"
    assert a1.agent_name == "unknown"

    # 2. Null agent fields
    a2 = canonicalize_wazuh_alert({**base_alert, "agent": {"id": None, "name": None}})
    assert a2.agent_id == "000"
    assert a2.agent_name == "unknown"
    assert "None" not in (a2.agent_id, a2.agent_name)

    # 3. Empty/whitespace agent fields
    a3 = canonicalize_wazuh_alert({**base_alert, "agent": {"id": "   ", "name": "   "}})
    assert a3.agent_id == "000"
    assert a3.agent_name == "unknown"

    # 4. Fallback to manager name when agent name is absent
    a4 = canonicalize_wazuh_alert({
        **base_alert,
        "agent": {"id": "000", "name": None},
        "manager": {"name": "wazuh-soc-master"},
    })
    assert a4.agent_id == "000"
    assert a4.agent_name == "wazuh-soc-master"

    # 5. Normal valid agent preserved
    a5 = canonicalize_wazuh_alert({**base_alert, "agent": {"id": "001", "name": "soc-1"}})
    assert a5.agent_id == "001"
    assert a5.agent_name == "soc-1"


# ── MITRE Deduplication & Case-Insensitive Normalization Tests (FIX 3) ────────

def test_canonicalize_mitre_case_insensitive_deduplication():
    """Test that MITRE tactics are deduplicated case-insensitively, preserving first valid casing (FIX 3)."""
    alert_data = {
        "id": "1001",
        "timestamp": "2026-08-28T05:00:00Z",
        "rule": {
            "id": "5501",
            "level": 5,
            "groups": ["pam"],
            "mitre": {
                "tactic": ["Execution", "execution", " EXECUTION "],
            },
            "mitre_tactics": ["defense evasion", "Defense Evasion"],
        },
    }
    alert = canonicalize_wazuh_alert(alert_data)
    assert alert.mitre_tactics == ("Execution", "defense evasion")


# ── Rule Groups Normalization Tests (FIX 4) ───────────────────────────────────

def test_canonicalize_rule_groups_malformed_normalization():
    """Test that malformed rule groups (None, empty, null, none) are filtered out safely (FIX 4)."""
    base_alert = {
        "id": "1001",
        "timestamp": "2026-08-28T05:00:00Z",
        "rule": {"id": "500", "level": 3},
    }

    # 1. Missing groups
    a1 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 3}})
    assert a1.rule_group_primary == "unknown"
    assert a1.metadata["rule_groups_all"] == ()

    # 2. Empty list of groups
    a2 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 3, "groups": []}})
    assert a2.rule_group_primary == "unknown"
    assert a2.metadata["rule_groups_all"] == ()

    # 3. List of None and empty strings
    a3 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 3, "groups": [None, "", "   "]}})
    assert a3.rule_group_primary == "unknown"
    assert a3.metadata["rule_groups_all"] == ()

    # 4. List containing 'none', 'null' string literals
    a4 = canonicalize_wazuh_alert({**base_alert, "rule": {"id": "500", "level": 3, "groups": ["none", "null"]}})
    assert a4.rule_group_primary == "unknown"
    assert a4.metadata["rule_groups_all"] == ()

    # 5. Mixed valid groups with whitespace and casing variants
    a5 = canonicalize_wazuh_alert({
        **base_alert,
        "rule": {"id": "500", "level": 3, "groups": [None, " Web ", "web", "syslog"]},
    })
    # 'web' has higher weight than 'syslog' (7 vs 3)
    assert a5.rule_group_primary == "web"
    assert a5.metadata["rule_groups_all"] == ("web", "syslog")


# ── OpenSearch Envelope and Traceability Tests ────────────────────────────────

def test_canonicalize_opensearch_hit_envelope_alert_id_traceability():
    """Test that wazuh_alert_id strictly comes from _source.id and NOT from OpenSearch _id."""
    hit = {
        "_index": "wazuh-alerts-4.x-2026.08.28",
        "_id": "opensearch-doc-id-xyz",
        "_source": {
            "id": "wazuh-alert-123.456",
            "timestamp": "2026-08-28T05:00:00Z",
            "rule": {"id": "500", "level": 4, "groups": ["syslog"]},
            "agent": {"id": "001", "name": "soc-1"},
        },
        "sort": [1787895526000, "wazuh-alert-123.456"],
    }
    alert = canonicalize_wazuh_alert(hit)
    assert alert.wazuh_alert_id == "wazuh-alert-123.456"
    assert alert.wazuh_alert_id != "opensearch-doc-id-xyz"
    assert alert.metadata["source_document_id"] == "opensearch-doc-id-xyz"
    assert alert.metadata["source_index"] == "wazuh-alerts-4.x-2026.08.28"
    assert alert.metadata["source_sort"] == (1787895526000, "wazuh-alert-123.456")


def test_canonicalize_missing_source_id_fails_fast():
    """Test that OpenSearch hit with missing _source.id fails fast even if _id is present."""
    hit = {
        "_index": "wazuh-alerts-4.x-2026.08.28",
        "_id": "opensearch-doc-id-xyz",
        "_source": {
            # missing "id"
            "timestamp": "2026-08-28T05:00:00Z",
            "rule": {"id": "500", "level": 4, "groups": ["syslog"]},
        },
    }
    with pytest.raises(CanonicalizationError, match="id"):
        canonicalize_wazuh_alert(hit)


# ── Fixtures & Parity Tests ───────────────────────────────────────────────────

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


def test_canonicalize_alert_no_mitre():
    """Test canonicalization of alert without MITRE section."""
    data = load_fixture("raw_alert_no_mitre.json")
    alert = canonicalize_wazuh_alert(data)

    assert alert.wazuh_alert_id == "1787897412.1001"
    assert alert.agent_name == "pusatkarir"
    assert alert.agent_criticality == 3
    assert alert.rule_group_primary == "syslog"
    assert alert.mitre_tactics == ()
    assert alert.srcip is None


def test_canonicalize_alert_flattened_mitre():
    """Test canonicalization of alert with flattened MITRE tactics."""
    data = load_fixture("raw_alert_flattened_mitre.json")
    alert = canonicalize_wazuh_alert(data)

    assert alert.wazuh_alert_id == "1787901330.2002"
    assert alert.agent_name == "dfir-iris"
    assert alert.agent_criticality == 4
    assert alert.rule_group_primary == "sql_injection"
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
    with pytest.raises(CanonicalizationError, match="[Tt]imestamp"):
        canonicalize_wazuh_alert({
            "id": "123.456",
            "rule": {"id": "100", "level": 3, "groups": ["syslog"]},
        })

    # Missing Rule block
    with pytest.raises(CanonicalizationError, match="rule"):
        canonicalize_wazuh_alert({
            "id": "123.456",
            "timestamp": "2026-08-28T05:00:00Z",
        })


def test_canonicalize_deterministic_output():
    """Test that repeated canonicalization on same object returns equal CanonicalRawAlert."""
    data = load_fixture("raw_alert_standard.json")
    a1 = canonicalize_wazuh_alert(data)
    a2 = canonicalize_wazuh_alert(data)
    assert a1 == a2
