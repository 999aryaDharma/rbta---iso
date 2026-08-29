"""Unit tests for CanonicalRawAlert DTO contract."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import pytest
from src.contracts.raw_alert import CanonicalRawAlert


def test_canonical_raw_alert_valid_instantiation():
    """Test valid instantiation with all required and optional fields."""
    now = datetime.now(timezone.utc)
    alert = CanonicalRawAlert(
        wazuh_alert_id="1787895525.48425",
        timestamp=now,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=3,
        rule_id="5501",
        mitre_tactics=("Defense Evasion", "Initial Access"),
        srcip="192.168.1.50",
        agent_criticality=1,
        metadata={"location": "journald", "source_document_id": "doc123"},
    )
    assert alert.wazuh_alert_id == "1787895525.48425"
    assert alert.timestamp == now
    assert alert.agent_id == "001"
    assert alert.agent_name == "soc-1"
    assert alert.rule_group_primary == "pam"
    assert alert.rule_level == 3
    assert alert.rule_id == "5501"
    assert alert.mitre_tactics == ("Defense Evasion", "Initial Access")
    assert alert.srcip == "192.168.1.50"
    assert alert.agent_criticality == 1
    assert alert.metadata["location"] == "journald"
    assert alert.metadata["source_document_id"] == "doc123"


def test_canonical_raw_alert_immutability():
    """Test that CanonicalRawAlert attributes cannot be modified after creation."""
    now = datetime.now(timezone.utc)
    alert = CanonicalRawAlert(
        wazuh_alert_id="1787895525.48425",
        timestamp=now,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=3,
        rule_id="5501",
        mitre_tactics=("Defense Evasion",),
        srcip=None,
        agent_criticality=1,
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        alert.rule_level = 5


def test_canonical_raw_alert_metadata_immutability():
    """Test that CanonicalRawAlert metadata mapping cannot be mutated in place (FIX 7)."""
    now = datetime.now(timezone.utc)
    alert = CanonicalRawAlert(
        wazuh_alert_id="1787895525.48425",
        timestamp=now,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=3,
        rule_id="5501",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
        metadata={"initial_key": "initial_value"},
    )
    with pytest.raises(TypeError):
        alert.metadata["initial_key"] = "mutated_value"

    with pytest.raises(TypeError):
        alert.metadata["new_key"] = "new_value"


def test_canonical_raw_alert_timezone_awareness_enforced():
    """Test that naive datetimes are rejected with ValueError."""
    naive_dt = datetime(2026, 8, 28, 5, 38, 45)
    with pytest.raises(ValueError, match="timezone-aware"):
        CanonicalRawAlert(
            wazuh_alert_id="1787895525.48425",
            timestamp=naive_dt,
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=3,
            rule_id="5501",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
        )


def test_canonical_raw_alert_rule_level_range_validation():
    """Test that rule_level must be within Wazuh standard 0..15 range."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="rule_level"):
        CanonicalRawAlert(
            wazuh_alert_id="1787895525.48425",
            timestamp=now,
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=16,
            rule_id="5501",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
        )

    with pytest.raises(ValueError, match="rule_level"):
        CanonicalRawAlert(
            wazuh_alert_id="1787895525.48425",
            timestamp=now,
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=-1,
            rule_id="5501",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
        )


def test_canonical_raw_alert_criticality_range_validation():
    """Test that agent_criticality must be within 1..4 range."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="agent_criticality"):
        CanonicalRawAlert(
            wazuh_alert_id="1787895525.48425",
            timestamp=now,
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=3,
            rule_id="5501",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=5,
        )


def test_canonical_raw_alert_wazuh_alert_id_required():
    """Test that wazuh_alert_id must not be empty."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="wazuh_alert_id"):
        CanonicalRawAlert(
            wazuh_alert_id="",
            timestamp=now,
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=3,
            rule_id="5501",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
        )
