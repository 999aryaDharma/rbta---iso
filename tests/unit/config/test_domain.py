"""Unit tests for centralized research domain constants."""
import pytest
from src.config.domain import (
    AGENT_CRITICALITY,
    CRITICAL_MITRE_TACTICS,
    DEFAULT_AGENT_CRITICALITY,
    DEFAULT_RULE_GROUP_WEIGHT,
    GROUP_SEVERITY_WEIGHT,
    get_agent_criticality,
    has_critical_mitre_tactic,
    resolve_primary_rule_group,
)


def test_agent_criticality_mapping():
    """Test that known agent names map to authoritative 1-4 criticality scores."""
    assert get_agent_criticality("dfir-iris") == 4
    assert get_agent_criticality("pusatkarir") == 3
    assert get_agent_criticality("proxy-manager") == 3
    assert get_agent_criticality("sads") == 3
    assert get_agent_criticality("siput") == 2
    assert get_agent_criticality("e-kuesioner") == 2
    assert get_agent_criticality("soc-1") == 1
    assert get_agent_criticality("dvwa") == 1


def test_agent_criticality_case_insensitivity_and_whitespace():
    """Test case insensitivity and whitespace handling in agent criticality lookup."""
    assert get_agent_criticality("  DFIR-IRIS  ") == 4
    assert get_agent_criticality("PusatKarir") == 3
    assert get_agent_criticality("DVWA") == 1


def test_agent_criticality_unknown_default():
    """Test unknown agent defaults to 1 (Low) without crashing."""
    assert get_agent_criticality("unknown-server") == DEFAULT_AGENT_CRITICALITY
    assert get_agent_criticality(None) == DEFAULT_AGENT_CRITICALITY
    assert get_agent_criticality("") == DEFAULT_AGENT_CRITICALITY


def test_group_severity_weight_values():
    """Test that high-priority groups have higher weight than low-priority groups."""
    assert GROUP_SEVERITY_WEIGHT["attack"] == 10
    assert GROUP_SEVERITY_WEIGHT["sql_injection"] == 10
    assert GROUP_SEVERITY_WEIGHT["authentication_failed"] == 9
    assert GROUP_SEVERITY_WEIGHT["ossec"] == 1
    assert GROUP_SEVERITY_WEIGHT["attack"] > GROUP_SEVERITY_WEIGHT["syslog"]


def test_resolve_primary_rule_group_single():
    """Test single rule group resolution."""
    assert resolve_primary_rule_group(["syslog"]) == "syslog"
    assert resolve_primary_rule_group(["attack"]) == "attack"


def test_resolve_primary_rule_group_multiple():
    """Test multiple rule groups selects the highest weight."""
    groups = ["pam", "syslog", "authentication_success"]
    # pam (7) > authentication_success (3) == syslog (3)
    assert resolve_primary_rule_group(groups) == "pam"

    groups2 = ["web", "accesslog", "sql_injection", "attack"]
    # sql_injection (10) appears before attack (10), both are 10
    assert resolve_primary_rule_group(groups2) in ("sql_injection", "attack")
    # deterministic: first highest seen
    assert resolve_primary_rule_group(groups2) == "sql_injection"


def test_resolve_primary_rule_group_empty_or_unknown():
    """Test empty or unknown rule group list handling."""
    assert resolve_primary_rule_group([]) == "unknown"
    assert resolve_primary_rule_group(["custom_unlisted_group"]) == "custom_unlisted_group"


def test_critical_mitre_tactics_membership():
    """Test the authoritative 6 critical MITRE tactics."""
    expected = {
        "Execution",
        "Lateral Movement",
        "Credential Access",
        "Exfiltration",
        "Privilege Escalation",
        "Defense Evasion",
    }
    assert CRITICAL_MITRE_TACTICS == expected


def test_has_critical_mitre_tactic():
    """Test detection of critical MITRE tactics in a list."""
    assert has_critical_mitre_tactic(["Defense Evasion", "Initial Access"]) is True
    assert has_critical_mitre_tactic(["Initial Access", "Reconnaissance"]) is False
    assert has_critical_mitre_tactic([]) is False
    assert has_critical_mitre_tactic(["credential access"]) is True  # case insensitive
