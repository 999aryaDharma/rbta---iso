"""Unit tests for recursive immutability utility and DTO defensive isolation."""
from types import MappingProxyType
import pytest

from src.contracts.immutability import freeze_value
from src.contracts.raw_alert import CanonicalRawAlert
from datetime import datetime, timezone


def test_freeze_value_primitives():
    """Primitives should remain unchanged."""
    assert freeze_value(1) == 1
    assert freeze_value("hello") == "hello"
    assert freeze_value(3.14) == 3.14
    assert freeze_value(True) is True
    assert freeze_value(None) is None


def test_freeze_value_recursive_dict_and_list():
    """Nested dicts and lists must become MappingProxyType and tuples recursively."""
    data = {
        "user": {"name": "alice", "roles": ["admin", "analyst"]},
        "flags": {1, 2, 3},
        "count": 42,
    }
    frozen = freeze_value(data)

    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["user"], MappingProxyType)
    assert isinstance(frozen["user"]["roles"], tuple)
    assert isinstance(frozen["flags"], frozenset)

    # Attempt mutations
    with pytest.raises(TypeError):
        frozen["user"]["name"] = "bob"

    with pytest.raises(AttributeError):
        frozen["user"]["roles"].append("guest")

    with pytest.raises(TypeError):
        frozen["new_key"] = 123


def test_canonical_raw_alert_nested_mutation_and_defensive_isolation():
    """CanonicalRawAlert must resist nested mutation and external source modification."""
    now = datetime.now(timezone.utc)
    source_meta = {
        "data": {"srcip": "1.2.3.4", "nested_list": [1, 2, 3]},
        "rule_groups_all": ["syslog", "pam"],
        "source_sort": [1787895525000, "1787895525.48425"],
    }

    alert = CanonicalRawAlert(
        wazuh_alert_id="1787895525.48425",
        timestamp=now,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        rule_level=3,
        rule_id="5501",
        mitre_tactics=("Defense Evasion",),
        srcip="1.2.3.4",
        agent_criticality=1,
        metadata=source_meta,
    )

    # 1. Direct mutation of nested mapping raises TypeError
    with pytest.raises(TypeError):
        alert.metadata["data"]["srcip"] = "8.8.8.8"

    # 2. Direct mutation of nested list/tuple raises AttributeError (no append)
    with pytest.raises(AttributeError):
        alert.metadata["rule_groups_all"].append("attack")

    with pytest.raises(AttributeError):
        alert.metadata["source_sort"].append("extra")

    # 3. Defensive isolation: mutating the source dict does not mutate the DTO
    source_meta["data"]["srcip"] = "8.8.8.8"
    source_meta["rule_groups_all"].append("attack")
    source_meta["new_field"] = "unexpected"

    assert alert.metadata["data"]["srcip"] == "1.2.3.4"
    assert alert.metadata["rule_groups_all"] == ("syslog", "pam")
    assert "new_field" not in alert.metadata
