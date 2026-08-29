import pytest
from datetime import datetime, timezone
import os

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.raw_evidence import RawAlertEvidenceStore

@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_raw_evidence.sqlite3"
    return RawAlertEvidenceStore(db_path)

def test_store_and_retrieve_single_alert(store):
    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-1",
        timestamp=datetime.now(timezone.utc),
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="syslog",
        rule_level=3,
        rule_id="1000",
        mitre_tactics=("Initial Access",),
        srcip=None,
        agent_criticality=1,
        metadata={
            "rule_description": "Test alert",
            "rule_groups_all": ["syslog", "test"],
            "mitre_techniques": ["T1190"],
        }
    )

    assert store.store(alert) is True

    retrieved = store.get("alert-1")
    assert retrieved is not None
    assert retrieved["wazuh_alert_id"] == "alert-1"
    assert retrieved["agent_id"] == "001"
    assert "Initial Access" in retrieved["mitre_tactics"]

def test_store_idempotent_duplicate_ignored(store):
    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-2",
        timestamp=datetime.now(timezone.utc),
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="syslog",
        rule_level=3,
        rule_id="1000",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
        metadata={"rule_description": "Test alert", "rule_groups_all": ["syslog"]}
    )

    assert store.store(alert) is True
    assert store.store(alert) is False  # Second time should be False

def test_get_nonexistent_returns_none(store):
    assert store.get("nonexistent") is None

def test_get_many_partial_resolution(store):
    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-3",
        timestamp=datetime.now(timezone.utc),
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="syslog",
        rule_level=3,
        rule_id="1000",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
        metadata={"rule_description": "Test alert", "rule_groups_all": ["syslog"]}
    )
    store.store(alert)

    results = store.get_many(["alert-3", "nonexistent"])
    assert len(results) == 1
    assert results[0]["wazuh_alert_id"] == "alert-3"

def test_search_with_filters(store):
    for i in range(5):
        alert = CanonicalRawAlert(
            wazuh_alert_id=f"alert-{i+10}",
            timestamp=datetime.now(timezone.utc),
            agent_id="001",
            agent_name="agent-1",
            rule_group_primary="syslog",
            rule_level=i+1,
            rule_id="1000",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
            metadata={"rule_description": "Test alert", "rule_groups_all": ["syslog"]}
        )
        store.store(alert)

    items, total = store.search(
        meta_id_alert_ids=[f"alert-{i+10}" for i in range(5)],
        level_min=3
    )
    assert total == 3
    assert len(items) == 3

def test_count(store):
    assert store.count() == 0
    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-99",
        timestamp=datetime.now(timezone.utc),
        agent_id="001",
        agent_name="agent-1",
        rule_group_primary="syslog",
        rule_level=3,
        rule_id="1000",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
        metadata={"rule_description": "Test alert", "rule_groups_all": ["syslog"]}
    )
    store.store(alert)
    assert store.count() == 1
