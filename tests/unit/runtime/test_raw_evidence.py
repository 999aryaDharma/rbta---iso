from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.runtime.raw_evidence import RawAlertEvidenceStore, RawEvidenceConflictError


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "raw_evidence.sqlite3"


@pytest.fixture
def store(store_path: Path) -> RawAlertEvidenceStore:
    return RawAlertEvidenceStore(store_path)


def make_alert(
    alert_id: str = "alert-001",
    agent_id: str = "001",
    rule_id: str = "5710",
    rule_level: int = 5,
    srcip: str = "192.168.1.100",
    metadata: dict | None = None,
) -> CanonicalRawAlert:
    meta = {
        "rule_description": "sshd brute force attempt",
        "location": "/var/log/auth.log",
        "decoder": "sshd",
        "full_log": "Failed password for root from 192.168.1.100 port 22",
        "source_index": "wazuh-alerts-4.x-2026.08.29",
        "source_document_id": "doc-12345",
    }
    if metadata:
        meta.update(metadata)

    return CanonicalRawAlert(
        wazuh_alert_id=alert_id,
        timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        agent_id=agent_id,
        agent_name="agent-ubuntu",
        rule_group_primary="authentication_failed",
        rule_level=rule_level,
        rule_id=rule_id,
        mitre_tactics=("credential-access", "initial-access"),
        srcip=srcip,
        agent_criticality=1.0,
        metadata=MappingProxyType(meta),
    )


def test_store_and_retrieve_roundtrip(store: RawAlertEvidenceStore):
    alert = make_alert("wazuh-001")
    inserted = store.store(alert, source_mode="LIVE")
    assert inserted is True

    record = store.get("wazuh-001", redact=False)
    assert record is not None
    assert record["wazuh_alert_id"] == "wazuh-001"
    assert record["rule_id"] == "5710"
    assert record["rule_level"] == 5
    assert record["rule_description"] == "sshd brute force attempt"
    assert record["source_index"] == "wazuh-alerts-4.x-2026.08.29"
    assert record["source_document_id"] == "doc-12345"
    assert "credential-access" in record["mitre_tactics"]


def test_store_idempotent_identical_duplicate(store: RawAlertEvidenceStore):
    alert = make_alert("wazuh-dup")
    assert store.store(alert) is True
    # Re-storing exact same alert is safe NO-OP
    assert store.store(alert) is False
    assert store.count() == 1


def test_store_conflicting_duplicate_raises_error(store: RawAlertEvidenceStore):
    alert1 = make_alert("wazuh-conflict", rule_level=5)
    assert store.store(alert1) is True

    # Same ID but different rule_level -> conflict error!
    alert2 = make_alert("wazuh-conflict", rule_level=12)
    with pytest.raises(RawEvidenceConflictError) as exc_info:
        store.store(alert2)
    assert "Conflicting canonical evidence" in str(exc_info.value)
    assert "wazuh-conflict" in str(exc_info.value)


def test_nested_immutable_metadata_serialization(store: RawAlertEvidenceStore):
    nested_meta = {
        "rule_description": "Nested test",
        "manager": {"name": "wazuh-master"},
        "data": {
            "srcip": "10.0.0.5",
            "tags": ("tag1", "tag2"),
            "sub": {"nested_key": "val"},
        },
        "secret_token": "super_secret_12345",
    }
    alert = make_alert("wazuh-nested", metadata=nested_meta)
    assert store.store(alert) is True

    # Unredacted
    unredacted = store.get("wazuh-nested", redact=False)
    assert unredacted is not None
    assert unredacted["metadata"]["secret_token"] == "super_secret_12345"
    assert unredacted["metadata"]["data"]["tags"] == ["tag1", "tag2"]

    # Redacted presentation
    redacted = store.get("wazuh-nested", redact=True)
    assert redacted is not None
    assert redacted["metadata"]["secret_token"] == "[REDACTED]"
    assert redacted["metadata"]["data"]["srcip"] == "10.0.0.5"


def test_get_meta_alert_raw_alerts_resolution_and_unresolved(store: RawAlertEvidenceStore):
    alert1 = make_alert("aid-1", rule_id="5710", rule_level=5, srcip="10.0.0.1")
    alert2 = make_alert("aid-2", rule_id="5715", rule_level=10, srcip="10.0.0.2")
    store.store(alert1)
    store.store(alert2)

    # Source IDs include aid-1, aid-2, and missing aid-3
    source_ids = ["aid-1", "aid-2", "aid-3"]
    res = store.get_meta_alert_raw_alerts(source_ids, meta_id=101, page=1, page_size=10)

    assert res["meta_id"] == 101
    assert res["source_total"] == 3
    assert res["resolved_total"] == 2
    assert res["unresolved_alert_ids"] == ["aid-3"]
    assert res["filtered_total"] == 2
    assert len(res["items"]) == 2
    assert res["items"][0]["wazuh_alert_id"] == "aid-1"
    assert res["items"][1]["wazuh_alert_id"] == "aid-2"


def test_get_meta_alert_raw_alerts_filters_and_search(store: RawAlertEvidenceStore):
    alert1 = make_alert("aid-a", rule_id="5710", rule_level=4, srcip="192.168.1.10", metadata={"rule_description": "SSH login failure"})
    alert2 = make_alert("aid-b", rule_id="5715", rule_level=9, srcip="192.168.1.20", metadata={"rule_description": "SSH root login accepted"})
    store.store(alert1)
    store.store(alert2)

    source_ids = ["aid-a", "aid-b"]

    # Filter by level_min
    res = store.get_meta_alert_raw_alerts(source_ids, level_min=8)
    assert res["source_total"] == 2
    assert res["resolved_total"] == 2
    assert res["filtered_total"] == 1
    assert res["items"][0]["wazuh_alert_id"] == "aid-b"

    # Multi-field search by description
    res_search = store.get_meta_alert_raw_alerts(source_ids, search="root login")
    assert res_search["filtered_total"] == 1
    assert res_search["items"][0]["wazuh_alert_id"] == "aid-b"


def test_search_multi_field(store: RawAlertEvidenceStore):
    alert1 = make_alert("search-1", rule_id="1001", srcip="172.16.0.1", metadata={"full_log": "Kernel panic detected"})
    alert2 = make_alert("search-2", rule_id="1002", srcip="172.16.0.2", metadata={"rule_description": "Firewall drop event"})
    store.store(alert1)
    store.store(alert2)

    # Search by full_log content
    found = store.search(query="Kernel panic")
    assert len(found) == 1
    assert found[0]["wazuh_alert_id"] == "search-1"

    # Search by IP
    found_ip = store.search(query="172.16.0.2")
    assert len(found_ip) == 1
    assert found_ip[0]["wazuh_alert_id"] == "search-2"
