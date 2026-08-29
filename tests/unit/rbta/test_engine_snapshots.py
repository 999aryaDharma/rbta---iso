import pytest
from datetime import datetime, timedelta, timezone

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine

def test_snapshot_agents_empty_engine():
    engine = RBTAEngine()
    assert engine.snapshot_agents() == []

def test_snapshot_buckets_empty_engine():
    engine = RBTAEngine()
    assert engine.snapshot_buckets() == []

def test_snapshot_agents_returns_agent_state():
    engine = RBTAEngine()

    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-1",
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

    engine.process(alert)

    agents = engine.snapshot_agents()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "001"
    assert agents[0]["agent_name"] == "agent-1"
    assert agents[0]["event_count"] == 1
    assert agents[0]["status"] == "WARMUP"
    assert agents[0]["active_bucket_count"] == 1

def test_snapshot_buckets_returns_active_buckets():
    engine = RBTAEngine()

    alert = CanonicalRawAlert(
        wazuh_alert_id="alert-1",
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

    engine.process(alert)

    buckets = engine.snapshot_buckets()
    assert len(buckets) == 1
    assert buckets[0]["agent_id"] == "001"
    assert buckets[0]["rule_group_primary"] == "syslog"
    assert buckets[0]["alert_count"] == 1
