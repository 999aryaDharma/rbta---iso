"""Unit tests for DurableStateManager and crash-recovery state restoration (Sprint 7)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.rbta.engine import RBTAEngine
from src.runtime.durable_state import DurableStateManager


def make_alert(idx: int, ts: datetime, agent_id: str = "001", group: str = "pam") -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"alert_{idx}",
        timestamp=ts,
        agent_id=agent_id,
        agent_name=f"soc-{agent_id}",
        rule_group_primary=group,
        rule_level=3,
        rule_id="5501",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
    )


def test_durable_state_save_and_restore_engine(tmp_path: Path):
    """Engine state (seen IDs, temporal state, active buckets, counter) persists to disk and restores identically."""
    state_file = tmp_path / "runtime_state.json"
    manager = DurableStateManager(state_file)

    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    # Ingest 2 alerts into active bucket
    a1 = make_alert(1, base_t, agent_id="001", group="pam")
    a2 = make_alert(2, base_t + timedelta(minutes=5), agent_id="001", group="pam")
    engine.process(a1)
    engine.process(a2)

    # Ingest 1 alert into separate agent bucket
    b1 = make_alert(3, base_t + timedelta(minutes=2), agent_id="002", group="syslog")
    engine.process(b1)

    # Save state
    manager.save_state(
        engine=engine,
        outbox=[{"item": 1, "meta_id": 100}],
        source_checkpoint={"mode": "live", "offset": 42},
    )
    assert state_file.exists()

    # Create fresh empty engine and restore state from disk
    restored_engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    restored_data = manager.restore_state(restored_engine)

    assert restored_data["source_checkpoint"] == {"mode": "live", "offset": 42}
    assert restored_data["outbox"] == [{"item": 1, "meta_id": 100}]

    # Verify internal engine structures restored
    assert restored_engine._seen_alert_ids == {"alert_1", "alert_2", "alert_3"}
    assert ("001", "pam") in restored_engine._active_buckets
    assert ("002", "syslog") in restored_engine._active_buckets

    bucket_001 = restored_engine._active_buckets[("001", "pam")]
    assert bucket_001.alert_count == 2
    assert bucket_001.wazuh_alert_ids == ["alert_1", "alert_2"]
    assert bucket_001.end_time == base_t + timedelta(minutes=5)

    # Processing duplicate alert_1 in restored engine is idempotent
    assert restored_engine.process(a1) == []
    assert restored_engine._active_buckets[("001", "pam")].alert_count == 2

    # Processing new alert_4 in restored engine merges into existing restored bucket
    a4 = make_alert(4, base_t + timedelta(minutes=10), agent_id="001", group="pam")
    assert restored_engine.process(a4) == []
    assert restored_engine._active_buckets[("001", "pam")].alert_count == 3
    assert restored_engine._active_buckets[("001", "pam")].wazuh_alert_ids == ["alert_1", "alert_2", "alert_4"]
