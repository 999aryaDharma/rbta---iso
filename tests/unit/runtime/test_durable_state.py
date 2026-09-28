"""Unit tests for DurableStateManager and crash-recovery state restoration (Sprint 7)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
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
    assert restored_engine.snapshot_agents()[0]["agent_name"] == "soc-001"

    # Processing duplicate alert_1 in restored engine is idempotent
    assert restored_engine.process(a1) == []
    assert restored_engine._active_buckets[("001", "pam")].alert_count == 2

    # Processing new alert_4 in restored engine merges into existing restored bucket
    a4 = make_alert(4, base_t + timedelta(minutes=10), agent_id="001", group="pam")
    assert restored_engine.process(a4) == []
    assert restored_engine._active_buckets[("001", "pam")].alert_count == 3
    assert restored_engine._active_buckets[("001", "pam")].wazuh_alert_ids == ["alert_1", "alert_2", "alert_4"]


def test_seen_alert_ids_are_persisted_in_sqlite_not_rewritten_in_json(tmp_path: Path):
    """Checkpoint JSON stays bounded while duplicate protection survives restart."""
    state_file = tmp_path / "runtime_state.json"
    manager = DurableStateManager(state_file)
    engine = RBTAEngine(base_delta_t=timedelta(minutes=15))
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)

    for idx in range(25):
        engine.process(make_alert(idx, base_t + timedelta(seconds=idx)))
    manager.save_state(engine=engine)

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert "seen_alert_ids" not in payload
    assert manager.count_seen_alert_ids() == 25

    restored = RBTAEngine(base_delta_t=timedelta(minutes=15))
    manager.restore_state(restored)
    assert restored._seen_alert_ids == {f"alert_{idx}" for idx in range(25)}


def test_finalized_history_supports_indexed_pagination_and_direct_lookup(tmp_path: Path):
    manager = DurableStateManager(tmp_path / "state.json")
    manager.append_finalized([
        {"meta_id": idx, "decision": "CRITICAL" if idx % 2 else "NOISE", "agent_id": "001", "agent_name": "soc", "rule_group_primary": "pam", "anomaly_score": idx / 10}
        for idx in range(1, 11)
    ])

    items, total = manager.query_finalized(page=2, page_size=2, decision="CRITICAL", sort_by="meta_id", sort_order="desc")
    assert total == 5
    assert [item["meta_id"] for item in items] == [5, 3]
    assert manager.get_finalized(8)["meta_id"] == 8
    assert manager.get_finalized(999) is None


def test_restore_only_loads_bounded_recent_history(tmp_path: Path):
    manager = DurableStateManager(tmp_path / "state.json")
    manager.append_finalized([{"meta_id": idx} for idx in range(1, 1202)])
    restored = manager.restore_state(RBTAEngine(base_delta_t=timedelta(minutes=15)))
    assert len(restored["finalized_history"]) == 1000
    assert restored["finalized_history"][0]["meta_id"] == 202


def test_failed_checkpoint_does_not_commit_seen_ids_or_history(tmp_path):
    manager = DurableStateManager(tmp_path / "state.json")
    engine = RBTAEngine()
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    engine.process(make_alert(1, now))
    manager.save_state(engine)
    engine.process(make_alert(2, now + timedelta(seconds=1)))

    with pytest.raises(TypeError):
        manager.save_state(
            engine, source_checkpoint={"invalid": object()},
            new_finalized_history=[{"meta_id": 42}],
        )

    restored = RBTAEngine()
    manager.restore_state(restored)
    assert not manager.has_seen_alert_id("alert_2")
    assert manager.get_finalized(42) is None
    assert restored._active_buckets[("001", "pam")].wazuh_alert_ids == ["alert_1"]
    assert "alert_2" in engine._new_seen_alert_ids


def test_recovery_after_json_publication_failure_keeps_dedup_and_bucket_together(tmp_path, monkeypatch):
    manager = DurableStateManager(tmp_path / "state.json")
    engine = RBTAEngine()
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    engine.process(make_alert(1, now))
    manager.save_state(engine)
    engine.process(make_alert(2, now + timedelta(seconds=1)))

    def fail_replace(*args, **kwargs):
        raise OSError("simulated interrupted JSON publication")

    monkeypatch.setattr(Path, "replace", fail_replace)
    try:
        manager.save_state(engine)
    except OSError:
        pass
    restored = RBTAEngine()
    DurableStateManager(manager.state_path).restore_state(restored)
    assert restored._seen_alert_ids == set(restored._active_buckets[("001", "pam")].wazuh_alert_ids)
    assert restored._active_buckets[("001", "pam")].wazuh_alert_ids == ["alert_1", "alert_2"]


def test_sqlite_checkpoint_failure_rolls_back_history_dedup_and_snapshot(tmp_path):
    manager = DurableStateManager(tmp_path / "state.json")
    engine = RBTAEngine()
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    engine.process(make_alert(1, now))
    manager.save_state(engine, source_checkpoint={"cursor": 1})
    engine.process(make_alert(2, now + timedelta(seconds=1)))
    with sqlite3.connect(manager.history_db_path) as conn:
        conn.execute("CREATE TRIGGER fail_checkpoint BEFORE INSERT ON runtime_snapshot BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        manager.save_state(engine, source_checkpoint={"cursor": 2}, new_finalized_history=[{"meta_id": 42}])
    recovered = RBTAEngine()
    restored = manager.restore_state(recovered)
    assert restored["source_checkpoint"] == {"cursor": 1}
    assert not manager.has_seen_alert_id("alert_2")
    assert manager.get_finalized(42) is None
    assert recovered._active_buckets[("001", "pam")].alert_count == 1
    assert "alert_2" in engine._new_seen_alert_ids


def test_legacy_json_snapshot_migrates_without_reset(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema_version": "1.0", "seen_alert_ids": ["legacy-alert"],
        "meta_id_counter": 7, "source_checkpoint": {"cursor": "legacy"},
        "pending_scoring": [], "outbox": [{"meta_id": 6}],
    }), encoding="utf-8")
    manager = DurableStateManager(state)
    engine = RBTAEngine()
    restored = manager.restore_state(engine)
    manager.save_state(engine, outbox=restored["outbox"], source_checkpoint=restored["source_checkpoint"])
    state.unlink()  # SQLite alone must suffice after migration.
    recovered = RBTAEngine()
    restored = DurableStateManager(state).restore_state(recovered)
    assert recovered._seen_alert_ids == {"legacy-alert"}
    assert recovered._meta_id_counter == 7
    assert restored["outbox"] == [{"meta_id": 6}]
    assert restored["source_checkpoint"] == {"cursor": "legacy"}
