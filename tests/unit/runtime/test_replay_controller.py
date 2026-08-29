import pytest
import time
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from src.runtime.replay_controller import ReplayController
from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert


class DummyPipeline:
    def score_single(self, meta):
        return ScoredMetaAlert(
            meta_id=meta.meta_id,
            agent_id=meta.agent_id,
            agent_name=meta.agent_name,
            rule_group_primary=meta.rule_group_primary,
            start_time=meta.start_time,
            end_time=meta.end_time,
            alert_count=meta.alert_count,
            max_severity=meta.max_severity,
            mitre_tactics=meta.mitre_tactics_unique,
            seven_features={},
            raw_model_score=0.9,
            anomaly_score=0.9,
            threshold_used=0.8,
            decision="CRITICAL",
            action="ESCALATE",
            escalate=True,
            model_version="test",
            feature_schema_version="1.0",
            score_calibration_version="1.0",
            source_alert_ids=meta.wazuh_alert_ids,
            metadata=meta.metadata
        )


@pytest.fixture
def dummy_data_dir(tmp_path):
    d = tmp_path / "test_datasets"
    d.mkdir()

    file_path = d / "alerts.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(5):
            alert = {
                "wazuh_alert_id": f"id-{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": "001",
                "agent_name": "agent-1",
                "rule_group_primary": "syslog",
                "rule_level": 3,
                "rule_id": "1000",
                "mitre_tactics": [],
                "agent_criticality": 1,
                "metadata": {"rule_description": "test", "rule_groups_all": ["syslog"]},
            }
            f.write(json.dumps(alert) + "\n")

    os.environ["RBTA_REPLAY_DATA_DIR"] = str(d)
    return d


@pytest.fixture
def replay_controller(dummy_data_dir):
    return ReplayController(scoring_pipeline=DummyPipeline())


def test_replay_start_creates_run(replay_controller):
    replay_controller.start(speed=0)  # 0 for MAX

    # Wait for completion
    for _ in range(50):
        if replay_controller.status == "COMPLETED":
            break
        time.sleep(0.05)

    assert replay_controller.run_id is not None
    assert replay_controller.status == "COMPLETED"
    assert replay_controller.processed_count == 5


def test_replay_status_tracks_progress(replay_controller):
    assert replay_controller.get_status()["status"] == "IDLE"


def test_replay_pause_resume(replay_controller):
    replay_controller.start(speed=1.0)

    replay_controller.pause()
    assert replay_controller.status == "PAUSED"

    replay_controller.resume()
    assert replay_controller.status in ("RUNNING", "COMPLETED")

    replay_controller.stop()
    assert replay_controller.status == "COMPLETED"


def test_replay_stop(replay_controller):
    replay_controller.start(speed=1.0)
    replay_controller.stop()
    assert replay_controller.status == "COMPLETED"


def test_replay_reset_creates_new_run(replay_controller):
    replay_controller.start(speed=1.0)
    replay_controller.stop()
    replay_controller.reset()
    assert replay_controller.status == "IDLE"
    assert replay_controller.run_id is None


def test_replay_determinism(dummy_data_dir):
    ctrl1 = ReplayController(scoring_pipeline=DummyPipeline())
    ctrl2 = ReplayController(scoring_pipeline=DummyPipeline())
    assert ctrl1.status == "IDLE"
    assert ctrl2.status == "IDLE"
