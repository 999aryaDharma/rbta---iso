import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import pytest

from src.contracts.meta_alert import MetaAlert
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runtime.replay_controller import ReplayController, ALL_DATASETS_SENTINEL


@pytest.fixture
def test_bundle():
    base_t = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    metas = [
        MetaAlert(
            meta_id=i,
            agent_id="001",
            agent_name="agent-ubuntu",
            rule_group_primary="authentication_failed",
            start_time=base_t + timedelta(hours=i),
            end_time=base_t + timedelta(hours=i, minutes=10),
            alert_count=5 + i,
            max_severity=3 + (i % 10),
            rule_id_distribution={"5710": 5 + i},
            severity_distribution={3: 5 + i},
            agent_criticality=1.0,
            wazuh_alert_ids=(f"a_{i}_1", f"a_{i}_2"),
            mitre_tactics_unique=("credential-access",),
            critical_mitre_present=False,
            metadata={},
        )
        for i in range(1, 10)
    ]
    return train_reference_pipeline(metas, random_state=42, model_version="replay-v1")


def create_sample_wazuh_jsonl(file_path: Path, count: int = 20, start_idx: int = 0) -> None:
    base_t = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    lines = []
    for i in range(start_idx, start_idx + count):
        ts = base_t + timedelta(minutes=i * 2)
        event = {
            "id": f"event-{i:04d}",
            "timestamp": ts.isoformat(),
            "agent": {"id": "001", "name": "ubuntu-srv"},
            "rule": {
                "id": f"571{i % 5}",
                "level": (i % 12) + 1,
                "description": f"Test alert {i}",
                "groups": ["authentication_failed", "syslog"],
                "mitre": {"tactic": ["credential-access"]},
            },
            "data": {"srcip": f"192.168.1.{100 + (i % 10)}"},
            "full_log": f"Failed password for user from 192.168.1.{100 + (i % 10)}",
        }
        lines.append(json.dumps(event))
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_all_datasets_discovery_and_sorted_order(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "a_dataset.jsonl", 5)
    create_sample_wazuh_jsonl(data_dir / "c_dataset.jsonl", 5)
    create_sample_wazuh_jsonl(data_dir / "b_dataset.jsonl", 5)

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start(ALL_DATASETS_SENTINEL, speed_factor="MAX")
    controller.wait_until_complete(5.0)
    status = controller.get_status()

    assert status["status"] == "COMPLETED"
    assert status["total_count"] == 15
    assert status["processed_count"] == 15
    assert status["dataset_mode"] == "all"
    assert status["dataset_count"] == 3


def test_all_datasets_single_run_id(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "a.jsonl", 2)
    create_sample_wazuh_jsonl(data_dir / "b.jsonl", 3)

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start(ALL_DATASETS_SENTINEL, speed_factor="MAX")
    controller.wait_until_complete(5.0)
    
    assert controller.run_id is not None
    # Verify there is exactly one run workspace created
    runs = list(runs_dir.iterdir())
    assert len(runs) == 1
    assert runs[0].name == controller.run_id


def test_all_datasets_continuous_state(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "part1.jsonl", 10, start_idx=0)
    create_sample_wazuh_jsonl(data_dir / "part2.jsonl", 10, start_idx=10)

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start(ALL_DATASETS_SENTINEL, speed_factor="MAX")
    controller.wait_until_complete(5.0)

    # Check that events from both datasets were processed continuously
    assert controller.processed_count == 20
    # Current dataset should end at part2
    assert controller.current_dataset == "part2.jsonl"
    assert controller.current_dataset_index == 1


def test_all_datasets_error_reports_correct_dataset(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "1_good.jsonl", 5)
    
    bad_file = data_dir / "2_bad.jsonl"
    # Line 1 good, line 2 bad
    bad_file.write_text("{\"id\": \"1\", \"timestamp\": \"2026-08-29T10:00:00Z\", \"agent\": {\"id\": \"001\"}, \"rule\": {\"id\": \"1\", \"level\": 3, \"groups\": [\"syslog\"]}}\nNOT_JSON\n", encoding="utf-8")

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start(ALL_DATASETS_SENTINEL, speed_factor="MAX")
    time.sleep(0.5)

    status = controller.get_status()
    assert status["status"] == "ERROR"
    assert status["last_error"] is not None
    assert status["last_error"]["dataset"] == "2_bad.jsonl"
    assert status["last_error"]["line_number"] == 2
    assert status["processed_count"] == 6 # 5 from good, 1 from bad


def test_all_datasets_pause_resume(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "part1.jsonl", 10)
    create_sample_wazuh_jsonl(data_dir / "part2.jsonl", 10)

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start(ALL_DATASETS_SENTINEL, speed_factor="1")
    time.sleep(0.05)
    
    controller.pause()
    assert controller.status == "PAUSED"
    
    time.sleep(0.1)
    
    controller.resume()
    assert controller.status == "RUNNING"
    
    controller.stop()


def test_single_dataset_backward_compatible(tmp_path: Path, test_bundle):
    data_dir = tmp_path / "datasets"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    create_sample_wazuh_jsonl(data_dir / "single.jsonl", 5)

    controller = ReplayController(
        scoring_pipeline=ScoringPipeline(test_bundle),
        replay_data_dir=data_dir,
        replay_runs_dir=runs_dir,
    )

    controller.start("single.jsonl", speed_factor="MAX")
    controller.wait_until_complete(5.0)

    status = controller.get_status()
    assert status["status"] == "COMPLETED"
    assert status["total_count"] == 5
    assert status["dataset_mode"] == "single"
    assert status["dataset_count"] == 1
    assert status["current_dataset"] == "single.jsonl"
    assert status["current_dataset_index"] == 0
