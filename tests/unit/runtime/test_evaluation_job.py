from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.contracts.meta_alert import MetaAlert
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.research.orchestrator import _generate_engineering_smoke_fixture
from src.runners.batch_runner import BatchResearchRunner
from src.runtime.evaluation_job import EvaluationJobController


def _pipeline() -> ScoringPipeline:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metas = [
        MetaAlert(
            meta_id=i,
            agent_id=str(i % 3),
            agent_name=f"agent-{i % 3}",
            rule_group_primary="auth" if i % 2 else "web",
            start_time=base + timedelta(hours=i),
            end_time=base + timedelta(hours=i, minutes=2),
            alert_count=i + 1,
            max_severity=(i % 14) + 1,
            rule_id_distribution={str(i % 5): i + 1},
            severity_distribution={(i % 14) + 1: i + 1},
            agent_criticality=(i % 4) + 1,
            wazuh_alert_ids=(f"train-{i}",),
            mitre_tactics_unique=("Execution",) if i % 4 == 0 else (),
            critical_mitre_present=i % 4 == 0,
        )
        for i in range(1, 40)
    ]
    return ScoringPipeline(train_reference_pipeline(metas, model_version="frozen-test-v1"))


def test_evaluation_job_runs_every_phase_and_persists_atomic_artifact(tmp_path: Path):
    alerts = _generate_engineering_smoke_fixture(n_alerts=80, seed=42)
    job = EvaluationJobController(_pipeline())
    artifact = tmp_path / "evaluation.json"

    started = job.start("run-123", alerts, artifact)
    assert started["status"] in {"RUNNING", "COMPLETED"}
    completed = job.wait_until_complete(timeout=15.0)

    assert completed["status"] == "COMPLETED"
    assert completed["progress_percent"] == 100.0
    assert completed["completed_phases"] == 7
    assert completed["total_phases"] == 7
    assert artifact.exists()
    results = completed["results"]
    assert len(results["sensitivity"]) == 8
    assert {row["variant"] for row in results["aggregation_ablation"]} == {
        "time_only_fixed", "contextual_fixed", "contextual_adaptive"
    }
    assert len(results["noise_robustness"]) == 15
    assert len(results["runtime"]["subsets"]) == 8
    assert results["runtime"]["repetitions"] == 5
    assert results["isolation_forest"]["evaluation_population"] == "frozen_model_replay"
    assert results["interpretation"]["attack_detection_accuracy_available"] is False
