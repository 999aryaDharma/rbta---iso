from datetime import datetime, timezone

from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.runtime.replay_evaluation import ReplayEvaluationTracker


def _scored(meta_id: int, action: str, decision: str, score: float, source_count: int) -> ScoredMetaAlert:
    return ScoredMetaAlert(
        meta_id=meta_id,
        agent_id="001",
        agent_name="demo",
        rule_group_primary="auth",
        start_time=datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 9, 7, 10, 1, tzinfo=timezone.utc),
        alert_count=source_count,
        max_severity=12 if action == "ESCALATE" else 3,
        mitre_tactics=("Execution",) if action == "ESCALATE" else (),
        seven_features={
            "max_severity": 12.0,
            "mitre_tactic_count": 1.0,
            "critical_mitre_tactic_present": 1.0,
            "alert_count_log": 1.0,
            "rule_diversity_shannon": 0.0,
            "severity_dispersion": 0.0,
            "agent_criticality": 2.0,
        },
        raw_model_score=score,
        anomaly_score=score,
        threshold_used=0.5,
        decision=decision,
        action=action,
        escalate=action == "ESCALATE",
        model_version="frozen-v1",
        feature_schema_version="1.0",
        score_calibration_version="minmax-v1",
        source_alert_ids=tuple(f"source-{meta_id}-{idx}" for idx in range(source_count)),
    )


def test_live_tracker_reports_reduction_scores_decisions_and_boundaries():
    tracker = ReplayEvaluationTracker(
        model_metadata={
            "model_version": "frozen-v1",
            "training_run_id": "train-123",
            "training_period_start": "2025-05-10T00:00:00+00:00",
            "training_period_end": "2026-01-01T00:00:00+00:00",
            "validation_strategy": "chronological_reference_calibration_test",
            "contamination": "auto",
        }
    )
    tracker.record_raw(10)
    tracker.record_scored(_scored(1, "ESCALATE", "CRITICAL", 0.8, 3))
    tracker.record_scored(_scored(2, "SUPPRESS", "NOISE", 1.2, 2))

    snapshot = tracker.snapshot(active_buckets=2, evidence_count=10)

    assert snapshot["schema_version"] == "1.0"
    assert snapshot["triage_units_current"] == 4
    assert snapshot["live_arr_percent"] == 60.0
    assert snapshot["decision_distribution"] == {
        "CRITICAL": 1, "SUSPICIOUS": 0, "NOISE_HIGH": 0, "NOISE": 1
    }
    assert snapshot["action_distribution"]["ESCALATE"] == 1
    assert snapshot["threshold"]["above_count"] == 2
    assert snapshot["score_distribution"]["count"] == 2
    assert sum(snapshot["score_distribution"]["histogram"]) == 2
    assert snapshot["reference_range_exceedance_count"] == 1
    assert snapshot["evidence_coverage_percent"] == 100.0
    assert snapshot["source_reference_coverage_percent"] == 50.0
    assert snapshot["model_provenance"]["training_run_id"] == "train-123"
    assert snapshot["claim_boundary"]["accuracy_available"] is False
    assert "bukan" in snapshot["claim_boundary"]["silhouette_interpretation"].lower()
