"""Unit tests for Structural Silhouette and Permutation Baseline Evaluation (Sprint 8)."""
from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.structural_silhouette import run_structural_silhouette_evaluation
from src.model.scoring_pipeline import ScoringPipeline, train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner


def make_alert(idx: int, ts: datetime, agent_id: str = "001", group: str = "pam", level: int = 3, crit: int = 1) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"alert_{idx}",
        timestamp=ts,
        agent_id=agent_id,
        agent_name=f"soc-{agent_id}",
        rule_group_primary=group,
        rule_level=level,
        rule_id=f"550{idx % 5}",
        mitre_tactics=("Execution",) if idx % 4 == 0 else (),
        srcip=None,
        agent_criticality=crit,
    )


def test_structural_silhouette_and_permutation_baseline():
    """Structural silhouette evaluates observed partition against 100 same-proportion permutations."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [
        make_alert(
            i,
            base_t + timedelta(minutes=i * 20),
            agent_id=f"agent_{(i % 3) + 1}",
            group="pam" if i % 2 == 0 else "syslog",
            level=(i % 12) + 1,
            crit=(i % 3) + 1,
        )
        for i in range(50)
    ]

    # Batch run & train model
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="eval-v1")
    scoring_pipe = ScoringPipeline(bundle)
    _, scored_metas = scoring_pipe.score_meta_alerts(batch_res.meta_alerts)

    result = run_structural_silhouette_evaluation(
        scored_metas=scored_metas,
        model_bundle=bundle,
        n_permutations=100,
        random_seed=42,
    )

    if result.is_calculable:
        assert -1.0 <= result.observed_silhouette <= 1.0
        assert result.n_valid_permutations == 100
        assert 0.0 <= result.observed_percentile <= 100.0
        assert 0.0 <= result.empirical_p_value <= 1.0
        assert result.random_mean is not None
        assert result.random_std is not None
    else:
        # If partition only had 1 class, is_calculable is False
        assert result.uncalculable_reason is not None
