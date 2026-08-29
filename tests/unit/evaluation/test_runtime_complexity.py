"""Unit tests for Runtime Complexity and Throughput Evaluation (Sprint 8)."""
from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.runtime_complexity import run_runtime_complexity_evaluation


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


def test_runtime_complexity_eight_subsets_and_linear_regression():
    """Runtime evaluation measures 8 subsets, computes throughput, and fits linear regression."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [make_alert(i, base_t + timedelta(seconds=i * 5)) for i in range(160)]

    result = run_runtime_complexity_evaluation(alerts, n_subsets=8, delta_t=timedelta(minutes=15))

    assert isinstance(result.subset_df, pd.DataFrame)
    assert len(result.subset_df) == 8

    for col in ["n_alerts", "n_meta", "execution_time_ms", "throughput_alerts_per_ms"]:
        assert col in result.subset_df.columns

    assert result.slope > 0
    assert 0.0 <= result.r_squared <= 1.0
    assert result.mean_throughput > 0
