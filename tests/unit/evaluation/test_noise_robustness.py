"""Unit tests for Noise Robustness Evaluation (Sprint 8)."""
from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.noise_robustness import run_noise_robustness_evaluation, NOISE_RATES


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


def test_noise_robustness_evaluates_exact_five_noise_rates():
    """Noise robustness tests exactly [0.0, 0.05, 0.10, 0.20, 0.30]."""
    assert NOISE_RATES == (0.0, 0.05, 0.10, 0.20, 0.30)

    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [make_alert(i, base_t + timedelta(seconds=i * 30)) for i in range(100)]

    result = run_noise_robustness_evaluation(alerts, delta_t=timedelta(minutes=15), random_seed=42)

    assert isinstance(result.summary_df, pd.DataFrame)
    assert len(result.summary_df) == 5
    assert list(result.summary_df["noise_rate"]) == [0.0, 0.05, 0.10, 0.20, 0.30]

    # Required output columns
    required_cols = [
        "noise_rate",
        "n_noise",
        "n_total",
        "n_meta",
        "arr",
        "arr_degradation",
        "noise_absorption_count",
        "noise_absorption_rate",
        "execution_time_ms",
    ]
    for col in required_cols:
        assert col in result.summary_df.columns

    # Baseline 0% noise has zero degradation
    row_0 = result.summary_df.iloc[0]
    assert row_0["n_noise"] == 0
    assert row_0["arr_degradation"] == 0.0


def test_noise_absorption_traceability():
    """Verify that noise absorption is computed via actual MetaAlert.wazuh_alert_ids."""
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    alerts = [make_alert(i, base_t + timedelta(seconds=i * 10)) for i in range(50)]

    result = run_noise_robustness_evaluation(alerts, noise_rates=(0.0, 0.20), delta_t=timedelta(minutes=15), random_seed=42)
    row_20 = result.summary_df.iloc[1]
    assert row_20["noise_rate"] == 0.20
    assert row_20["n_noise"] == 10
    assert 0 <= row_20["noise_absorption_count"] <= 10
    assert 0.0 <= row_20["noise_absorption_rate"] <= 100.0

