from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.context_quality import compute_context_quality
from src.evaluation.fixed_window_baseline import run_fixed_window_baseline


def _alert(idx: int, minute: int, agent: str, group: str) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"a-{idx}",
        timestamp=datetime(2026, 9, 7, 10, minute, tzinfo=timezone.utc),
        agent_id=agent,
        agent_name=f"agent-{agent}",
        rule_group_primary=group,
        rule_level=3,
        rule_id="1",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
    )


def test_context_quality_exposes_mixed_bucket_instead_of_rewarding_arr_only():
    alerts = [
        _alert(1, 1, "001", "pam"),
        _alert(2, 2, "002", "web"),
        _alert(3, 16, "001", "pam"),
    ]
    metas = run_fixed_window_baseline(alerts, timedelta(minutes=15)).meta_alerts

    quality = compute_context_quality(metas, alerts)

    assert quality.total_meta_alerts == 2
    assert quality.pure_meta_alerts == 1
    assert quality.contaminated_meta_alerts == 1
    assert quality.context_purity_percent == 50.0
    assert quality.context_contamination_percent == 50.0


def test_context_quality_fails_closed_when_source_member_is_missing():
    alerts = [_alert(1, 1, "001", "pam"), _alert(2, 2, "002", "web")]
    metas = run_fixed_window_baseline(alerts, timedelta(minutes=15)).meta_alerts

    with pytest.raises(ValueError, match="missing source alert"):
        compute_context_quality(metas, alerts[:1])
