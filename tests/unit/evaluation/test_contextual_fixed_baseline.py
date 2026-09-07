from datetime import datetime, timedelta, timezone

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.contextual_fixed_baseline import run_contextual_fixed_window_baseline


def _alert(idx: int, minute: int, agent: str, group: str) -> CanonicalRawAlert:
    return CanonicalRawAlert(
        wazuh_alert_id=f"a-{idx}",
        timestamp=datetime(2026, 9, 7, 10, minute, tzinfo=timezone.utc),
        agent_id=agent,
        agent_name=f"agent-{agent}",
        rule_group_primary=group,
        rule_level=5,
        rule_id="5710",
        mitre_tactics=(),
        srcip=None,
        agent_criticality=1,
    )


def test_contextual_fixed_window_splits_same_time_window_by_context():
    alerts = [
        _alert(1, 1, "001", "pam"),
        _alert(2, 2, "001", "pam"),
        _alert(3, 3, "002", "pam"),
        _alert(4, 4, "001", "web"),
    ]

    result = run_contextual_fixed_window_baseline(alerts, timedelta(minutes=15))

    assert result.n_raw == 4
    assert result.n_meta == 3
    assert result.arr == 25.0
    assert {(m.agent_id, m.rule_group_primary) for m in result.meta_alerts} == {
        ("001", "pam"), ("002", "pam"), ("001", "web")
    }
    assert sorted(m.alert_count for m in result.meta_alerts) == [1, 1, 2]
