"""Fixed tumbling time-window baseline implementation (time-only, non-contextual)."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

from src.contracts.meta_alert import MetaAlert
from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.metrics import compute_arr


@dataclass(frozen=True)
class FixedWindowResult:
    """Evaluation result for Fixed Tumbling Window Baseline."""

    n_raw: int
    n_meta: int
    arr: float
    meta_alerts: List[MetaAlert]


def run_fixed_window_baseline(
    alerts: Iterable[CanonicalRawAlert],
    window_duration: timedelta = timedelta(minutes=15),
) -> FixedWindowResult:
    """Partition incoming raw alerts strictly by tumbling calendar time windows.

    Parameters
    ----------
    alerts : Iterable[CanonicalRawAlert]
        Incoming sequence of canonical raw alerts.
    window_duration : timedelta
        Fixed time window slicing duration (default 15 minutes).

    Returns
    -------
    FixedWindowResult
        Aggregated results and computed ARR for fixed-window baseline.
    """
    sorted_alerts = sorted(list(alerts), key=lambda a: a.timestamp)
    n_raw = len(sorted_alerts)

    if n_raw == 0:
        return FixedWindowResult(n_raw=0, n_meta=0, arr=0.0, meta_alerts=[])

    window_sec = window_duration.total_seconds()
    first_ts = sorted_alerts[0].timestamp
    windows: Dict[int, List[CanonicalRawAlert]] = {}

    for a in sorted_alerts:
        offset_sec = (a.timestamp - first_ts).total_seconds()
        bucket_idx = int(offset_sec // window_sec)
        if bucket_idx not in windows:
            windows[bucket_idx] = []
        windows[bucket_idx].append(a)

    meta_alerts: List[MetaAlert] = []
    meta_id = 1

    for bucket_idx in sorted(windows.keys()):
        b_alerts = windows[bucket_idx]
        start_t = b_alerts[0].timestamp
        end_t = b_alerts[-1].timestamp
        count = len(b_alerts)
        max_sev = max(a.rule_level for a in b_alerts)

        rule_counts = Counter(a.rule_id for a in b_alerts)
        sev_counts = Counter(a.rule_level for a in b_alerts)

        # Baseline combines across agents/groups
        first_a = b_alerts[0]
        agent_id = "baseline_mixed" if len(set(a.agent_id for a in b_alerts)) > 1 else first_a.agent_id
        agent_name = "baseline_mixed" if len(set(a.agent_name for a in b_alerts)) > 1 else first_a.agent_name
        rule_group = "baseline_mixed" if len(set(a.rule_group_primary for a in b_alerts)) > 1 else first_a.rule_group_primary
        agent_crit = max(a.agent_criticality for a in b_alerts)

        mitre_tactics: List[str] = []
        for a in b_alerts:
            for t in a.mitre_tactics:
                if t not in mitre_tactics:
                    mitre_tactics.append(t)

        crit_mitre = any(t in ("Initial Access", "Execution", "Impact", "Lateral Movement", "Exfiltration") for t in mitre_tactics)

        meta = MetaAlert(
            meta_id=meta_id,
            agent_id=agent_id,
            agent_name=agent_name,
            rule_group_primary=rule_group,
            start_time=start_t,
            end_time=end_t,
            alert_count=count,
            max_severity=max_sev,
            rule_id_distribution=dict(rule_counts),
            severity_distribution=dict(sev_counts),
            mitre_tactics_unique=tuple(mitre_tactics),
            critical_mitre_present=crit_mitre,
            agent_criticality=agent_crit,
            wazuh_alert_ids=tuple(a.wazuh_alert_id for a in b_alerts),
            metadata={"baseline_window_idx": bucket_idx},
        )
        meta_alerts.append(meta)
        meta_id += 1

    n_meta = len(meta_alerts)
    arr = compute_arr(n_raw, n_meta)

    return FixedWindowResult(
        n_raw=n_raw,
        n_meta=n_meta,
        arr=arr,
        meta_alerts=meta_alerts,
    )
