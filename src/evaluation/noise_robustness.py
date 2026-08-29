"""Noise robustness evaluation injecting realistic false-positive alerts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.metrics import compute_arr
from src.runners.batch_runner import BatchResearchRunner

NOISE_RATES: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class NoiseRobustnessResult:
    """Outcome of noise robustness evaluation."""

    summary_df: pd.DataFrame


def _generate_noise_alerts(
    clean_alerts: Sequence[CanonicalRawAlert],
    n_noise: int,
    rng: random.Random,
) -> List[CanonicalRawAlert]:
    """Synthesize false-positive-like noise alerts preserving valid agent/group semantics."""
    if not clean_alerts or n_noise <= 0:
        return []

    # Extract valid agent tuples (agent_id, agent_name, agent_criticality)
    agents = list({(a.agent_id, a.agent_name, a.agent_criticality) for a in clean_alerts})
    rule_groups = list({a.rule_group_primary for a in clean_alerts})
    min_ts = min(a.timestamp for a in clean_alerts)
    max_ts = max(a.timestamp for a in clean_alerts)
    span_sec = max(1.0, (max_ts - min_ts).total_seconds())

    noise_alerts: List[CanonicalRawAlert] = []
    for i in range(n_noise):
        agent_id, agent_name, agent_crit = rng.choice(agents)
        rule_grp = rng.choice(rule_groups)
        offset_sec = rng.uniform(0.0, span_sec)
        ts = min_ts + timedelta(seconds=offset_sec)
        sev = rng.randint(1, 4)

        noise_alerts.append(
            CanonicalRawAlert(
                wazuh_alert_id=f"noise_{i:06d}",
                timestamp=ts,
                agent_id=agent_id,
                agent_name=agent_name,
                rule_group_primary=rule_grp,
                rule_level=sev,
                rule_id=f"noise_rule_{rng.randint(1, 5)}",
                mitre_tactics=(),
                srcip=None,
                agent_criticality=agent_crit,
            )
        )

    return noise_alerts


def run_noise_robustness_evaluation(
    alerts: Iterable[CanonicalRawAlert],
    noise_rates: Sequence[float] = NOISE_RATES,
    delta_t: timedelta = timedelta(minutes=15),
    random_seed: int = 42,
) -> NoiseRobustnessResult:
    """Evaluate RBTA resilience against low-severity benign noise injection.

    Parameters
    ----------
    alerts : Iterable[CanonicalRawAlert]
        Clean evaluation alerts dataset.
    noise_rates : Sequence[float]
        Noise injection proportions to evaluate (default: 0.0, 0.05, 0.10, 0.20, 0.30).
    delta_t : timedelta
        Aggregation time window.
    random_seed : int
        Deterministic random seed.

    Returns
    -------
    NoiseRobustnessResult
        Summary DataFrame across noise levels.
    """
    clean_list = list(alerts)
    n_clean = len(clean_list)
    rng = random.Random(random_seed)

    records: List[Dict[str, Any]] = []
    baseline_arr: Optional[float] = None
    baseline_n_meta: int = 0

    for rate in noise_rates:
        n_noise = int(round(n_clean * rate))
        noise_alerts = _generate_noise_alerts(clean_list, n_noise, rng)
        combined_stream = sorted(clean_list + noise_alerts, key=lambda a: a.timestamp)
        n_total = len(combined_stream)

        runner = BatchResearchRunner(base_delta_t=delta_t, adaptive=True)
        start_t = time.perf_counter()
        res = runner.run(combined_stream)
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        n_meta = len(res.meta_alerts)
        arr = compute_arr(n_total, n_meta) if n_total > 0 else 0.0

        if rate == 0.0:
            baseline_arr = arr
            baseline_n_meta = n_meta
            degradation = 0.0
            absorption_count = 0
            absorption_rate = 100.0
        else:
            base = baseline_arr if baseline_arr is not None else arr
            degradation = float(base - arr)
            # Traceable absorption via MetaAlert.wazuh_alert_ids
            absorbed_count = 0
            for meta in res.meta_alerts:
                has_clean = any(not wid.startswith("noise_") for wid in meta.wazuh_alert_ids)
                if has_clean:
                    absorbed_count += sum(1 for wid in meta.wazuh_alert_ids if wid.startswith("noise_"))
            absorption_count = absorbed_count
            absorption_rate = (absorption_count / n_noise * 100.0) if n_noise > 0 else 100.0

        records.append({
            "noise_rate": rate,
            "n_noise": n_noise,
            "n_total": n_total,
            "n_meta": n_meta,
            "arr": arr,
            "arr_degradation": degradation,
            "noise_absorption_count": absorption_count,
            "noise_absorption_rate": absorption_rate,
            "execution_time_ms": exec_ms,
        })

    return NoiseRobustnessResult(summary_df=pd.DataFrame(records))
