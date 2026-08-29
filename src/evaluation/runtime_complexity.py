"""Empirical runtime complexity and throughput evaluation module."""

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.contracts.raw_alert import CanonicalRawAlert
from src.runners.batch_runner import BatchResearchRunner


RUNTIME_EVALUATION_SUBSETS: int = 8


@dataclass(frozen=True)
class RuntimeComplexityResult:
    """Outcome of empirical runtime complexity evaluation."""

    subset_df: pd.DataFrame
    slope: float
    intercept: float
    r_squared: float
    mean_throughput: float
    throughput_variation: float


def run_runtime_complexity_evaluation(
    alerts: Iterable[CanonicalRawAlert],
    n_subsets: int = RUNTIME_EVALUATION_SUBSETS,
    delta_t: timedelta = timedelta(minutes=15),
) -> RuntimeComplexityResult:
    """Measure RBTA throughput across increasing data scale subsets and fit linear regression.

    Parameters
    ----------
    alerts : Iterable[CanonicalRawAlert]
        Evaluation dataset.
    n_subsets : int
        Number of scaling steps (default 8).
    delta_t : timedelta
        Aggregation window.

    Returns
    -------
    RuntimeComplexityResult
        Subset measurements, regression parameters (slope, intercept, R^2), and throughput statistics.
    """
    sorted_alerts = sorted(list(alerts), key=lambda a: a.timestamp)
    total_len = len(sorted_alerts)

    subset_fractions = np.linspace(1.0 / n_subsets, 1.0, n_subsets)
    records: List[Dict[str, Any]] = []

    for frac in subset_fractions:
        k = max(1, int(round(total_len * frac)))
        subset = sorted_alerts[:k]

        runner = BatchResearchRunner(base_delta_t=delta_t, adaptive=True)
        start_t = time.perf_counter()
        res = runner.run(subset)
        exec_ms = max(0.001, (time.perf_counter() - start_t) * 1000.0)

        throughput = len(subset) / exec_ms

        records.append({
            "n_alerts": len(subset),
            "n_meta": len(res.meta_alerts),
            "execution_time_ms": exec_ms,
            "throughput_alerts_per_ms": throughput,
        })

    df = pd.DataFrame(records)

    # Linear regression: n_alerts -> execution_time_ms
    x = df["n_alerts"].values
    y = df["execution_time_ms"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = float(r_value ** 2)

    throughputs = df["throughput_alerts_per_ms"].values
    mean_thr = float(np.mean(throughputs))
    std_thr = float(np.std(throughputs))

    return RuntimeComplexityResult(
        subset_df=df,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        mean_throughput=mean_thr,
        throughput_variation=std_thr,
    )
