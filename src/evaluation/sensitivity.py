"""Delta-t sensitivity analysis across eight static window durations."""

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.metrics import compute_arr
from src.runners.batch_runner import BatchResearchRunner

SENSITIVITY_DELTA_T_MINUTES: Tuple[int, ...] = (1, 5, 10, 15, 20, 30, 45, 60)


@dataclass(frozen=True)
class SensitivityResult:
    """Outcome of Delta-t sensitivity evaluation."""

    summary_df: pd.DataFrame
    recommended_elbow_delta_t: int


def _find_elbow_point(x_vals: Sequence[float], y_vals: Sequence[float]) -> int:
    """Identify optimal elbow point using maximum perpendicular distance to secant line."""
    if len(x_vals) < 3:
        return int(x_vals[0])

    p1 = np.array([x_vals[0], y_vals[0]])
    p2 = np.array([x_vals[-1], y_vals[-1]])

    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return int(x_vals[0])

    distances = []
    for x, y in zip(x_vals, y_vals):
        p = np.array([x, y])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / line_len
        distances.append(d)

    best_idx = int(np.argmax(distances))
    return int(x_vals[best_idx])


def run_delta_t_sensitivity_analysis(
    alerts: Iterable[CanonicalRawAlert],
    delta_t_values: Sequence[int] = SENSITIVITY_DELTA_T_MINUTES,
) -> SensitivityResult:
    """Execute RBTA batch aggregation across static delta-t values with adaptive ETW disabled.

    Parameters
    ----------
    alerts : Iterable[CanonicalRawAlert]
        Evaluation alerts dataset.
    delta_t_values : Sequence[int]
        Delta-t values in minutes to evaluate (default: 1, 5, 10, 15, 20, 30, 45, 60).

    Returns
    -------
    SensitivityResult
        Summary DataFrame and computed elbow recommendation.
    """
    alert_list = list(alerts)
    n_raw = len(alert_list)

    records: List[Dict[str, Any]] = []

    for dt_min in delta_t_values:
        runner = BatchResearchRunner(
            base_delta_t=timedelta(minutes=dt_min),
            adaptive=False,  # Strict ceteris paribus
        )

        start_time = time.perf_counter()
        result = runner.run(alert_list)
        exec_ms = (time.perf_counter() - start_time) * 1000.0

        n_meta = len(result.meta_alerts)
        arr = compute_arr(n_raw, n_meta) if n_raw > 0 else 0.0

        records.append({
            "delta_t_min": dt_min,
            "n_raw": n_raw,
            "n_meta": n_meta,
            "arr": arr,
            "execution_time_ms": exec_ms,
        })

    df = pd.DataFrame(records)

    # Compute elbow point from ARR curve
    x_vals = [r["delta_t_min"] for r in records]
    y_vals = [r["arr"] for r in records]
    elbow_dt = _find_elbow_point(x_vals, y_vals)

    return SensitivityResult(
        summary_df=df,
        recommended_elbow_delta_t=elbow_dt,
    )
