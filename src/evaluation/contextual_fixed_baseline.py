"""Context-aware fixed tumbling baseline for RBTA ablation."""

from datetime import timedelta
from typing import Iterable

from src.contracts.raw_alert import CanonicalRawAlert
from src.evaluation.fixed_window_baseline import FixedWindowResult, _run_tumbling_window_baseline


def run_contextual_fixed_window_baseline(
    alerts: Iterable[CanonicalRawAlert],
    window_duration: timedelta = timedelta(minutes=15),
) -> FixedWindowResult:
    """Aggregate by calendar window plus the RBTA contextual key."""
    return _run_tumbling_window_baseline(alerts, window_duration, contextual=True)
