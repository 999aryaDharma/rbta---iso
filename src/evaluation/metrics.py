"""Evaluation metrics module for Alert Reduction Rate (ARR)."""


class MetricsError(ValueError):
    """Raised when metric calculation encounters invalid inputs."""
    pass


def compute_arr(n_raw: int, n_meta: int) -> float:
    """Calculate Alert Reduction Rate (ARR).

    Formula:
    ARR = ((N_raw - N_meta) / N_raw) * 100.0%

    Parameters
    ----------
    n_raw : int
        Total number of raw alerts.
    n_meta : int
        Total number of aggregated meta-alerts.

    Returns
    -------
    float
        Alert reduction rate as a percentage [0.0, 100.0].

    Raises
    ------
    MetricsError
        If n_raw <= 0, n_meta < 0, or n_meta > n_raw.
    """
    if n_raw <= 0:
        raise MetricsError(f"n_raw must be positive, got {n_raw}")
    if n_meta < 0:
        raise MetricsError(f"n_meta cannot be negative, got {n_meta}")
    if n_meta > n_raw:
        raise MetricsError(f"n_meta cannot exceed n_raw ({n_meta} > {n_raw})")

    return float((n_raw - n_meta) / n_raw * 100.0)
