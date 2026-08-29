"""Decision Matrix and False Positive Gate evaluation module."""

from typing import Tuple


def evaluate_decision(
    anomaly_score: float,
    threshold: float,
    max_severity: int,
    alert_count: int,
    mitre_tactic_count: int,
) -> Tuple[str, str, bool]:
    """Evaluate final operational decision and action for a scored meta-alert.

    Applies the False Positive Gate and the 4-quadrant Decision Matrix:

    False Positive Gate:
        if max_severity < 7 and alert_count < 5 and mitre_tactic_count == 0 and anomaly_score >= threshold:
            decision = CONTEXTUAL_ANOMALY
            action = SUPPRESS
            escalate = False

    Decision Matrix:
        anomaly_high (score >= threshold) & severity_high (max_severity >= 7):
            decision = CRITICAL, action = ESCALATE, escalate = True
        anomaly_high & severity_low:
            decision = SUSPICIOUS, action = ESCALATE, escalate = True
        anomaly_low & severity_high:
            decision = NOISE_HIGH, action = DAILY_DIGEST, escalate = False
        anomaly_low & severity_low:
            decision = NOISE, action = SUPPRESS, escalate = False

    Parameters
    ----------
    anomaly_score : float
        Calibrated anomaly score.
    threshold : float
        Active model Tukey threshold.
    max_severity : int
        Maximum severity level in bucket.
    alert_count : int
        Number of raw alerts in bucket.
    mitre_tactic_count : int
        Number of unique MITRE tactics in bucket.

    Returns
    -------
    Tuple[str, str, bool]
        (decision, action, escalate)
    """
    anomaly_high = anomaly_score >= threshold
    severity_high = max_severity >= 7

    # False Positive Gate Check
    if anomaly_high:
        if max_severity < 7 and alert_count < 5 and mitre_tactic_count == 0:
            return "CONTEXTUAL_ANOMALY", "SUPPRESS", False

        if severity_high:
            return "CRITICAL", "ESCALATE", True
        else:
            return "SUSPICIOUS", "ESCALATE", True
    else:
        if severity_high:
            return "NOISE_HIGH", "DAILY_DIGEST", False
        else:
            return "NOISE", "SUPPRESS", False
