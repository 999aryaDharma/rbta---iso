"""Unit tests for Decision Matrix and False Positive Gate (Sprint 4)."""
import pytest
from src.model.decision import evaluate_decision


def test_decision_matrix_four_quadrants():
    """Verify the 4 standard decision matrix quadrants when False Positive Gate does not trigger."""
    threshold = 0.70

    # 1. Anomaly High + Severity High -> CRITICAL -> ESCALATE
    decision, action, escalate = evaluate_decision(
        anomaly_score=0.85, threshold=threshold, max_severity=10, alert_count=10, mitre_tactic_count=2
    )
    assert decision == "CRITICAL"
    assert action == "ESCALATE"
    assert escalate is True

    # 2. Anomaly High + Severity Low (with mitre tactics or count >= 5) -> SUSPICIOUS -> ESCALATE
    decision, action, escalate = evaluate_decision(
        anomaly_score=0.85, threshold=threshold, max_severity=4, alert_count=10, mitre_tactic_count=1
    )
    assert decision == "SUSPICIOUS"
    assert action == "ESCALATE"
    assert escalate is True

    # 3. Anomaly Low + Severity High -> NOISE_HIGH -> DAILY_DIGEST
    decision, action, escalate = evaluate_decision(
        anomaly_score=0.50, threshold=threshold, max_severity=8, alert_count=10, mitre_tactic_count=0
    )
    assert decision == "NOISE_HIGH"
    assert action == "DAILY_DIGEST"
    assert escalate is False

    # 4. Anomaly Low + Severity Low -> NOISE -> SUPPRESS
    decision, action, escalate = evaluate_decision(
        anomaly_score=0.40, threshold=threshold, max_severity=3, alert_count=2, mitre_tactic_count=0
    )
    assert decision == "NOISE"
    assert action == "SUPPRESS"
    assert escalate is False


def test_false_positive_gate_contextual_anomaly():
    """If anomaly_score >= threshold BUT max_severity < 7 AND alert_count < 5 AND mitre_tactic_count == 0,
    it must be classified as CONTEXTUAL_ANOMALY -> SUPPRESS.
    """
    threshold = 0.70

    decision, action, escalate = evaluate_decision(
        anomaly_score=0.90,  # anomaly high
        threshold=threshold,
        max_severity=4,       # severity < 7
        alert_count=2,        # count < 5
        mitre_tactic_count=0, # no mitre tactics
    )
    assert decision == "CONTEXTUAL_ANOMALY"
    assert action == "SUPPRESS"
    assert escalate is False


def test_false_positive_gate_mitre_presence_prevents_suppression():
    """If mitre_tactic_count > 0, False Positive Gate does NOT trigger even if count < 5 and severity < 7."""
    threshold = 0.70

    decision, action, escalate = evaluate_decision(
        anomaly_score=0.90,
        threshold=threshold,
        max_severity=4,
        alert_count=2,
        mitre_tactic_count=1,  # MITRE present -> bypasses FP gate
    )
    assert decision == "SUSPICIOUS"
    assert action == "ESCALATE"
    assert escalate is True
