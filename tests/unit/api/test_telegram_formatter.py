"""Unit tests for Telegram notification message formatting (Sprint 9)."""
from datetime import datetime, timezone
import pytest

from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.api.telegram_formatter import format_telegram_alert


def test_telegram_formatter_presentation_only():
    """Formatter generates structured Markdown message without performing calculations."""
    meta = ScoredMetaAlert(
        meta_id=99,
        agent_id="001",
        agent_name="db-prod",
        rule_group_primary="sql_injection",
        start_time=datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 28, 10, 15, 0, tzinfo=timezone.utc),
        alert_count=25,
        max_severity=12,
        mitre_tactics=("Initial Access", "Impact"),
        seven_features={},
        raw_model_score=0.88,
        anomaly_score=0.95,
        threshold_used=0.68,
        decision="CRITICAL",
        action="ESCALATE",
        escalate=True,
        model_version="rbta-v1.0",
        feature_schema_version="1.0",
        score_calibration_version="minmax-v1",
        source_alert_ids=("a1", "a2"),
    )

    msg = format_telegram_alert(meta)

    assert "🚨 *SECURITY META-ALERT: CRITICAL*" in msg
    assert "db-prod (ID: `001`)" in msg
    assert "sql_injection" in msg
    assert "25 raw alerts" in msg
    assert "0.9500" in msg
    assert "0.6800" in msg
    assert "Initial Access, Impact" in msg
