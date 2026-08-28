"""Unit tests for ShuffleWebhookForwarder (Sprint 9)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.api.shuffle_adapter import ShuffleForwarderError, ShuffleWebhookForwarder


def make_scored_meta(meta_id: int) -> ScoredMetaAlert:
    return ScoredMetaAlert(
        meta_id=meta_id,
        agent_id="001",
        agent_name="soc-1",
        rule_group_primary="pam",
        start_time=datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 28, 10, 10, 0, tzinfo=timezone.utc),
        alert_count=5,
        max_severity=8,
        mitre_tactics=("Initial Access",),
        seven_features={},
        raw_model_score=0.72,
        anomaly_score=0.85,
        threshold_used=0.65,
        decision="CRITICAL",
        action="ESCALATE",
        escalate=True,
        model_version="v1",
        feature_schema_version="1.0",
        score_calibration_version="minmax-v1",
        source_alert_ids=("a1", "a2", "a3", "a4", "a5"),
    )


def test_shuffle_forwarder_sends_idempotent_event_header():
    """Forwarder sends payload with X-Event-ID header and webhook key."""
    forwarder = ShuffleWebhookForwarder(
        webhook_url="https://shuffle.campus.local/api/v1/hooks/rbta_hook",
        api_key="shuffle-token-123",
    )
    meta = make_scored_meta(42)

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.Session.post", return_value=mock_resp) as mock_post:
        success = forwarder.forward(meta)
        assert success is True

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["X-Event-ID"] == "rbta-meta-42"
        assert kwargs["headers"]["Authorization"] == "Bearer shuffle-token-123"
        assert kwargs["json"]["meta_id"] == 42
        assert kwargs["json"]["decision"] == "CRITICAL"
