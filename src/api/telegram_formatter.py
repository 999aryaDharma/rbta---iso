"""Presentation-only Markdown formatter for Telegram SOC notifications."""

from src.contracts.scored_meta_alert import ScoredMetaAlert


def format_telegram_alert(scored_meta: ScoredMetaAlert) -> str:
    """Format a ScoredMetaAlert into a high-visibility Markdown Telegram message.

    Parameters
    ----------
    scored_meta : ScoredMetaAlert
        Target scored meta-alert to format.

    Returns
    -------
    str
        Markdown formatted message text.
    """
    header_emoji = "🚨" if scored_meta.escalate else "ℹ️"
    tactics_str = ", ".join(scored_meta.mitre_tactics) if scored_meta.mitre_tactics else "None"

    return (
        f"{header_emoji} *SECURITY META-ALERT: {scored_meta.decision}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• *Meta ID:* `{scored_meta.meta_id}`\n"
        f"• *Agent:* {scored_meta.agent_name} (ID: `{scored_meta.agent_id}`)\n"
        f"• *Rule Group:* `{scored_meta.rule_group_primary}`\n"
        f"• *Aggregated Events:* {scored_meta.alert_count} raw alerts\n"
        f"• *Max Severity:* {scored_meta.max_severity}/15\n"
        f"• *MITRE Tactics:* {tactics_str}\n"
        f"• *Anomaly Score:* `{scored_meta.anomaly_score:.4f}` (Threshold: `{scored_meta.threshold_used:.4f}`)\n"
        f"• *Recommended Action:* `{scored_meta.action}`\n"
        f"• *Model Version:* `{scored_meta.model_version}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ Window: {scored_meta.start_time.isoformat()} ➔ {scored_meta.end_time.isoformat()}"
    )
