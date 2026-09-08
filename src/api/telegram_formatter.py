"""Canonical, escaped Telegram HTML formatter for MetaAlert notifications."""

from datetime import timezone
from html import escape
from zoneinfo import ZoneInfo

from src.contracts.scored_meta_alert import ScoredMetaAlert


def format_telegram_alert(scored_meta: ScoredMetaAlert, run_id: str = "live") -> str:
    """Format a scored MetaAlert as escaped Telegram HTML; no routing logic.

    Parameters
    ----------
    scored_meta : ScoredMetaAlert
        Target scored meta-alert to format.

    Returns
    -------
    str
        Markdown formatted message text.
    """
    wita = ZoneInfo("Asia/Makassar")
    window_start = scored_meta.start_time.astimezone(wita).strftime("%Y-%m-%d %H:%M:%S WITA")
    window_end = scored_meta.end_time.astimezone(wita).strftime("%Y-%m-%d %H:%M:%S WITA")
    tactics_str = ", ".join(scored_meta.mitre_tactics) if scored_meta.mitre_tactics else "Tidak ada"
    margin = float(scored_meta.anomaly_score) - float(scored_meta.threshold_used)

    return (
        f"<b>SECURITY META-ALERT: {escape(scored_meta.decision)}</b>\n"
        f"<b>Decision:</b> {escape(scored_meta.decision)} | <b>Action:</b> {escape(scored_meta.action)}\n"
        f"<b>Agent:</b> {escape(scored_meta.agent_name)} ({escape(scored_meta.agent_id)})\n"
        f"<b>Rule group:</b> <code>{escape(scored_meta.rule_group_primary)}</code>\n"
        f"<b>Waktu:</b> {window_start} → {window_end}\n"
        f"<b>Evidence:</b> {scored_meta.alert_count} | <b>Severity:</b> {scored_meta.max_severity}/15\n"
        f"<b>MITRE:</b> {escape(tactics_str)}\n"
        f"<b>Anomaly:</b> {scored_meta.anomaly_score:.4f} | <b>Threshold:</b> {scored_meta.threshold_used:.4f} | <b>Margin:</b> {margin:+.4f}\n"
        f"<b>Model:</b> <code>{escape(scored_meta.model_version)}</code> | <b>Run:</b> <code>{escape(run_id)}</code>\n"
        "<i>Anomaly score bukan bukti serangan; ini adalah prioritas triase.</i>"
    )
