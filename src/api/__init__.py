"""Operational REST API, Shuffle SOAR integration, and Telegram formatting package."""

__all__ = [
    "ShuffleForwarderError",
    "ShuffleWebhookForwarder",
    "create_app",
    "format_telegram_alert",
]


def __getattr__(name: str):
    """Load public API helpers lazily to avoid runtime worker import cycles."""
    if name == "create_app":
        from src.api.app import create_app
        return create_app
    if name in {"ShuffleForwarderError", "ShuffleWebhookForwarder"}:
        from src.api.shuffle_adapter import ShuffleForwarderError, ShuffleWebhookForwarder
        return {"ShuffleForwarderError": ShuffleForwarderError, "ShuffleWebhookForwarder": ShuffleWebhookForwarder}[name]
    if name == "format_telegram_alert":
        from src.runtime.telegram_formatter import format_telegram_alert
        return format_telegram_alert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
