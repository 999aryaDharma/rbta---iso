"""Canonical domain data contracts (DTOs) for RBTA + Isolation Forest research pipeline."""

from src.contracts.immutability import freeze_value
from src.contracts.raw_alert import CanonicalRawAlert
from src.contracts.meta_alert import MetaAlert
from src.contracts.scored_meta_alert import ScoredMetaAlert

__all__ = [
    "freeze_value",
    "CanonicalRawAlert",
    "MetaAlert",
    "ScoredMetaAlert",
]
