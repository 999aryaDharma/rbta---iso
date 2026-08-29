"""Scored meta-alert contract for Isolation Forest inference and operational outbox."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from src.contracts.immutability import freeze_value

VALID_DECISIONS: frozenset[str] = frozenset({
    "CRITICAL",
    "SUSPICIOUS",
    "NOISE_HIGH",
    "NOISE",
    "CONTEXTUAL_ANOMALY",
})

VALID_ACTIONS: frozenset[str] = frozenset({
    "ESCALATE",
    "DAILY_DIGEST",
    "SUPPRESS",
})


@dataclass(frozen=True, slots=True)
class ScoredMetaAlert:
    """Immutable domain representation of a scored and prioritized Meta-Alert.

    Attributes
    ----------
    meta_id : int
        Meta-alert ID.
    agent_id : str
        Agent identifier.
    agent_name : str
        Agent host name.
    rule_group_primary : str
        Primary rule group.
    start_time : datetime
        Timezone-aware start timestamp.
    end_time : datetime
        Timezone-aware end timestamp.
    alert_count : int
        Number of raw alerts aggregated.
    max_severity : int
        Maximum rule severity.
    mitre_tactics : tuple[str, ...]
        Unique MITRE ATT&CK tactics.
    seven_features : Mapping[str, float]
        Recursively immutable dictionary mapping feature name to normalized value for the canonical 7 features.
    raw_model_score : float
        Oriented raw anomaly score from Isolation Forest.
    anomaly_score : float
        Calibrated stream-safe anomaly score [calibrated against reference].
    threshold_used : float
        Tukey IQR threshold used for decision.
    decision : str
        Decision Matrix quadrant or contextual anomaly classification.
    action : str
        Action recommendation ('ESCALATE', 'DAILY_DIGEST', 'SUPPRESS').
    escalate : bool
        Flag indicating if action is ESCALATE.
    model_version : str
        Version identifier of the loaded Isolation Forest model artifact.
    feature_schema_version : str
        Version of the feature schema.
    score_calibration_version : str
        Version of the score calibration transform.
    source_alert_ids : tuple[str, ...]
        Traceable tuple of all member Wazuh alert IDs.
    metadata : Mapping[str, Any]
        Recursively immutable audit metadata.
    """

    meta_id: int
    agent_id: str
    agent_name: str
    rule_group_primary: str
    start_time: datetime
    end_time: datetime
    alert_count: int
    max_severity: int
    mitre_tactics: tuple[str, ...]
    seven_features: Mapping[str, float]
    raw_model_score: float
    anomaly_score: float
    threshold_used: float
    decision: str
    action: str
    escalate: bool
    model_version: str
    feature_schema_version: str
    score_calibration_version: str
    source_alert_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision '{self.decision}'. Allowed: {sorted(VALID_DECISIONS)}")

        if self.action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{self.action}'. Allowed: {sorted(VALID_ACTIONS)}")

        if not isinstance(self.mitre_tactics, tuple):
            object.__setattr__(self, "mitre_tactics", tuple(self.mitre_tactics))

        if not isinstance(self.source_alert_ids, tuple):
            object.__setattr__(self, "source_alert_ids", tuple(self.source_alert_ids))

        object.__setattr__(self, "seven_features", freeze_value(self.seven_features))
        object.__setattr__(self, "metadata", freeze_value(self.metadata))
