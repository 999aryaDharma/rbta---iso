"""Meta-alert contract for aggregated temporal bucket representations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MetaAlert:
    """Immutable domain representation of an aggregated Meta-Alert produced by RBTA.

    Attributes
    ----------
    meta_id : int
        Unique identifier for the meta-alert.
    agent_id : str
        Agent identifier.
    agent_name : str
        Agent host name.
    rule_group_primary : str
        Primary rule group key of the RBTA bucket.
    start_time : datetime
        Timezone-aware start timestamp of the aggregation window.
    end_time : datetime
        Timezone-aware end timestamp of the aggregation window.
    alert_count : int
        Total number of raw alerts aggregated in this bucket (>= 1).
    max_severity : int
        Maximum Wazuh rule level observed in this bucket [0, 15].
    rule_id_distribution : Mapping[str, int]
        Distribution map of rule IDs to counts.
    severity_distribution : Mapping[int, int]
        Distribution map of severity levels to counts.
    mitre_tactics_unique : tuple[str, ...]
        Unique MITRE ATT&CK tactics seen across all alerts in the bucket.
    critical_mitre_present : bool
        Flag indicating if any critical MITRE tactic was triggered.
    agent_criticality : int
        Criticality score of the host asset [1, 4].
    wazuh_alert_ids : tuple[str, ...]
        Traceable tuple of all member Wazuh alert IDs.
    metadata : Mapping[str, Any]
        Audit metadata.
    """

    meta_id: int
    agent_id: str
    agent_name: str
    rule_group_primary: str
    start_time: datetime
    end_time: datetime
    alert_count: int
    max_severity: int
    rule_id_distribution: Mapping[str, int]
    severity_distribution: Mapping[int, int]
    mitre_tactics_unique: tuple[str, ...]
    critical_mitre_present: bool
    agent_criticality: int
    wazuh_alert_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.start_time, datetime) or not isinstance(self.end_time, datetime):
            raise TypeError("start_time and end_time must be datetime instances")

        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise ValueError("start_time and end_time must be timezone-aware")

        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) cannot be earlier than start_time ({self.start_time})")

        if self.alert_count < 1:
            raise ValueError(f"alert_count must be at least 1, got {self.alert_count}")

        if not (0 <= self.max_severity <= 15):
            raise ValueError(f"max_severity must be in range [0, 15], got {self.max_severity}")

        if not (1 <= self.agent_criticality <= 4):
            raise ValueError(f"agent_criticality must be in range [1, 4], got {self.agent_criticality}")

        if not isinstance(self.mitre_tactics_unique, tuple):
            object.__setattr__(self, "mitre_tactics_unique", tuple(self.mitre_tactics_unique))

        if not isinstance(self.wazuh_alert_ids, tuple):
            object.__setattr__(self, "wazuh_alert_ids", tuple(self.wazuh_alert_ids))

    @property
    def duration_sec(self) -> float:
        """Window duration in seconds."""
        return max(0.0, (self.end_time - self.start_time).total_seconds())
