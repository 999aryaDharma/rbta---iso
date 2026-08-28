"""Canonical raw alert contract for normalized Wazuh event representation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CanonicalRawAlert:
    """Immutable domain representation of a normalized raw alert from Wazuh SIEM.

    Attributes
    ----------
    wazuh_alert_id : str
        Original Wazuh alert identifier (e.g. '1787895525.48425').
    timestamp : datetime
        Timezone-aware event timestamp (UTC normalized).
    agent_id : str
        Wazuh agent ID (e.g. '001', '000').
    agent_name : str
        Wazuh agent host name (e.g. 'soc-1', 'pusatkarir').
    rule_group_primary : str
        Authoritatively resolved primary rule group.
    rule_level : int
        Wazuh rule severity level in range [0, 15].
    rule_id : str
        Wazuh rule ID (e.g. '5501').
    mitre_tactics : tuple[str, ...]
        Tuple of unique MITRE ATT&CK tactics associated with this alert.
    srcip : str | None
        Source IP address if present, or None for local/HIDS events.
    agent_criticality : int
        Domain asset criticality score in range [1, 4].
    metadata : Mapping[str, Any]
        Audit and envelope metadata (e.g. OpenSearch _id, _index, sort, location, full_log).
    """

    wazuh_alert_id: str
    timestamp: datetime
    agent_id: str
    agent_name: str
    rule_group_primary: str
    rule_level: int
    rule_id: str
    mitre_tactics: tuple[str, ...]
    srcip: str | None
    agent_criticality: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.wazuh_alert_id or not str(self.wazuh_alert_id).strip():
            raise ValueError("wazuh_alert_id must be a non-empty string")

        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be a datetime instance, got {type(self.timestamp)}")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (tzinfo must not be None)")

        if not (0 <= self.rule_level <= 15):
            raise ValueError(f"rule_level must be between 0 and 15, got {self.rule_level}")

        if not (1 <= self.agent_criticality <= 4):
            raise ValueError(f"agent_criticality must be between 1 and 4, got {self.agent_criticality}")

        if not isinstance(self.mitre_tactics, tuple):
            object.__setattr__(self, "mitre_tactics", tuple(self.mitre_tactics))
