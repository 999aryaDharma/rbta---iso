"""Domain configuration and authoritative constants for RBTA + Isolation Forest research.

Single source of truth for:
- AGENT_CRITICALITY
- GROUP_SEVERITY_WEIGHT
- CRITICAL_MITRE_TACTICS
"""

from typing import Sequence

# ── Agent Criticality (Scale 1-4: Low=1, Medium=2, High=3, Critical=4) ────────
# Domain asset criticality mapping for UPT TIK INSTIKI infrastructure.
AGENT_CRITICALITY: dict[str, int] = {
    "soc-1": 1,
    "pusatkarir": 3,
    "dfir-iris": 4,
    "siput": 2,
    "proxy-manager": 3,
    "e-kuesioner": 2,
    "sads": 3,
    "dvwa": 1,
    "wazuh-soc": 1,
}

DEFAULT_AGENT_CRITICALITY: int = 1

CRITICALITY_LABEL_TO_SCORE: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "unknown": 1,
}

CRITICALITY_SCORE_TO_LABEL: dict[int, str] = {
    4: "critical",
    3: "high",
    2: "medium",
    1: "low",
}


def get_agent_criticality(agent_name: str | None, agent_id: str | None = None) -> int:
    """Resolve agent criticality score (1-4) from agent name or ID.

    Parameters
    ----------
    agent_name : str | None
        Name of the agent (e.g. 'dfir-iris', 'pusatkarir').
    agent_id : str | None, optional
        ID of the agent.

    Returns
    -------
    int
        Criticality score in range [1, 4].
    """
    if agent_name:
        clean_name = str(agent_name).strip().lower()
        if clean_name in AGENT_CRITICALITY:
            return AGENT_CRITICALITY[clean_name]

    if agent_id:
        clean_id = str(agent_id).strip().lower()
        if clean_id in AGENT_CRITICALITY:
            return AGENT_CRITICALITY[clean_id]

    return DEFAULT_AGENT_CRITICALITY


# ── Rule Group Severity Weights ───────────────────────────────────────────────
# Semantic rule group ordering derived from average severity and domain analysis.
GROUP_SEVERITY_WEIGHT: dict[str, int] = {
    "attack": 10,
    "sql_injection": 10,
    "authentication_failed": 9,
    "access_control": 8,
    "pam": 7,
    "web": 7,
    "virus": 7,
    "nginx": 6,
    "audit": 6,
    "clamd": 6,
    "accesslog": 5,
    "system_error": 5,
    "audit_selinux": 5,
    "syscheck": 5,
    "syscheck_file": 5,
    "syscheck_entry_modified": 5,
    "syscheck_entry_added": 4,
    "authentication_success": 3,
    "freshclam": 3,
    "syslog": 3,
    "rootcheck": 2,
    "ossec": 1,
}

DEFAULT_RULE_GROUP_WEIGHT: int = 2


def resolve_primary_rule_group(groups: Sequence[str] | None) -> str:
    """Select primary rule group using authoritative severity weights.

    If multiple groups exist, selects the group with the highest severity weight.
    If weights are tied, preserves the first occurrence in Wazuh rule order.

    Parameters
    ----------
    groups : Sequence[str] | None
        List of rule group names from Wazuh rule definition.

    Returns
    -------
    str
        Primary rule group name.
    """
    if not groups:
        return "unknown"

    clean_groups = [str(g).strip().lower() for g in groups if str(g).strip()]
    if not clean_groups:
        return "unknown"

    return max(clean_groups, key=lambda g: GROUP_SEVERITY_WEIGHT.get(g, DEFAULT_RULE_GROUP_WEIGHT))


# ── Critical MITRE ATT&CK Tactics ─────────────────────────────────────────────
# Authoritative set of high-risk tactics that elevate threat significance.
CRITICAL_MITRE_TACTICS: frozenset[str] = frozenset({
    "Execution",
    "Lateral Movement",
    "Credential Access",
    "Exfiltration",
    "Privilege Escalation",
    "Defense Evasion",
})

_CRITICAL_MITRE_TACTICS_LOWER: frozenset[str] = frozenset(
    t.lower() for t in CRITICAL_MITRE_TACTICS
)


def has_critical_mitre_tactic(tactics: Sequence[str] | None) -> bool:
    """Check whether any tactic in the sequence belongs to CRITICAL_MITRE_TACTICS.

    Parameters
    ----------
    tactics : Sequence[str] | None
        List or tuple of MITRE tactic names.

    Returns
    -------
    bool
        True if at least one critical tactic is present, False otherwise.
    """
    if not tactics:
        return False
    return any(str(t).strip().lower() in _CRITICAL_MITRE_TACTICS_LOWER for t in tactics if t)
