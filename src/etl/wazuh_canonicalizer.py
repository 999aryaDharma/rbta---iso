"""Canonicalizer for converting raw Wazuh events and OpenSearch search hits into CanonicalRawAlert.

Supports:
- Raw Wazuh alert JSON (from alerts.json or REST API)
- OpenSearch search hits ({_index, _id, _source, sort})
- Nested MITRE (rule.mitre.tactic / technique / id)
- Flattened MITRE (rule.mitre_tactics / rule.mitre_techniques)
- Timezone-safe timestamp normalization to UTC
"""

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.config.domain import get_agent_criticality, resolve_primary_rule_group
from src.contracts.raw_alert import CanonicalRawAlert


class CanonicalizationError(ValueError):
    """Raised when raw alert data is missing mandatory fields or malformed."""
    pass


def parse_wazuh_timestamp(val: Any) -> datetime:
    """Parse any valid Wazuh timestamp representation into a timezone-aware UTC datetime.

    Parameters
    ----------
    val : Any
        Timestamp value (ISO8601 string, epoch number, or datetime).

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.

    Raises
    ------
    CanonicalizationError
        If timestamp is None, empty, or unparseable.
    """
    if val is None or val == "":
        raise CanonicalizationError("Event timestamp is required and cannot be empty")

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)

    if isinstance(val, (int, float)):
        # If timestamp is in milliseconds (epoch > 1e11)
        if val > 1e11:
            val = val / 1000.0
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception as exc:
            raise CanonicalizationError(f"Invalid numeric epoch timestamp '{val}': {exc}") from exc

    if isinstance(val, str):
        ts_str = val.strip()
        if not ts_str:
            raise CanonicalizationError("Event timestamp string is empty")

        # Handle trailing Z
        if ts_str.endswith("Z") or ts_str.endswith("z"):
            ts_str = ts_str[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        # Try standard common formatting fallbacks
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue

        raise CanonicalizationError(f"Could not parse timestamp string '{val}' into a valid datetime")

    raise CanonicalizationError(f"Unsupported timestamp type {type(val)}: '{val}'")


def extract_mitre_tactics(rule_dict: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract unique MITRE ATT&CK tactic names from a rule dictionary.

    Supports:
    - rule.mitre.tactic (list of strings, single string, or pipe-separated)
    - rule.mitre_tactics (list of strings)
    - rule.mitre_tactic (string or list)

    Parameters
    ----------
    rule_dict : Mapping[str, Any]
        Rule dictionary from Wazuh alert.

    Returns
    -------
    tuple[str, ...]
        Tuple of unique MITRE tactic names.
    """
    tactics: list[str] = []

    # Check nested rule.mitre
    mitre_block = rule_dict.get("mitre")
    if isinstance(mitre_block, dict):
        raw_tactic = mitre_block.get("tactic")
        if isinstance(raw_tactic, (list, tuple)):
            for item in raw_tactic:
                if item and str(item).strip():
                    tactics.append(str(item).strip())
        elif isinstance(raw_tactic, str) and raw_tactic.strip():
            for part in raw_tactic.split("|"):
                if part.strip():
                    tactics.append(part.strip())

    # Check flattened rule.mitre_tactics
    flat_tactics = rule_dict.get("mitre_tactics")
    if isinstance(flat_tactics, (list, tuple)):
        for item in flat_tactics:
            if item and str(item).strip():
                tactics.append(str(item).strip())
    elif isinstance(flat_tactics, str) and flat_tactics.strip():
        for part in flat_tactics.split("|"):
            if part.strip():
                tactics.append(part.strip())

    # Check flattened rule.mitre_tactic
    single_flat = rule_dict.get("mitre_tactic")
    if isinstance(single_flat, str) and single_flat.strip():
        for part in single_flat.split("|"):
            if part.strip():
                tactics.append(part.strip())
    elif isinstance(single_flat, (list, tuple)):
        for item in single_flat:
            if item and str(item).strip():
                tactics.append(str(item).strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return tuple(deduped)


def extract_source_ip(data_dict: Mapping[str, Any], raw_dict: Mapping[str, Any]) -> str | None:
    """Extract source IP from data.srcip or top-level srcip.

    Parameters
    ----------
    data_dict : Mapping[str, Any]
        Data payload dictionary.
    raw_dict : Mapping[str, Any]
        Raw alert dictionary.

    Returns
    -------
    str | None
        Normalized source IP address or None.
    """
    ip = data_dict.get("srcip") if isinstance(data_dict, dict) else None
    if not ip:
        ip = raw_dict.get("srcip")

    if ip is not None:
        ip_str = str(ip).strip()
        if ip_str and ip_str.lower() not in ("none", "nan", "null", ""):
            return ip_str

    return None


def canonicalize_wazuh_alert(event: Mapping[str, Any]) -> CanonicalRawAlert:
    """Canonicalize a raw Wazuh alert dictionary or OpenSearch search hit into CanonicalRawAlert.

    Parameters
    ----------
    event : Mapping[str, Any]
        Raw Wazuh alert or OpenSearch hit dictionary.

    Returns
    -------
    CanonicalRawAlert
        Normalized, validated, immutable canonical raw alert.

    Raises
    ------
    CanonicalizationError
        If mandatory fields are missing or schema is invalid.
    """
    if not isinstance(event, (dict, Mapping)):
        raise CanonicalizationError(f"Expected event dictionary/mapping, got {type(event)}")

    envelope_meta: dict[str, Any] = {}

    # Check for OpenSearch hit envelope
    if "_source" in event and isinstance(event["_source"], dict):
        alert_body = event["_source"]
        if "_index" in event:
            envelope_meta["source_index"] = event["_index"]
        if "_id" in event:
            envelope_meta["source_document_id"] = event["_id"]
        if "sort" in event:
            envelope_meta["source_sort"] = event["sort"]
    else:
        alert_body = event

    # 1. Wazuh Alert ID
    raw_id = alert_body.get("id")
    if raw_id is None or not str(raw_id).strip():
        raise CanonicalizationError("Wazuh alert must contain a valid top-level 'id'")
    wazuh_alert_id = str(raw_id).strip()

    # 2. Timestamp
    raw_ts = alert_body.get("timestamp")
    timestamp = parse_wazuh_timestamp(raw_ts)

    # 3. Rule Block
    rule = alert_body.get("rule")
    if not isinstance(rule, dict) or not rule:
        raise CanonicalizationError("Wazuh alert missing required 'rule' dictionary")

    rule_id_raw = rule.get("id")
    if rule_id_raw is None or not str(rule_id_raw).strip():
        raise CanonicalizationError("Wazuh alert rule missing required 'rule.id'")
    rule_id = str(rule_id_raw).strip()

    try:
        rule_level = int(rule.get("level", 0))
    except (ValueError, TypeError) as exc:
        raise CanonicalizationError(f"Invalid rule level '{rule.get('level')}': {exc}") from exc

    if not (0 <= rule_level <= 15):
        raise CanonicalizationError(f"Rule level {rule_level} outside valid Wazuh range [0, 15]")

    rule_groups = rule.get("groups")
    if isinstance(rule_groups, (list, tuple)):
        clean_groups = [str(g).strip() for g in rule_groups if str(g).strip()]
    elif isinstance(rule_groups, str) and rule_groups.strip():
        clean_groups = [g.strip() for g in rule_groups.split(",") if g.strip()]
    else:
        clean_groups = []

    rule_group_primary = resolve_primary_rule_group(clean_groups)

    # 4. Agent Block
    agent = alert_body.get("agent")
    if isinstance(agent, dict):
        agent_id = str(agent.get("id", "000")).strip()
        agent_name = str(agent.get("name", "")).strip()
    else:
        agent_id = "000"
        agent_name = ""

    # Fallback for manager / self alerts
    if not agent_name:
        manager = alert_body.get("manager")
        if isinstance(manager, dict) and manager.get("name"):
            agent_name = str(manager.get("name")).strip()
        else:
            agent_name = "unknown"

    agent_criticality = get_agent_criticality(agent_name=agent_name, agent_id=agent_id)

    # 5. MITRE Tactics
    mitre_tactics = extract_mitre_tactics(rule)

    # 6. Source IP
    data_block = alert_body.get("data", {})
    srcip = extract_source_ip(data_block if isinstance(data_block, dict) else {}, alert_body)

    # 7. Metadata preservation
    metadata: dict[str, Any] = {**envelope_meta}
    if "description" in rule:
        metadata["rule_description"] = rule["description"]
    metadata["rule_groups_all"] = clean_groups

    if "location" in alert_body:
        metadata["location"] = alert_body["location"]
    if "full_log" in alert_body:
        metadata["full_log"] = alert_body["full_log"]
    if "decoder" in alert_body:
        metadata["decoder"] = alert_body["decoder"]
    if "manager" in alert_body:
        metadata["manager"] = alert_body["manager"]
    if isinstance(data_block, dict) and data_block:
        metadata["data"] = data_block

    return CanonicalRawAlert(
        wazuh_alert_id=wazuh_alert_id,
        timestamp=timestamp,
        agent_id=agent_id,
        agent_name=agent_name,
        rule_group_primary=rule_group_primary,
        rule_level=rule_level,
        rule_id=rule_id,
        mitre_tactics=mitre_tactics,
        srcip=srcip,
        agent_criticality=agent_criticality,
        metadata=metadata,
    )
