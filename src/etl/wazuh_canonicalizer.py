"""Canonicalizer for converting raw Wazuh events and OpenSearch search hits into CanonicalRawAlert.

Supports:
- Raw Wazuh alert JSON (from alerts.json or REST API)
- OpenSearch search hits ({_index, _id, _source, sort})
- Nested MITRE (rule.mitre.tactic / technique / id)
- Flattened MITRE (rule.mitre_tactics / rule.mitre_techniques)
- Strict timezone normalization to UTC (rejection of naive timestamps)
- Case-insensitive MITRE tactic deduplication (preserving first encountered casing)
- Safe malformed rule group normalization
"""

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

from src.config.domain import get_agent_criticality, resolve_primary_rule_group
from src.contracts.raw_alert import CanonicalRawAlert

_OFFSET_REGEX = re.compile(r"([+-]\d{2}:?\d{2}|Z|z)$")


class CanonicalizationError(ValueError):
    """Raised when raw alert data is missing mandatory fields or malformed."""
    pass


def _clean_str_field(val: Any) -> str | None:
    """Clean a string field, returning None if value is None, empty, or 'none'/'null'."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.casefold() in ("none", "null"):
        return None
    return s


def parse_wazuh_timestamp(val: Any) -> datetime:
    """Parse a valid Wazuh timestamp representation into a timezone-aware UTC datetime.

    Strict Policy:
    - Timezone-aware datetime: converted to UTC.
    - Explicit 'Z' or offset (e.g. +08:00, +0000): converted to UTC.
    - Numeric epoch: converted to UTC.
    - Naive datetime or naive string: REJECTED with CanonicalizationError.

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
        If timestamp is missing, naive, or unparseable.
    """
    if val is None or val == "":
        raise CanonicalizationError("Event timestamp is required and cannot be empty")

    if isinstance(val, datetime):
        if val.tzinfo is None:
            raise CanonicalizationError("Event timestamp datetime object is naive; timezone-aware datetime is required")
        return val.astimezone(timezone.utc)

    if isinstance(val, (int, float)):
        # If timestamp is in milliseconds (epoch > 1e11)
        epoch_sec = val / 1000.0 if val > 1e11 else float(val)
        try:
            return datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
        except Exception as exc:
            raise CanonicalizationError(f"Invalid numeric epoch timestamp '{val}': {exc}") from exc

    if isinstance(val, str):
        ts_str = val.strip()
        if not ts_str:
            raise CanonicalizationError("Event timestamp string is empty")

        # Check for timezone offset or Z
        if not _OFFSET_REGEX.search(ts_str):
            raise CanonicalizationError(
                f"Invalid or naive timestamp '{val}'; explicit UTC 'Z' or timezone offset is required"
            )

        # Normalize trailing Z/z to standard +00:00 for fromisoformat
        if ts_str.endswith("Z") or ts_str.endswith("z"):
            ts_iso = ts_str[:-1] + "+00:00"
        else:
            ts_iso = ts_str

        try:
            dt = datetime.fromisoformat(ts_iso)
            if dt.tzinfo is None:
                raise CanonicalizationError(f"Parsed timestamp '{val}' has no tzinfo")
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        # Try strptime with standard offset formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
        ):
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    continue
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue

        raise CanonicalizationError(f"Could not parse timestamp string '{val}' into a valid datetime")

    raise CanonicalizationError(f"Unsupported timestamp type {type(val)}: '{val}'")


def extract_mitre_tactics(rule_dict: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract unique MITRE ATT&CK tactic names with case-insensitive deduplication.

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
        Tuple of unique MITRE tactic names preserving discovery order and first representation.
    """
    raw_candidates: list[str] = []

    # Check nested rule.mitre
    mitre_block = rule_dict.get("mitre")
    if isinstance(mitre_block, dict):
        raw_tactic = mitre_block.get("tactic")
        if isinstance(raw_tactic, (list, tuple)):
            for item in raw_tactic:
                clean = _clean_str_field(item)
                if clean:
                    raw_candidates.append(clean)
        elif isinstance(raw_tactic, str):
            for part in raw_tactic.split("|"):
                clean = _clean_str_field(part)
                if clean:
                    raw_candidates.append(clean)

    # Check flattened rule.mitre_tactics
    flat_tactics = rule_dict.get("mitre_tactics")
    if isinstance(flat_tactics, (list, tuple)):
        for item in flat_tactics:
            clean = _clean_str_field(item)
            if clean:
                raw_candidates.append(clean)
    elif isinstance(flat_tactics, str):
        for part in flat_tactics.split("|"):
            clean = _clean_str_field(part)
            if clean:
                raw_candidates.append(clean)

    # Check flattened rule.mitre_tactic
    single_flat = rule_dict.get("mitre_tactic")
    if isinstance(single_flat, str):
        for part in single_flat.split("|"):
            clean = _clean_str_field(part)
            if clean:
                raw_candidates.append(clean)
    elif isinstance(single_flat, (list, tuple)):
        for item in single_flat:
            clean = _clean_str_field(item)
            if clean:
                raw_candidates.append(clean)

    # Deduplicate case-insensitively while preserving first representation and discovery order
    seen_keys: set[str] = set()
    deduped: list[str] = []
    for t in raw_candidates:
        key = t.casefold()
        if key not in seen_keys:
            seen_keys.add(key)
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
    raw_ip = data_dict.get("srcip") if isinstance(data_dict, dict) else None
    if raw_ip is None:
        raw_ip = raw_dict.get("srcip")

    return _clean_str_field(raw_ip)


def extract_clean_rule_groups(rule_dict: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract valid cleaned rule groups, filtering out None, null, empty strings, and 'none'."""
    raw_groups = rule_dict.get("groups")
    if isinstance(raw_groups, (list, tuple)):
        items = raw_groups
    elif isinstance(raw_groups, str):
        items = raw_groups.split(",")
    else:
        items = []

    clean_groups: list[str] = []
    seen_groups: set[str] = set()
    for g in items:
        cleaned = _clean_str_field(g)
        if cleaned:
            key = cleaned.casefold()
            if key not in seen_groups:
                seen_groups.add(key)
                clean_groups.append(cleaned.lower())

    return tuple(clean_groups)


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

    # 1. Wazuh Alert ID — MUST come from alert_body.id (or _source.id), NEVER OpenSearch _id
    raw_id = alert_body.get("id")
    cleaned_id = _clean_str_field(raw_id)
    if cleaned_id is None:
        raise CanonicalizationError("Wazuh alert must contain a valid top-level 'id'")
    wazuh_alert_id = cleaned_id

    # 2. Timestamp
    raw_ts = alert_body.get("timestamp")
    timestamp = parse_wazuh_timestamp(raw_ts)

    # 3. Rule Block
    rule = alert_body.get("rule")
    if not isinstance(rule, dict) or not rule:
        raise CanonicalizationError("Wazuh alert missing required 'rule' dictionary")

    rule_id_raw = rule.get("id")
    cleaned_rule_id = _clean_str_field(rule_id_raw)
    if cleaned_rule_id is None:
        raise CanonicalizationError("Wazuh alert rule missing required 'rule.id'")
    rule_id = cleaned_rule_id

    # Strict rule.level validation
    raw_level = rule.get("level")
    if raw_level is None or isinstance(raw_level, bool):
        raise CanonicalizationError("Wazuh alert rule missing required 'rule.level'")

    try:
        rule_level = int(raw_level)
    except (ValueError, TypeError) as exc:
        raise CanonicalizationError(f"Invalid rule.level '{raw_level}': {exc}") from exc

    if not (0 <= rule_level <= 15):
        raise CanonicalizationError(f"Rule level {rule_level} outside valid Wazuh range [0, 15]")

    # Cleaned rule groups
    clean_groups = extract_clean_rule_groups(rule)
    rule_group_primary = resolve_primary_rule_group(clean_groups)

    # 4. Agent Block (deterministic null and whitespace normalization)
    agent = alert_body.get("agent")
    if isinstance(agent, dict):
        agent_id_clean = _clean_str_field(agent.get("id"))
        agent_name_clean = _clean_str_field(agent.get("name"))
        agent_id = agent_id_clean if agent_id_clean is not None else "000"
    else:
        agent_id = "000"
        agent_name_clean = None

    # Fallback to manager name if agent name is not present
    if agent_name_clean is not None:
        agent_name = agent_name_clean
    else:
        manager = alert_body.get("manager")
        manager_name_clean = _clean_str_field(manager.get("name")) if isinstance(manager, dict) else None
        if manager_name_clean is not None:
            agent_name = manager_name_clean
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
