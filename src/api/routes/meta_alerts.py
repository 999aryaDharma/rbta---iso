from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.auth import get_api_key
from src.runtime.context_resolver import DashboardRuntimeResolver

router = APIRouter(prefix="/api/v1/meta-alerts", tags=["meta-alerts"])

SortByField = Literal["meta_id", "start_time", "end_time", "alert_count", "max_severity", "anomaly_score"]


def _get_resolver(request: Request) -> DashboardRuntimeResolver:
    return request.app.state.runtime_resolver


def _scored_alert_to_dict(scored: Any) -> Dict[str, Any]:
    return {
        "meta_id": scored.meta_id,
        "agent_id": scored.agent_id,
        "agent_name": scored.agent_name,
        "rule_group_primary": scored.rule_group_primary,
        "start_time": scored.start_time.isoformat() if hasattr(scored.start_time, "isoformat") else str(scored.start_time),
        "end_time": scored.end_time.isoformat() if hasattr(scored.end_time, "isoformat") else str(scored.end_time),
        "alert_count": scored.alert_count,
        "max_severity": scored.max_severity,
        "mitre_tactics": list(scored.mitre_tactics),
        "seven_features": dict(scored.seven_features),
        "raw_model_score": scored.raw_model_score,
        "anomaly_score": scored.anomaly_score,
        "threshold_used": scored.threshold_used,
        "decision": scored.decision,
        "action": scored.action,
        "escalate": scored.escalate,
        "model_version": scored.model_version,
        "feature_schema_version": scored.feature_schema_version,
        "score_calibration_version": scored.score_calibration_version,
        "source_alert_ids": list(scored.source_alert_ids),
        "metadata": dict(scored.metadata),
    }


@router.get("")
def list_meta_alerts(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=200, description="Items per page (max 200)"),
    decision: Optional[str] = Query(None, description="Filter by decision"),
    action: Optional[str] = Query(None, description="Filter by action (ESCALATE, DAILY_DIGEST, SUPPRESS)"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    search: Optional[str] = Query(None, description="Search across meta_id, rule_group, agent_name"),
    sort_by: SortByField = Query("end_time", description="Sort field allowlist"),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort direction"),
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, _, _ = resolver.resolve(run_id)
    history = list(service.finalized_history)

    filtered = []
    search_lower = search.strip().lower() if search else None

    for m in history:
        if decision and m.decision != decision:
            continue
        if action and m.action != action:
            continue
        if agent_id and m.agent_id != agent_id:
            continue
        if search_lower:
            match = (
                search_lower in str(m.meta_id)
                or search_lower in m.rule_group_primary.lower()
                or search_lower in m.agent_name.lower()
                or search_lower in m.agent_id.lower()
            )
            if not match:
                continue
        filtered.append(m)

    # Sort
    reverse = (sort_order == "desc")
    filtered.sort(key=lambda x: getattr(x, sort_by, 0), reverse=reverse)

    total = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = [_scored_alert_to_dict(m) for m in filtered[start_idx:end_idx]]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{meta_id}")
def get_meta_alert_detail(
    meta_id: int,
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, _, _ = resolver.resolve(run_id)
    for m in service.finalized_history:
        if m.meta_id == meta_id:
            return _scored_alert_to_dict(m)
    raise HTTPException(status_code=404, detail=f"MetaAlert #{meta_id} not found")


@router.get("/{meta_id}/trace")
def get_meta_alert_trace(
    meta_id: int,
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, _, _ = resolver.resolve(run_id)
    for m in service.finalized_history:
        if m.meta_id == meta_id:
            return {
                "meta_id": m.meta_id,
                "agent_id": m.agent_id,
                "rule_group": m.rule_group_primary,
                "source_alert_ids": list(m.source_alert_ids),
                "count": len(m.source_alert_ids),
                "model_version": m.model_version,
                "decision": m.decision,
                "action": m.action,
            }
    raise HTTPException(status_code=404, detail=f"MetaAlert #{meta_id} trace not found")


@router.get("/{meta_id}/raw-alerts")
def get_meta_alert_raw_alerts(
    meta_id: int,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page (max 200)"),
    search: Optional[str] = Query(None, description="Search query across alert ID, rule, description, IP, log"),
    rule_id: Optional[str] = Query(None, description="Filter by rule ID"),
    level_min: Optional[int] = Query(None, ge=0, le=15, description="Filter by minimum severity level"),
    level_max: Optional[int] = Query(None, ge=0, le=15, description="Filter by maximum severity level"),
    srcip: Optional[str] = Query(None, description="Filter by source IP"),
    mitre_tactic: Optional[str] = Query(None, description="Filter by MITRE tactic"),
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, evidence_store, _ = resolver.resolve(run_id)
    meta = None
    for m in service.finalized_history:
        if m.meta_id == meta_id:
            meta = m
            break

    if meta is None:
        raise HTTPException(status_code=404, detail=f"MetaAlert #{meta_id} not found")

    source_ids = list(meta.source_alert_ids)
    return evidence_store.get_meta_alert_raw_alerts(
        source_alert_ids=source_ids,
        meta_id=meta_id,
        page=page,
        page_size=page_size,
        search=search,
        rule_id=rule_id,
        level_min=level_min,
        level_max=level_max,
        srcip=srcip,
        mitre_tactic=mitre_tactic,
        redact=True,
    )
