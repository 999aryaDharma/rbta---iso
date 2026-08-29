from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import get_api_key
from src.runtime.context_resolver import DashboardRuntimeResolver
from src.runtime.observability import (
    get_dashboard_agents,
    get_dashboard_buckets,
    get_dashboard_integrations,
    get_dashboard_summary,
    get_dashboard_system,
    get_dashboard_timeseries,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def _get_resolver(request: Request) -> DashboardRuntimeResolver:
    return request.app.state.runtime_resolver


@router.get("/summary")
def dashboard_summary(
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, evidence_store, _ = resolver.resolve(run_id)
    return get_dashboard_summary(service, evidence_store)


@router.get("/agents")
def dashboard_agents(
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    service, _, _ = resolver.resolve(run_id)
    return get_dashboard_agents(service)


@router.get("/buckets")
def dashboard_buckets(
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    service, _, _ = resolver.resolve(run_id)
    return get_dashboard_buckets(service)


@router.get("/timeseries")
def dashboard_timeseries(
    window_hours: int = Query(24, ge=1, le=168, description="Time series lookback window in hours"),
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    service, evidence_store, _ = resolver.resolve(run_id)
    return get_dashboard_timeseries(service, evidence_store, window_hours=window_hours)


@router.get("/system")
def dashboard_system(
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    service, _, _ = resolver.resolve(run_id)
    return get_dashboard_system(service)


@router.get("/integrations")
def dashboard_integrations(
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return get_dashboard_integrations()
