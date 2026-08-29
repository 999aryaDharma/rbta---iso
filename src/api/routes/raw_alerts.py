from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.auth import get_api_key
from src.runtime.context_resolver import DashboardRuntimeResolver

router = APIRouter(prefix="/api/v1/raw-alerts", tags=["raw-alerts"])


def _get_resolver(request: Request) -> DashboardRuntimeResolver:
    return request.app.state.runtime_resolver


@router.get("/{wazuh_alert_id}")
def get_raw_alert_detail(
    wazuh_alert_id: str,
    run_id: Optional[str] = Query(None, description="Optional replay run context ID"),
    resolver: DashboardRuntimeResolver = Depends(_get_resolver),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    _, evidence_store, _ = resolver.resolve(run_id)
    alert = evidence_store.get(wazuh_alert_id, redact=True)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Raw alert evidence '{wazuh_alert_id}' not found")
    return alert
