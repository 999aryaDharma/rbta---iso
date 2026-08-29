"""FastAPI operational service application exposing REST ingress, outbox, and liveness."""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.registry import ModelRegistry
from src.runtime.ingress import CollectorIngressBoundary, IngressPayloadError
from src.runtime.service import LiveRBTAService, _serialize_scored_alert


def create_app(
    service: Optional[LiveRBTAService] = None,
    model_registry: Optional[ModelRegistry] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Factory creating configured FastAPI instance with dependency injection."""
    app = FastAPI(
        title="RBTA Security Analytics REST Service",
        version="1.0.0",
        description="Dual-mode operational runtime for Rule-Based Temporal Aggregation and Isolation Forest scoring.",
    )

    auth_key = api_key or os.getenv("RBTA_API_KEY")
    ingress_boundary = CollectorIngressBoundary(api_key=auth_key)

    def verify_auth(authorization: Optional[str] = Header(None)) -> None:
        if auth_key:
            expected = f"Bearer {auth_key}"
            if authorization != expected and authorization != auth_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing Authorization header",
                )

    @app.get("/health", tags=["Monitoring"])
    def health() -> Dict[str, str]:
        """Liveness check probe."""
        return {"status": "ok", "service": "rbta-security-analytics"}

    @app.get("/ready", tags=["Monitoring"])
    def ready() -> JSONResponse:
        """Readiness probe validating model artifacts and runtime state."""
        if model_registry is not None and service is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"ready": False, "reason": "Service not initialized"},
            )

        if model_registry is None:
            if service is None or service.scoring_pipeline is None:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"ready": False, "reason": "No active model bundle or scoring pipeline configured"},
                )
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"ready": True, "active_model_version": service.scoring_pipeline.metadata.get("model_version", "unknown")},
            )

        active_version = model_registry.get_active_version()
        if not active_version:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"ready": False, "reason": "No active model version published in registry"},
            )

        try:
            bundle = model_registry.load_bundle(active_version)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "ready": True,
                    "active_model_version": active_version,
                    "threshold_q3": bundle.threshold.q3,
                    "threshold": bundle.threshold.threshold,
                },
            )
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"ready": False, "reason": f"Failed loading active bundle: {exc}"},
            )

    @app.get("/runtime/stats", tags=["Monitoring"])
    def runtime_stats(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        """Retrieve live runtime aggregation statistics."""
        verify_auth(authorization)
        if service is None:
            return {"status": "uninitialized"}

        return {
            "seen_alerts_count": len(service.engine._seen_alert_ids),
            "active_buckets_count": len(service.engine._active_buckets),
            "outbox_depth": len(service.get_outbox()),
            "meta_id_counter": service.engine._meta_id_counter,
        }

    @app.post("/api/v1/alerts/ingest", tags=["Ingress"])
    def ingest_alert(
        payload: Dict[str, Any],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Ingest raw Wazuh alert, aggregate in RBTA, and trigger online scoring if bucket finalized."""
        verify_auth(authorization)

        try:
            res = ingress_boundary.process_incoming(payload, auth_header=authorization)
        except IngressPayloadError as exc:
            if "Unauthorized" in str(exc):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        if res.is_duplicate or res.canonical_alert is None:
            return {
                "status": "accepted",
                "alert_id": res.alert_id,
                "is_duplicate": True,
            }

        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized — cannot process alerts",
            )
        if res.canonical_alert.wazuh_alert_id in service.engine._seen_alert_ids:
            return {"status": "accepted", "alert_id": res.alert_id, "is_duplicate": True}
        service.ingest_alert(res.canonical_alert)

        return {
            "status": "accepted",
            "alert_id": res.alert_id,
            "is_duplicate": False,
        }

    @app.get("/api/v1/outbox", tags=["Outbox"])
    def get_outbox(authorization: Optional[str] = Header(None)) -> List[Dict[str, Any]]:
        """Retrieve unacknowledged scored meta-alerts from the durable outbox."""
        verify_auth(authorization)
        if service is None:
            return []

        outbox_metas = service.get_outbox()
        return [_serialize_scored_alert(m) for m in outbox_metas]

    @app.post("/api/v1/outbox/{meta_id}/ack", tags=["Outbox"])
    def acknowledge_outbox(
        meta_id: int,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Acknowledge and dequeue a scored meta-alert from the outbox."""
        verify_auth(authorization)
        if service is not None:
            service.acknowledge_outbox(meta_id)
        return {"status": "acknowledged", "meta_id": meta_id}

    @app.get("/api/v1/meta-alerts/{meta_id}", tags=["Outbox"])
    def get_meta_alert_detail(
        meta_id: int,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Retrieve detail and provenance trace for a specific meta-alert."""
        verify_auth(authorization)
        if service is None:
            raise HTTPException(status_code=404, detail="Service not initialized")

        # Search finalized history first (survives ACK)
        detail = service.get_meta_detail(meta_id) if hasattr(service, 'get_meta_detail') else None
        if detail is not None:
            return _serialize_scored_alert(detail)

        # Fallback to active outbox
        for item in service.get_outbox():
            if item.meta_id == meta_id:
                return _serialize_scored_alert(item)

        raise HTTPException(status_code=404, detail=f"MetaAlert {meta_id} not found")

    @app.get("/api/v1/meta-alerts/{meta_id}/trace", tags=["Outbox"])
    def get_meta_alert_trace(
        meta_id: int,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            raise HTTPException(status_code=404, detail="Service not initialized")

        detail = service.get_meta_detail(meta_id) if hasattr(service, 'get_meta_detail') else None
        if detail is None:
            for item in service.get_outbox():
                if item.meta_id == meta_id:
                    detail = item
                    break

        if detail is None:
            raise HTTPException(status_code=404, detail=f"MetaAlert {meta_id} not found")

        return {
            "meta_id": detail.meta_id,
            "source_alert_ids": list(detail.source_alert_ids),
            "agent_id": detail.agent_id,
            "rule_group_primary": detail.rule_group_primary,
            "decision": detail.decision,
            "action": detail.action,
            "model_version": detail.model_version,
        }

    return app
