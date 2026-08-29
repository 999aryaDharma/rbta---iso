from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import auth, dashboard, meta_alerts, raw_alerts, replay
from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.model.registry import ModelRegistry
from src.runtime.context_resolver import DashboardRuntimeResolver
from src.runtime.ingress import CollectorIngressBoundary, IngressPayloadError
from src.runtime.raw_evidence import RawAlertEvidenceStore
from src.runtime.replay_controller import ReplayController
from src.runtime.service import LiveRBTAService, _serialize_scored_alert


def create_app(
    service: Optional[LiveRBTAService] = None,
    model_registry: Optional[ModelRegistry] = None,
    api_key: Optional[str] = None,
    raw_evidence_store: Optional[RawAlertEvidenceStore] = None,
    replay_controller: Optional[ReplayController] = None,
) -> FastAPI:
    """Factory creating configured FastAPI instance with dependency injection and modular routers."""
    app = FastAPI(
        title="RBTA Security Analytics REST Service",
        version="1.0.0",
        description="Dual-mode operational runtime for Rule-Based Temporal Aggregation and Isolation Forest scoring.",
    )

    auth_key = api_key or os.getenv("RBTA_API_KEY")
    ingress_boundary = CollectorIngressBoundary(api_key=auth_key)

    # Initialize runtime resolver for Live vs. Replay Run data isolation
    scoring_pipe = service.scoring_pipeline if service else None
    if scoring_pipe is None and replay_controller:
        scoring_pipe = replay_controller.scoring_pipeline

    runtime_resolver = DashboardRuntimeResolver(
        live_service=service,
        live_evidence_store=raw_evidence_store or RawAlertEvidenceStore(),
        replay_controller=replay_controller or ReplayController(scoring_pipeline=scoring_pipe),
        scoring_pipeline=scoring_pipe,
    )

    app.state.runtime_resolver = runtime_resolver
    app.state.raw_evidence_store = raw_evidence_store or runtime_resolver.live_evidence_store
    app.state.replay_controller = replay_controller or runtime_resolver.replay_controller
    app.state.auth_key = auth_key

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

            # Extract version from metadata or bundle
            version = "unknown"
            pipe = service.scoring_pipeline
            if hasattr(pipe, "metadata") and isinstance(pipe.metadata, dict):
                version = pipe.metadata.get("model_version", "unknown")
            elif hasattr(pipe, "bundle") and pipe.bundle is not None:
                version = getattr(pipe.bundle, "model_version", "unknown")

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"ready": True, "active_model_version": version},
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
                    "threshold_q3": bundle.threshold.q3 if hasattr(bundle, "threshold") else 0.0,
                    "threshold": bundle.threshold.threshold if hasattr(bundle, "threshold") else 0.0,
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

        seen_cnt = len(service.engine._seen_alert_ids) if hasattr(service.engine, "_seen_alert_ids") else 0
        buckets_cnt = len(service.engine.snapshot_buckets())

        return {
            "status": "ready",
            "active_buckets": buckets_cnt,
            "active_buckets_count": buckets_cnt,
            "temporal_agents": len(service.engine.snapshot_agents()),
            "seen_alerts_count": seen_cnt,
            "outbox_count": len(service.outbox),
            "finalized_history_count": len(service.finalized_history),
            "pending_scoring_count": len(service.pending_scoring),
            "raw_evidence_count": app.state.raw_evidence_store.count() if app.state.raw_evidence_store else 0,
        }
    @app.post("/ingest/wazuh", tags=["Ingress"])
    def ingest_wazuh(
        payload: Dict[str, Any],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Ingest raw Wazuh log item via boundary canonicalization."""
        verify_auth(authorization)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Runtime service uninitialized",
            )
        try:
            canonical_alert = ingress_boundary.ingest(payload, authorization=authorization)
        except IngressPayloadError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        scored_alerts = service.ingest_alert(canonical_alert)
        return {
            "status": "accepted",
            "wazuh_alert_id": canonical_alert.wazuh_alert_id,
            "generated_meta_alerts": len(scored_alerts),
            "meta_alerts": [_serialize_scored_alert(m) for m in scored_alerts],
        }

    @app.post("/api/v1/alerts/ingest", tags=["Ingress"])
    def ingest_alert_v1(
        payload: Dict[str, Any],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Ingest raw alert compatibility endpoint."""
        verify_auth(authorization)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Runtime service uninitialized",
            )
        try:
            canonical_alert = ingress_boundary.ingest(payload, authorization=authorization)
        except IngressPayloadError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        is_dup = hasattr(service.engine, "_seen_alert_ids") and (canonical_alert.wazuh_alert_id in service.engine._seen_alert_ids)
        service.ingest_alert(canonical_alert)
        return {
            "status": "accepted",
            "alert_id": canonical_alert.wazuh_alert_id,
            "is_duplicate": is_dup,
        }

    @app.get("/outbox/pending", tags=["Outbox"])
    def get_pending_outbox(
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Retrieve uncommitted scored meta-alerts waiting for external dispatch."""
        verify_auth(authorization)
        if service is None:
            return {"count": 0, "alerts": []}

        alerts = [_serialize_scored_alert(m) for m in service.outbox]
        return {"count": len(alerts), "alerts": alerts}

    @app.get("/api/v1/outbox", tags=["Outbox"])
    def get_outbox_v1(
        authorization: Optional[str] = Header(None),
    ) -> List[Dict[str, Any]]:
        """Retrieve uncommitted outbox alerts list directly."""
        verify_auth(authorization)
        if service is None:
            return []
        return [_serialize_scored_alert(m) for m in service.outbox]

    @app.post("/api/v1/outbox/{meta_id}/ack", tags=["Outbox"])
    def ack_outbox_v1(
        meta_id: int,
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Acknowledge dispatch of single meta alert."""
        verify_auth(authorization)
        if service is None:
            raise HTTPException(status_code=503, detail="Service uninitialized")
        service.commit_outbox([meta_id])
        return {"status": "acknowledged", "meta_id": meta_id}

    @app.post("/outbox/commit", tags=["Outbox"])
    def commit_outbox(
        meta_ids: List[int],
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """Acknowledge successful dispatch of meta-alerts."""
        verify_auth(authorization)
        if service is None:
            return {"committed": 0, "remaining": 0}

        committed_count = service.commit_outbox(meta_ids)
        return {"committed": committed_count, "remaining": len(service.outbox)}

    # Mount Modular Dashboard APIRouters
    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(meta_alerts.router)
    app.include_router(raw_alerts.router)
    app.include_router(replay.router)

    # Mount Dashboard Static Files for production serving
    dashboard_dist_env = os.getenv("RBTA_DASHBOARD_DIST", "dashboard/dist")
    dashboard_dist_path = Path(dashboard_dist_env).resolve()
    if not dashboard_dist_path.exists():
        dashboard_dist_path = Path("/app/dashboard/dist").resolve()

    if dashboard_dist_path.exists() and (dashboard_dist_path / "index.html").exists():
        index_file = dashboard_dist_path / "index.html"

        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dist_path), html=True),
            name="dashboard",
        )

        @app.get("/", include_in_schema=False)
        def root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/dashboard/")

        @app.get("/dashboard/{full_path:path}", include_in_schema=False)
        def dashboard_spa_fallback(full_path: str) -> FileResponse:
            target_path = dashboard_dist_path / full_path
            if target_path.is_file():
                return FileResponse(target_path)
            return FileResponse(index_file)

    return app
