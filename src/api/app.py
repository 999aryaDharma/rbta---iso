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
    raw_evidence_store: Optional[Any] = None,
    replay_controller: Optional[Any] = None,
) -> FastAPI:
    """Factory creating configured FastAPI instance with dependency injection."""
    app = FastAPI(
        title="RBTA Security Analytics REST Service",
        version="1.0.0",
        description="Dual-mode operational runtime for Rule-Based Temporal Aggregation and Isolation Forest scoring.",
    )

    auth_key = api_key or os.getenv("RBTA_API_KEY")
    ingress_boundary = CollectorIngressBoundary(api_key=auth_key)

    app.state.raw_evidence_store = raw_evidence_store
    app.state.replay_controller = replay_controller

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

    @app.get("/api/v1/dashboard/summary", tags=["Dashboard"])
    def dashboard_summary(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            return {"ready": False}

        raw_count = len(service.engine._seen_alert_ids)
        meta_count = len(service.finalized_history)

        return {
            "raw_alert_count": raw_count,
            "meta_alert_count": meta_count,
            "alert_reduction_rate": 1 - (meta_count / raw_count) if raw_count > 0 else 0.0,
            "escalate_count": sum(1 for m in service.finalized_history if m.escalate),
            "suppress_count": sum(1 for m in service.finalized_history if m.action == 'SUPPRESS'),
            "active_agents_count": len(service.engine._temporal_states),
            "active_buckets_count": len(service.engine._active_buckets),
            "outbox_depth": len(service.get_outbox()),
            "source_mode": getattr(service, 'source_mode', 'LIVE'),
            "model_version": service.scoring_pipeline.metadata.get("model_version", "unknown") if service.scoring_pipeline else "unknown",
            "ready": True,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/api/v1/dashboard/agents", tags=["Dashboard"])
    def dashboard_agents(authorization: Optional[str] = Header(None)) -> List[Dict[str, Any]]:
        verify_auth(authorization)
        if service is None:
            return []
        return service.engine.snapshot_agents()

    @app.get("/api/v1/dashboard/buckets", tags=["Dashboard"])
    def dashboard_buckets(authorization: Optional[str] = Header(None)) -> List[Dict[str, Any]]:
        verify_auth(authorization)
        if service is None:
            return []
        return service.engine.snapshot_buckets()

    @app.get("/api/v1/dashboard/timeseries", tags=["Dashboard"])
    def dashboard_timeseries(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            return {"series": []}

        from collections import defaultdict

        buckets = defaultdict(lambda: {"raw_alerts": 0, "meta_alerts": 0})
        for meta in service.finalized_history:
            if not meta.start_time:
                continue
            hour_str = meta.start_time.replace(minute=0, second=0, microsecond=0).isoformat()
            buckets[hour_str]["meta_alerts"] += 1
            buckets[hour_str]["raw_alerts"] += meta.alert_count

        series = [{"time": k, "raw_alerts": v["raw_alerts"], "meta_alerts": v["meta_alerts"]}
                  for k, v in sorted(buckets.items())]
        return {"series": series}

    @app.get("/api/v1/dashboard/system", tags=["Dashboard"])
    def dashboard_system(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            return {"api_status": "ok", "runtime_ready": False}

        pipeline = service.scoring_pipeline
        return {
            "api_status": "ok",
            "runtime_ready": True,
            "source_mode": getattr(service, 'source_mode', 'LIVE'),
            "model_version": pipeline.metadata.get("model_version", "unknown") if pipeline else "unknown",
            "feature_schema_version": "1.0",
            "score_calibration_version": pipeline.metadata.get("score_calibration_version", "minmax-v1") if pipeline else "unknown",
            "threshold": pipeline.threshold.threshold if pipeline and pipeline.threshold else 0.0,
            "seen_alerts": len(service.engine._seen_alert_ids),
            "active_buckets": len(service.engine._active_buckets),
            "outbox_depth": len(service.get_outbox()),
            "current_run_id": app.state.replay_controller.run_id if app.state.replay_controller else None
        }

    @app.get("/api/v1/meta-alerts", tags=["Dashboard"])
    def meta_alerts_list(
        page: int = 1,
        page_size: int = 20,
        decision: Optional[str] = None,
        agent_id: Optional[str] = None,
        rule_group: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "desc",
        authorization: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        filtered = service.finalized_history
        if decision:
            filtered = [m for m in filtered if m.decision == decision]
        if agent_id:
            filtered = [m for m in filtered if m.agent_id == agent_id]
        if rule_group:
            filtered = [m for m in filtered if m.rule_group_primary == rule_group]
        if search:
            search = search.lower()
            filtered = [m for m in filtered if search in m.agent_name.lower() or search in m.rule_group_primary.lower()]

        if sort_by:
            rev = (sort_order == "desc")
            try:
                filtered = sorted(filtered, key=lambda m: getattr(m, sort_by), reverse=rev)
            except AttributeError:
                pass
        else:
            filtered = sorted(filtered, key=lambda m: m.meta_id, reverse=True)

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size

        items = [_serialize_scored_alert(m) for m in filtered[start:end]]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size
        }

    @app.get("/api/v1/meta-alerts/{meta_id}/raw-alerts", tags=["Dashboard"])
    def meta_alert_raw_alerts(
        meta_id: int,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        rule_id: Optional[str] = None,
        level_min: Optional[int] = None,
        level_max: Optional[int] = None,
        srcip: Optional[str] = None,
        mitre_tactic: Optional[str] = None,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
        authorization: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        verify_auth(authorization)
        if service is None:
            raise HTTPException(404)

        detail = service.get_meta_detail(meta_id)
        if not detail:
            for item in service.get_outbox():
                if item.meta_id == meta_id:
                    detail = item
                    break
        if not detail:
            raise HTTPException(404, "Meta alert not found")

        store = app.state.raw_evidence_store
        if not store:
            return {
                "meta_id": meta_id,
                "total": 0,
                "resolved_count": 0,
                "unresolved_alert_ids": list(detail.source_alert_ids),
                "items": [],
                "page": page,
                "page_size": page_size
            }

        items, total = store.search(
            meta_id_alert_ids=list(detail.source_alert_ids),
            page=page,
            page_size=page_size,
            search=search,
            rule_id=rule_id,
            level_min=level_min,
            level_max=level_max,
            srcip=srcip,
            mitre_tactic=mitre_tactic,
            from_ts=from_ts,
            to_ts=to_ts
        )

        return {
            "meta_id": meta_id,
            "total": total,
            "resolved_count": len(items) if not search and not rule_id and level_min is None else -1, # partial indicator
            "unresolved_alert_ids": [],
            "items": items,
            "page": page,
            "page_size": page_size
        }

    @app.get("/api/v1/raw-alerts/{wazuh_alert_id}", tags=["Dashboard"])
    def raw_alert_detail(
        wazuh_alert_id: str,
        authorization: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        verify_auth(authorization)
        store = app.state.raw_evidence_store
        if not store:
            raise HTTPException(503, "Store not initialized")

        alert = store.get(wazuh_alert_id)
        if not alert:
            raise HTTPException(404, "Raw alert not found")
        return alert

    @app.get("/api/v1/replay/status", tags=["Dashboard"])
    def replay_status(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            return {"status": "UNAVAILABLE"}
        return ctrl.get_status()

    @app.post("/api/v1/replay/start", tags=["Dashboard"])
    def replay_start(speed: float = 1.0, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            raise HTTPException(503)
        ctrl.start(speed)
        return ctrl.get_status()

    @app.post("/api/v1/replay/pause", tags=["Dashboard"])
    def replay_pause(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            raise HTTPException(503)
        ctrl.pause()
        return ctrl.get_status()

    @app.post("/api/v1/replay/resume", tags=["Dashboard"])
    def replay_resume(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            raise HTTPException(503)
        ctrl.resume()
        return ctrl.get_status()

    @app.post("/api/v1/replay/stop", tags=["Dashboard"])
    def replay_stop(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            raise HTTPException(503)
        ctrl.stop()
        return ctrl.get_status()

    @app.post("/api/v1/replay/reset", tags=["Dashboard"])
    def replay_reset(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
        verify_auth(authorization)
        ctrl = app.state.replay_controller
        if not ctrl:
            raise HTTPException(503)
        ctrl.reset()
        return ctrl.get_status()

    # Mount Dashboard Static Files for production serving
    from pathlib import Path
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

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
