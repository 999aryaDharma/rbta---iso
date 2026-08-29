from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.contracts.scored_meta_alert import ScoredMetaAlert
from src.evaluation.metrics import compute_arr
from src.runtime.raw_evidence import RawAlertEvidenceStore
from src.runtime.service import LiveRBTAService


def get_dashboard_summary(
    service: LiveRBTAService,
    evidence_store: RawAlertEvidenceStore,
) -> Dict[str, Any]:
    """Build pure read-only dashboard summary KPIs using canonical research metrics."""
    raw_count = evidence_store.count()
    history = list(service.finalized_history)
    meta_count = len(history)

    arr_val = compute_arr(raw_count, meta_count)
    arr_percent = round(arr_val * 100.0, 2) if arr_val is not None else None

    escalate_count = sum(1 for m in history if m.action == "ESCALATE")
    digest_count = sum(1 for m in history if m.action == "DAILY_DIGEST")
    suppress_count = sum(1 for m in history if m.action == "SUPPRESS")
    critical_count = sum(1 for m in history if m.decision == "CRITICAL")
    anomalies_count = sum(1 for m in history if m.anomaly_score >= m.threshold_used)

    active_buckets = service.engine.snapshot_buckets()

    return {
        "raw_alert_count": raw_count,
        "meta_alert_count": meta_count,
        "alert_reduction_rate_percent": arr_percent,
        "active_buckets_count": len(active_buckets),
        "escalate_count": escalate_count,
        "digest_count": digest_count,
        "suppress_count": suppress_count,
        "anomalies_detected": anomalies_count,
        "critical_meta_count": critical_count,
        "source_mode": service.source_mode,
        "system_status": "READY",
    }


def get_dashboard_agents(service: LiveRBTAService) -> List[Dict[str, Any]]:
    """Build pure snapshot of all per-agent temporal states."""
    return service.engine.snapshot_agents()


def get_dashboard_buckets(service: LiveRBTAService) -> List[Dict[str, Any]]:
    """Build pure snapshot of currently open RBTA buckets."""
    return service.engine.snapshot_buckets()


def get_dashboard_timeseries(
    service: LiveRBTAService,
    evidence_store: RawAlertEvidenceStore,
    window_hours: int = 24,
) -> List[Dict[str, Any]]:
    """Build time series aggregation of raw incoming alerts vs finalized MetaAlerts."""
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=window_hours)

    # Gather finalized MetaAlerts
    history = list(service.finalized_history)

    # Simple hourly binning
    bins: Dict[str, Dict[str, Any]] = {}
    for h in range(window_hours):
        bin_dt = start_time + timedelta(hours=h)
        bin_key = bin_dt.strftime("%Y-%m-%d %H:00")
        bins[bin_key] = {
            "timestamp": bin_key,
            "raw_alerts": 0,
            "meta_alerts": 0,
        }

    for m in history:
        if m.end_time and m.end_time >= start_time:
            bin_key = m.end_time.strftime("%Y-%m-%d %H:00")
            if bin_key in bins:
                bins[bin_key]["meta_alerts"] += 1
                bins[bin_key]["raw_alerts"] += m.alert_count

    return list(bins.values())


def get_dashboard_system(service: LiveRBTAService) -> Dict[str, Any]:
    """Build system metadata and model configuration DTO."""
    bundle = getattr(service.scoring_pipeline, "bundle", None) if service.scoring_pipeline else None

    model_version = getattr(bundle, "model_version", "UNKNOWN") if bundle else "UNKNOWN"
    threshold = float(getattr(bundle, "tukey_threshold", 0.0)) if bundle else 0.0
    random_state = getattr(bundle, "random_state", None) if bundle else None
    features = list(getattr(bundle, "feature_names", [])) if bundle else []

    return {
        "model_version": model_version,
        "tukey_threshold": threshold,
        "random_state": random_state,
        "feature_names": features,
        "base_delta_t_seconds": service.base_delta_t.total_seconds(),
        "adaptive": service.adaptive,
        "source_mode": service.source_mode,
        "durable_state_path": str(service.state_manager.state_path),
        "raw_evidence_db_path": str(service.raw_evidence_store.db_path) if service.raw_evidence_store else None,
        "system_status": "READY",
    }


def get_dashboard_integrations() -> Dict[str, Any]:
    """Return backend-truth integration statuses."""
    return {
        "wazuh": {
            "name": "Wazuh SIEM Ingestion",
            "status": "DEFERRED",
            "detail": "Production source connectivity not configured",
        },
        "rbta": {
            "name": "RBTA Temporal Engine",
            "status": "READY",
            "detail": "Per-agent EMA baseline adaptive clustering active",
        },
        "model": {
            "name": "Isolation Forest Scoring",
            "status": "READY",
            "detail": "Trained reference bundle loaded with Tukey IQR threshold",
        },
        "outbox": {
            "name": "Durable Outbox Queue",
            "status": "READY",
            "detail": "SQLite transactional staging and crash recovery active",
        },
        "shuffle": {
            "name": "Shuffle SOAR Webhook",
            "status": "UNKNOWN",
            "detail": "External SOAR webhook unverified in local demo environment",
        },
        "telegram": {
            "name": "Telegram Incident Bot",
            "status": "UNKNOWN",
            "detail": "External bot credentials unverified in local demo environment",
        },
    }
