"""Production FastAPI server bootstrap with validated configuration and lifecycle management."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI
import uvicorn

from src.api.app import create_app
from src.model.registry import ModelRegistry, ModelRegistryError
from src.model.scoring_pipeline import ScoringPipeline
from src.runtime.durable_state import DurableStateManager
from src.runtime.service import LiveRBTAService

logger = logging.getLogger("rbta.server")


def create_production_app(
    env: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> FastAPI:
    """Create and configure the production FastAPI application with full dependency injection.

    Parameters
    ----------
    env : Dict[str, str] | None
        Optional environment override dictionary for testing or programmatic bootstrap.
    strict : bool
        When True, strictly requires non-empty RBTA_API_KEY and RBTA_MODEL_VERSION.

    Returns
    -------
    FastAPI
        Configured production application instance.

    Raises
    ------
    RuntimeError
        If strict=True and mandatory security/model environment variables are missing.
    ValueError
        If mandatory paths cannot be validated or written.
    ModelRegistryError
        If model artifacts are corrupted or invalid.
    """
    env_map = os.environ if env is None else env

    api_key = env_map.get("RBTA_API_KEY")
    registry_dir = Path(env_map.get("RBTA_MODEL_REGISTRY_DIR", "artifacts/models")).resolve()
    model_version = env_map.get("RBTA_MODEL_VERSION")
    state_file_path = Path(env_map.get("RBTA_STATE_FILE", "data/runtime/state.json")).resolve()

    if strict:
        if not api_key or not api_key.strip():
            raise RuntimeError("RBTA_API_KEY environment variable is mandatory and must not be empty in production.")
        if not model_version or not model_version.strip():
            raise RuntimeError("RBTA_MODEL_VERSION environment variable is mandatory and must not be empty in production.")

    # N4: single live writer per state file for EVERY bootstrap (not only
    # worker-enabled): take the OS-level lock before any DurableStateManager
    # or service can write, so a second process fails fast here.
    from src.runtime.live_worker import acquire_state_lock as _acquire_state_lock

    _acquire_state_lock(state_file_path)

    # 1. State directory accessibility check
    state_dir = state_file_path.parent
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ValueError(f"Cannot create or access state directory '{state_dir}': {exc}") from exc

    state_mgr = DurableStateManager(state_file_path)

    # 2. Model registry initialization
    registry = ModelRegistry(base_dir=registry_dir, explicit_version=model_version)

    # 3. Model bundle resolution
    active_version = registry.get_active_version()
    scoring_pipe: Optional[ScoringPipeline] = None

    if active_version:
        logger.info("Loading active model artifact version: '%s'", active_version)
        bundle = registry.load_bundle(active_version)
        scoring_pipe = ScoringPipeline(bundle)
    else:
        logger.warning(
            "No active model bundle found in '%s' for version '%s'. /ready will return 503.",
            registry_dir,
            model_version,
        )

    from src.runtime.raw_evidence import RawAlertEvidenceStore
    from src.runtime.replay_controller import ReplayController

    raw_evidence_db = env_map.get("RBTA_RAW_EVIDENCE_DB", "data/runtime/raw_alert_evidence.sqlite3")
    raw_evidence_store = RawAlertEvidenceStore(raw_evidence_db)

    # 4. Construct live stateful service
    # When the live worker is opted in and RBTA_SOURCE_MODE is not set
    # explicitly, the live service defaults to LIVE. The demonstration
    # replay controller always builds its own REPLAY-scoped service, so
    # replay behavior is untouched by this default.
    from src.runtime.live_worker import worker_enabled_from_env as _worker_on

    _worker_enabled = _worker_on(env_map)
    raw_source_mode = env_map.get("RBTA_SOURCE_MODE")
    if raw_source_mode is None or not str(raw_source_mode).strip():
        source_mode = "LIVE" if _worker_enabled else "DEFERRED"
    else:
        source_mode = str(raw_source_mode).strip().upper()
    if source_mode not in ("DEFERRED", "LIVE"):
        raise ValueError(f"RBTA_SOURCE_MODE must be either 'DEFERRED' or 'LIVE', got '{source_mode}'")

    service: Optional[LiveRBTAService] = None
    if scoring_pipe is not None:
        service = LiveRBTAService(
            scoring_pipeline=scoring_pipe,
            state_manager=state_mgr,
            adaptive=True,
            raw_evidence_store=raw_evidence_store,
            source_mode=source_mode,
        )

    # Insecure-TLS fail-visible: warn at startup and always record the
    # operator choice in the live source state for thesis evidence lineage.
    tls_verify_raw = str(env_map.get("WAZUH_INDEXER_VERIFY_TLS", "true")).strip().lower()
    tls_verify = tls_verify_raw != "false"
    if not tls_verify:
        logger.warning(
            "WAZUH_INDEXER_VERIFY_TLS=false: Wazuh indexer TLS certificate "
            "verification is DISABLED. Transport is vulnerable to MITM; "
            "enable verification for production use."
        )
    if service is not None:
        service.update_live_source_state({"tls_verify": tls_verify})

    # 5. Construct demonstration replay controller
    replay_controller: Optional[ReplayController] = None
    if scoring_pipe is not None:
        default_replay_dir = "/app/data/replay" if Path("/app/data/replay").exists() else "data/test_datasets"
        replay_data_dir = Path(env_map.get("RBTA_REPLAY_DATA_DIR", default_replay_dir)).resolve()
        replay_controller = ReplayController(
            scoring_pipeline=scoring_pipe,
            replay_data_dir=replay_data_dir,
        )

    # 6. Lifespan for graceful shutdown (+ optional live worker thread, L2)
    from src.runtime.live_worker import (
        LiveWorker,
        drain_on_stop_from_env,
        poll_interval_from_env,
        worker_enabled_from_env,
    )

    live_worker: Optional[LiveWorker] = None
    if service is not None and worker_enabled_from_env(env_map):
        # The state lock was already taken for every bootstrap above; no
        # second acquire here.
        from src.runtime.live_coordinator import LiveIngestionCoordinator

        buffer_enabled = str(env_map.get("RBTA_ORDER_BUFFER_ENABLED", "false")).strip().lower() == "true"
        if buffer_enabled:
            logger.info("Order buffer enabled for live ingestion (waiting-room sorter).")
        live_source = str(env_map.get("RBTA_LIVE_SOURCE", "indexer")).strip().lower()
        poller = None
        if live_source == "api":
            from src.ingestion.wazuh_api_client import WazuhAPIClient
            from src.runtime.api_live_source import WazuhAPILivePoller

            poller = WazuhAPILivePoller(client=WazuhAPIClient())
            logger.info("Live ingestion source: custom flat API (FLAT-API-FREEZE).")
        elif live_source != "indexer":
            raise ValueError(f"RBTA_LIVE_SOURCE must be either 'indexer' or 'api', got '{live_source}'")
        live_worker = LiveWorker(
            service,
            LiveIngestionCoordinator(
                service=service,
                poller=poller,
                order_buffer_enabled=buffer_enabled,
                breaker_ack=(env_map.get("RBTA_QUARANTINE_BREAKER_ACK") or None),
            ),
            poll_interval=poll_interval_from_env(env_map),
            drain_on_stop=drain_on_stop_from_env(env_map),
            model_override=env_map.get("RBTA_LIVE_MODEL_OVERRIDE"),
            derivation_override=env_map.get("RBTA_LIVE_DERIVATION_OVERRIDE"),
        )

    # Telegram dispatcher (L5): attached beside the worker, never touching it.
    # Without bot credentials it stays disabled (dry-run only: the thread
    # still starts so the outbox is drained and each pass is logged, but
    # nothing is sent). The send path runs only when credentials exist via
    # make_telegram_sender.
    # RBTA_TELEGRAM_DRY_RUN=true forces dry-run mode: credentials are NOT
    # passed to the constructor, so the dispatcher stays disabled by design.
    telegram_dispatcher: Optional[Any] = None
    if live_worker is not None:
        from src.runtime.telegram_dispatcher import TelegramDispatcher, make_telegram_sender

        telegram_dry_run = str(env_map.get("RBTA_TELEGRAM_DRY_RUN", "false")).strip().lower() == "true"
        if telegram_dry_run:
            bot_token = ""
            chat_id = ""
            sender = None
        else:
            bot_token = (env_map.get("RBTA_TELEGRAM_BOT_TOKEN") or "").strip()
            chat_id = (env_map.get("RBTA_TELEGRAM_CHAT_ID") or "").strip()
            sender = make_telegram_sender(bot_token) if bot_token and chat_id else None
        telegram_dispatcher = TelegramDispatcher(
            service, sender=sender, bot_token=bot_token or None, chat_id=chat_id or None
        )
        # P2: fail-visible startup hint — worker exists but the dispatcher is
        # HOLDING the outbox for lack of credentials, and the operator did not
        # explicitly opt into dry-run. Warn exactly once here (not in the
        # lifespan loop) so a silent no-notify deployment is obvious.
        if not telegram_dry_run and not telegram_dispatcher.enabled:
            logger.warning(
                "Telegram credentials missing; dispatcher HOLDING outbox — "
                "set creds or RBTA_TELEGRAM_DRY_RUN=true"
            )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting RBTA production service...")
        if live_worker is not None:
            if live_worker.start():
                logger.info("Live worker thread started.")
            else:
                logger.warning("Live worker already running; keeping single thread.")
        if telegram_dispatcher is not None:
            telegram_dispatcher.start()
            logger.info(
                "Telegram dispatcher thread started (enabled=%s).",
                telegram_dispatcher.enabled,
            )
        yield
        if live_worker is not None and live_worker.drain_on_stop:
            logger.info(
                "Shutting down RBTA production service "
                "(draining active buckets — explicit decommission)..."
            )
        else:
            logger.info("Shutting down RBTA production service (preserving active buckets)...")
        if telegram_dispatcher is not None:
            telegram_dispatcher.stop()
        if live_worker is not None:
            live_worker.stop()
        if replay_controller is not None:
            replay_controller.stop()
        if service is not None:
            service.shutdown(drain=False)

    app = create_app(
        service=service,
        model_registry=registry,
        api_key=api_key,
        raw_evidence_store=raw_evidence_store,
        replay_controller=replay_controller,
    )
    app.router.lifespan_context = lifespan
    app.state.live_worker = live_worker
    app.state.telegram_dispatcher = telegram_dispatcher

    return app


def run() -> None:
    """Production server entrypoint."""
    log_level = os.getenv("RBTA_LOG_LEVEL", "info").lower()
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    host = os.getenv("RBTA_HOST", "0.0.0.0")
    port = int(os.getenv("RBTA_PORT", "8000"))

    app = create_production_app(strict=True)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    run()
