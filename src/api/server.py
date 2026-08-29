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
) -> FastAPI:
    """Create and configure the production FastAPI application with full dependency injection.

    Parameters
    ----------
    env : Dict[str, str] | None
        Optional environment override dictionary for testing or programmatic bootstrap.

    Returns
    -------
    FastAPI
        Configured production application instance.

    Raises
    ------
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

    # 4. Construct live stateful service
    service: Optional[LiveRBTAService] = None
    if scoring_pipe is not None:
        service = LiveRBTAService(
            scoring_pipeline=scoring_pipe,
            state_manager=state_mgr,
            adaptive=True,
        )

    # 5. Lifespan for graceful shutdown
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting RBTA production service...")
        yield
        logger.info("Shutting down RBTA production service (preserving active buckets)...")
        if service is not None:
            service.shutdown(drain=False)

    app = create_app(
        service=service,
        model_registry=registry,
        api_key=api_key,
    )
    app.router.lifespan_context = lifespan

    return app


def run() -> None:
    """Production server entrypoint."""
    log_level = os.getenv("RBTA_LOG_LEVEL", "info").lower()
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    host = os.getenv("RBTA_HOST", "127.0.0.1")
    port = int(os.getenv("RBTA_PORT", "8000"))

    app = create_production_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    run()
