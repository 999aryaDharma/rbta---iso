from typing import Any, Dict, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.auth import get_api_key
from src.runtime.replay_controller import ReplayController

router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


class ReplayStartRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the .jsonl file in the replay data directory")
    speed_factor: Literal["1", "10", "100", "MAX"] = Field("MAX", description="Replay speed factor")


def _get_replay_controller(request: Request) -> ReplayController:
    return request.app.state.replay_controller


@router.get("/datasets")
def list_datasets(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, List[Dict[str, Any]]]:
    items = controller.list_datasets()
    return {"items": items}


@router.get("/datasets/catalog-status")
def dataset_catalog_status(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.get_catalog_status()


@router.post("/datasets/refresh")
def refresh_dataset_catalog(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    try:
        return controller.start_catalog_refresh()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status")
def replay_status(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.get_status()


@router.get("/telegram-payloads")
def list_telegram_payloads(
    limit: int = 50,
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.get_telegram_payloads(limit=min(limit, 100))


@router.post("/start")
def replay_start(
    req: ReplayStartRequest,
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    try:
        return controller.start(dataset_name=req.dataset_name, speed_factor=req.speed_factor)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pause")
def replay_pause(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.pause()


@router.post("/resume")
def replay_resume(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.resume()


@router.post("/stop")
def replay_stop(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.stop()


@router.post("/reset")
def replay_reset(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.reset()


@router.post("/evaluation/start")
def evaluation_start(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    try:
        return controller.start_evaluation(random_seed=42)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/evaluation/status")
def evaluation_status(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.get_evaluation_status()


@router.post("/evaluation/cancel")
def evaluation_cancel(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.cancel_evaluation()


@router.get("/evaluation/artifact", response_class=FileResponse)
def evaluation_artifact(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> FileResponse:
    try:
        path = controller.get_evaluation_artifact_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/json", filename=path.name)
