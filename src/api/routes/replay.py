from typing import Any, Dict, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Request
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


@router.get("/status")
def replay_status(
    controller: ReplayController = Depends(_get_replay_controller),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    return controller.get_status()


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
