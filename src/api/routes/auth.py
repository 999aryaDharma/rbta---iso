from fastapi import APIRouter, Depends
from src.api.auth import get_api_key

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.get("/check")
def check_auth(api_key: str = Depends(get_api_key)):
    """Lightweight endpoint for frontend to validate an API key before storing in sessionStorage."""
    return {"status": "authenticated", "authenticated": True}
