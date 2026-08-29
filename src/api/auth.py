from typing import Optional
from fastapi import Header, HTTPException, Request, status
import os


def get_api_key(request: Request, authorization: Optional[str] = Header(None)) -> str:
    """Dependency validating HTTP Authorization header against configured API key."""
    expected_key = getattr(request.app.state, "auth_key", None)
    if expected_key is None:
        expected_key = os.getenv("RBTA_API_KEY", "")

    if not expected_key:
        return ""

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    expected_bearer = f"Bearer {expected_key}"
    if authorization != expected_bearer and authorization != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )

    return expected_key
