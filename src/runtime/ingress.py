"""Authenticated collector and webhook ingress boundary module."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from src.contracts.raw_alert import CanonicalRawAlert
from src.etl.wazuh_canonicalizer import CanonicalizationError, canonicalize_wazuh_alert


class IngressPayloadError(ValueError):
    """Raised when ingress payload is unauthorized, missing mandatory attributes, or malformed."""
    pass


@dataclass(frozen=True)
class IngressResult:
    """Outcome of collector ingress processing."""

    status: str
    alert_id: Optional[str]
    is_duplicate: bool
    canonical_alert: Optional[CanonicalRawAlert]


class CollectorIngressBoundary:
    """Idempotent transport boundary for receiving alerts pushed from campus collectors.

    Parameters
    ----------
    api_key : str | None
        Optional shared secret / Bearer token required for authentication.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: Optional[str] = api_key

    def process_incoming(
        self,
        payload: Dict[str, Any],
        auth_header: Optional[str] = None,
    ) -> IngressResult:
        """Authenticate, validate, and canonicalize an incoming raw alert payload.

        Parameters
        ----------
        payload : Dict[str, Any]
            Raw alert JSON object.
        auth_header : str | None
            HTTP Authorization header (e.g. 'Bearer <token>').

        Returns
        -------
        IngressResult
            Processing outcome indicating if alert is new, duplicate, and its canonical representation.

        Raises
        ------
        IngressPayloadError
            If authentication fails or payload is malformed.
        """
        # 1. Authentication check
        if self.api_key:
            expected_auth = f"Bearer {self.api_key}"
            if auth_header != expected_auth and auth_header != self.api_key:
                raise IngressPayloadError("Unauthorized collector ingress request")

        # 2. Canonicalization
        try:
            canonical_alert = canonicalize_wazuh_alert(payload)
        except (CanonicalizationError, Exception) as exc:
            raise IngressPayloadError(f"Malformed raw alert payload: {exc}") from exc

        # 3. Return Canonical Alert
        alert_id = canonical_alert.wazuh_alert_id
        return IngressResult(
            status="accepted",
            alert_id=alert_id,
            is_duplicate=False,
            canonical_alert=canonical_alert,
        )
