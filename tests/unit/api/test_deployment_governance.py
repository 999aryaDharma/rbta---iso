"""Governance and static invariant tests for ASUS deployment configuration and scripts."""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy" / "asus"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "deploy"


def test_compose_manifest_invariants():
    """Verify compose.yml satisfies all ASUS deployment contract invariants."""
    compose_path = DEPLOY_DIR / "compose.yml"
    assert compose_path.exists(), "deploy/asus/compose.yml must exist"

    content = compose_path.read_text(encoding="utf-8")

    # B7: No default port 8000 fallback
    assert "${RBTA_HOST_PORT:-8000}" not in content, "compose.yml must not allow fallback to port 8000"
    assert "RBTA_HOST_PORT:?" in content, "compose.yml must fail closed on missing RBTA_HOST_PORT"

    # B8: Image tag derived from required variable
    assert "rbta-service:s10" not in content, "Stale s10 image tag must not be present"
    assert "RBTA_IMAGE_TAG:?" in content, "compose.yml must require RBTA_IMAGE_TAG"

    # B9: Replay archive mounted read-only
    assert "RBTA_REPLAY_HOST_DIR:?" in content, "compose.yml must require RBTA_REPLAY_HOST_DIR"
    assert "/app/data/replay:ro" in content, "Replay archive must be mounted :ro"

    # Strict volume mounts
    assert "RBTA_STATE_HOST_DIR:?" in content
    assert "RBTA_MODEL_HOST_DIR:?" in content


def test_env_example_template_completeness():
    """Verify deploy/asus/.env.example contains all mandatory configuration keys."""
    env_ex_path = DEPLOY_DIR / ".env.example"
    assert env_ex_path.exists(), "deploy/asus/.env.example must exist"

    content = env_ex_path.read_text(encoding="utf-8")

    required_keys = [
        "RBTA_API_KEY",
        "RBTA_MODEL_VERSION",
        "RBTA_SOURCE_MODE=DEFERRED",
        "RBTA_HOST_PORT",
        "RBTA_STATE_HOST_DIR",
        "RBTA_MODEL_HOST_DIR",
        "RBTA_REPLAY_HOST_DIR",
        "RBTA_CODE_SHA",
        "RBTA_IMAGE_TAG",
    ]
    for key in required_keys:
        assert key in content, f"Expected '{key}' in .env.example"


def test_production_smoke_is_strictly_read_only():
    """Verify scripts/deploy/smoke.sh performs zero state-mutating requests."""
    smoke_path = SCRIPTS_DIR / "smoke.sh"
    assert smoke_path.exists(), "scripts/deploy/smoke.sh must exist"

    content = smoke_path.read_text(encoding="utf-8")

    # Must NOT contain alert ingestion or outbox commits
    assert "/api/v1/alerts/ingest" not in content, "Production smoke must not ingest alerts"
    assert "POST" not in content, "Production smoke must not issue any POST requests"
    assert "s10_smoke_test_alert_001" not in content, "Stale mutating fixture must be removed"

    # Must verify key observability and static routes
    assert "/health" in content
    assert "/ready" in content
    assert "/api/v1/auth/check" in content
    assert "/runtime/stats" in content
    assert "/dashboard/" in content
    assert "/api/v1/replay/datasets" in content
    assert "/api/v1/dashboard/system" in content
    assert "/api/v1/dashboard/integrations" in content


def test_isolated_smoke_guarantees_cleanup_and_non_production_id():
    """Verify scripts/deploy/smoke-isolated.sh runs in isolation and traps cleanup."""
    isolated_path = SCRIPTS_DIR / "smoke-isolated.sh"
    assert isolated_path.exists(), "scripts/deploy/smoke-isolated.sh must exist"

    content = isolated_path.read_text(encoding="utf-8")

    # Must use non-production engineering sentinel IDs (never '001')
    assert "__engineering_smoke_alert_001__" in content
    assert "__engineering_smoke_agent__" in content
    assert 'agent": {"id": "001"' not in content

    # Must trap cleanup
    assert "trap cleanup EXIT" in content
    assert "docker stop" in content
    assert "docker rm" in content


def test_deploy_scripts_ancestry_and_port_validation():
    """Verify deployment scripts enforce port range and code sha ancestry."""
    preflight_path = SCRIPTS_DIR / "asus-preflight.sh"
    deploy_path = SCRIPTS_DIR / "asus-deploy.sh"

    preflight_content = preflight_path.read_text(encoding="utf-8")
    deploy_content = deploy_path.read_text(encoding="utf-8")

    # Preflight validates port range 1024..65535
    assert "1024" in preflight_content
    assert "65535" in preflight_content

    # Deploy script checks git ancestry
    assert "merge-base --is-ancestor" in deploy_content
    assert "scripts/deploy/validate_model.py" in deploy_content
