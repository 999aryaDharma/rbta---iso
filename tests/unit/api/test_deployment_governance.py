"""Governance and security tests for production deployment manifests, Dockerfile, and Compose."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import pytest

from src.contracts.raw_alert import CanonicalRawAlert
from src.model.registry import ModelRegistry
from src.model.scoring_pipeline import train_reference_pipeline
from src.runners.batch_runner import BatchResearchRunner

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_dockerfile_security_and_non_root():
    """Verify Dockerfile enforces non-root execution and lean image boundaries."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "USER appuser:appgroup" in content or "10001" in content
    assert "HEALTHCHECK" in content
    assert "COPY artifacts" not in content
    assert "COPY data" not in content


def test_dockerfile_image_provenance_labels():
    """Verify Dockerfile includes OCI image provenance labels and build arguments."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "ARG GIT_SHA" in content
    assert "ARG BUILD_DATE" in content
    assert "org.opencontainers.image.revision" in content
    assert "org.opencontainers.image.source" in content


def test_compose_security_baseline():
    """Verify Docker Compose deployment manifest applies least-privilege security baseline."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    assert compose_path.exists(), "deploy/asus/compose.yml must exist"
    content = compose_path.read_text(encoding="utf-8")

    assert "127.0.0.1" in content, "Host port must bind to loopback 127.0.0.1"
    assert "no-new-privileges:true" in content
    assert "cap_drop:" in content
    assert "ALL" in content
    assert ":ro" in content, "Model directory must be mounted read-only"
    assert ":rw" in content, "State directory must be mounted read-write"


def test_compose_fail_closed_on_missing_api_key_and_model_version():
    """Verify Docker Compose manifest enforces fail-closed behavior for mandatory variables."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    content = compose_path.read_text(encoding="utf-8")

    assert "${RBTA_API_KEY:?" in content, "RBTA_API_KEY must be required via fail-closed parameter expansion"
    assert "${RBTA_MODEL_VERSION:?" in content, "RBTA_MODEL_VERSION must be required via fail-closed parameter expansion"
    assert "deploy-smoke-v1" not in content, "deploy-smoke-v1 must not be an automatic fallback in Compose"


def test_dockerignore_excludes_state_and_models():
    """Verify .dockerignore excludes data, models, environment files, and git."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist"
    content = dockerignore_path.read_text(encoding="utf-8")

    assert ".env" in content
    assert ".git" in content
    assert "artifacts/" in content
    assert "data/" in content


def test_no_tracked_secrets_or_env_files():
    """Verify git tracked files do not contain .env or private credentials."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = result.stdout.splitlines()

    for path_str in tracked_files:
        assert not path_str.endswith(".env"), f"Forbidden tracked .env file: {path_str}"
        assert not path_str.endswith(".key"), f"Forbidden tracked private key: {path_str}"
        assert not path_str.endswith(".pem"), f"Forbidden tracked certificate/key: {path_str}"


def test_deploy_script_runs_preflight_first_and_gates_on_ready():
    """Verify deployment script executes preflight validation first and verifies readiness."""
    deploy_script = REPO_ROOT / "scripts" / "deploy" / "asus-deploy.sh"
    assert deploy_script.exists(), "asus-deploy.sh must exist"
    content = deploy_script.read_text(encoding="utf-8")

    assert "asus-preflight.sh" in content, "Deploy script must invoke preflight validator first"
    assert "/ready" in content, "Deploy script must explicitly gate on /ready probe"
    assert "/health" in content, "Deploy script must check /health liveness"


def test_smoke_script_requires_api_key_fail_closed():
    """Verify smoke script fails closed when RBTA_API_KEY is not provided."""
    smoke_script = REPO_ROOT / "scripts" / "deploy" / "smoke.sh"
    assert smoke_script.exists(), "smoke.sh must exist"
    content = smoke_script.read_text(encoding="utf-8")

    assert "RBTA_API_KEY" in content
    assert "Authorization: Bearer" in content


def test_validate_model_script_cli(tmp_path: Path):
    """Verify validate_model.py script correctly accepts valid bundles and rejects invalid bundles."""
    validator_script = REPO_ROOT / "scripts" / "deploy" / "validate_model.py"
    assert validator_script.exists(), "validate_model.py must exist"

    # Create a valid test bundle
    models_dir = tmp_path / "models"
    base_t = datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc)
    sample_alerts = [
        CanonicalRawAlert(
            wazuh_alert_id=f"val_alert_{i}",
            timestamp=base_t + timedelta(minutes=i * 20),
            agent_id="001",
            agent_name="soc-1",
            rule_group_primary="pam",
            rule_level=(i % 12) + 1,
            rule_id=f"550{i % 5}",
            mitre_tactics=(),
            srcip=None,
            agent_criticality=1,
        )
        for i in range(30)
    ]
    batch_res = BatchResearchRunner(base_delta_t=timedelta(minutes=15), adaptive=False).run(sample_alerts)
    bundle = train_reference_pipeline(batch_res.meta_alerts, random_state=42, model_version="val-v1")
    registry = ModelRegistry(base_dir=models_dir)
    registry.publish_bundle(bundle, model_version="val-v1")

    # 1. Valid bundle test
    res_valid = subprocess.run(
        [sys.executable, str(validator_script), "--models-dir", str(models_dir), "--version", "val-v1"],
        capture_output=True,
        text=True,
    )
    assert res_valid.returncode == 0, f"Validator should pass: {res_valid.stderr}"
    assert "PASS" in res_valid.stdout

    # 2. Nonexistent version test
    res_missing = subprocess.run(
        [sys.executable, str(validator_script), "--models-dir", str(models_dir), "--version", "nonexistent"],
        capture_output=True,
        text=True,
    )
    assert res_missing.returncode != 0
    assert "FAIL" in res_missing.stderr
