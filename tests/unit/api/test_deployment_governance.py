"""Governance and behavioral security tests for production deployment manifests, scripts, and Compose."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import stat
import subprocess
import sys
import pytest

from scripts.deploy.validate_state_dir import check_state_dir_permissions
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
    assert "../../models:/app/artifacts/models:ro" in content, "Model directory must be mounted root-relative read-only"
    assert "../../state:/app/data/runtime:rw" in content, "State directory must be mounted root-relative read-write"


def test_compose_fail_closed_on_missing_api_key_and_model_version():
    """Verify Docker Compose manifest syntax enforces fail-closed parameter expansion."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    content = compose_path.read_text(encoding="utf-8")

    assert "${RBTA_API_KEY:?" in content, "RBTA_API_KEY must be required via fail-closed parameter expansion"
    assert "${RBTA_MODEL_VERSION:?" in content, "RBTA_MODEL_VERSION must be required via fail-closed parameter expansion"
    assert "deploy-smoke-v1" not in content, "deploy-smoke-v1 must not be an automatic fallback in Compose"


def test_compose_behavioral_fail_closed_missing_api_key():
    """Behavioral execution test: Compose fails when RBTA_API_KEY is missing."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    env = {k: v for k, v in os.environ.items() if k not in ("RBTA_API_KEY", "RBTA_MODEL_VERSION")}
    env["RBTA_MODEL_VERSION"] = "ci-test-v1"

    res = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "config"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "Compose config must fail when RBTA_API_KEY is missing"
    assert "RBTA_API_KEY" in res.stderr, f"Error output should mention missing RBTA_API_KEY: {res.stderr}"


def test_compose_behavioral_fail_closed_missing_model_version():
    """Behavioral execution test: Compose fails when RBTA_MODEL_VERSION is missing."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    env = {k: v for k, v in os.environ.items() if k not in ("RBTA_API_KEY", "RBTA_MODEL_VERSION")}
    env["RBTA_API_KEY"] = "ci-test-key"

    res = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "config"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "Compose config must fail when RBTA_MODEL_VERSION is missing"
    assert "RBTA_MODEL_VERSION" in res.stderr, f"Error output should mention missing RBTA_MODEL_VERSION: {res.stderr}"


def test_compose_behavioral_success_with_required_vars():
    """Behavioral execution test: Compose succeeds when both mandatory variables are supplied."""
    compose_path = REPO_ROOT / "deploy" / "asus" / "compose.yml"
    env = {k: v for k, v in os.environ.items()}
    env["RBTA_API_KEY"] = "ci-test-key"
    env["RBTA_MODEL_VERSION"] = "ci-test-v1"

    res = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "config", "--quiet"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Compose config must succeed when mandatory variables are provided: {res.stderr}"


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


def test_smoke_script_behavioral_auth_fail_closed():
    """Behavioral execution test: smoke.sh exits non-zero before network call if RBTA_API_KEY is unset."""
    if os.name != "posix":
        pytest.skip("Bash script behavioral execution requires POSIX environment (Linux CI)")

    smoke_script = REPO_ROOT / "scripts" / "deploy" / "smoke.sh"
    env = {k: v for k, v in os.environ.items() if k != "RBTA_API_KEY"}
    res = subprocess.run(
        ["bash", str(smoke_script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "smoke.sh must exit non-zero when unauthenticated"
    assert "RBTA_API_KEY" in res.stderr, f"Stderr should explain missing API key: {res.stderr}"


def test_preflight_behavioral_missing_env():
    """Behavioral execution test: asus-preflight.sh fails when RBTA_ENV_FILE is invalid."""
    if os.name != "posix":
        pytest.skip("Bash script behavioral execution requires POSIX environment (Linux CI)")

    preflight_script = REPO_ROOT / "scripts" / "deploy" / "asus-preflight.sh"
    env = {**os.environ, "RBTA_ENV_FILE": "nonexistent_env_file.env"}
    res = subprocess.run(
        ["bash", str(preflight_script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "preflight must exit non-zero when env file does not exist"
    assert "not found" in res.stderr, f"Stderr should indicate file not found: {res.stderr}"


def test_state_dir_validation_behavioral_wrong_uid_gid(tmp_path: Path):
    """Behavioral test: state validator rejects directories not owned by UID/GID 10001 on POSIX."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    if os.name == "posix":
        # Test with an impossible UID (e.g. 99999) to verify rejection logic
        is_valid, msg = check_state_dir_permissions(state_dir, target_uid=99999, target_gid=99999)
        assert is_valid is False
        assert "99999:99999" in msg
        assert "sudo chown -R 99999:99999" in msg
        assert "sudo chmod 0750" in msg


def test_state_dir_validation_behavioral_mock_uid_gid(tmp_path: Path, monkeypatch):
    """Behavioral test: state validator accepts matching UID/GID 10001 with 0750 mode."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    class MockStat:
        st_uid = 10001
        st_gid = 10001
        st_mode = stat.S_IFDIR | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP  # 0750

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: MockStat())

    is_valid, msg = check_state_dir_permissions(state_dir, target_uid=10001, target_gid=10001)
    assert is_valid is True
    assert "10001:10001" in msg
    assert "PASS" in msg


def test_state_dir_validation_behavioral_rejects_world_writable(tmp_path: Path, monkeypatch):
    """Behavioral test: state validator rejects insecure world-writable (0777) state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    class MockStat777:
        st_uid = 10001
        st_gid = 10001
        st_mode = stat.S_IFDIR | 0o777

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: MockStat777())

    is_valid, msg = check_state_dir_permissions(state_dir, target_uid=10001, target_gid=10001)
    assert is_valid is False
    assert "world-writable" in msg.lower()
    assert "forbidden" in msg.lower()


def test_state_dir_validation_behavioral_rejects_missing_execute_0640(tmp_path: Path, monkeypatch):
    """Behavioral test: state validator rejects directories without owner execute/traverse permission (e.g. 0640)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    class MockStat0640:
        st_uid = 10001
        st_gid = 10001
        st_mode = stat.S_IFDIR | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP  # 0640 (rw-r-----)

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "stat", lambda *args, **kwargs: MockStat0640())

    is_valid, msg = check_state_dir_permissions(state_dir, target_uid=10001, target_gid=10001)
    assert is_valid is False
    assert "executable" in msg.lower() or "traversable" in msg.lower()
    assert "sudo chmod 0750" in msg


def test_preflight_behavioral_empty_api_key(tmp_path: Path):
    """Behavioral test: asus-preflight.sh fails when RBTA_API_KEY is empty in .env."""
    if os.name != "posix":
        pytest.skip("Bash script behavioral execution requires POSIX environment (Linux CI)")

    preflight_script = REPO_ROOT / "scripts" / "deploy" / "asus-preflight.sh"
    env_file = tmp_path / ".env"
    env_file.write_text("RBTA_API_KEY=\nRBTA_MODEL_VERSION=deploy-smoke-v1\n", encoding="utf-8")

    env = {**os.environ, "RBTA_ENV_FILE": str(env_file)}
    res = subprocess.run(
        ["bash", str(preflight_script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "preflight must exit non-zero when RBTA_API_KEY is empty"
    assert "RBTA_API_KEY is missing or empty" in res.stderr


def test_preflight_behavioral_empty_model_version(tmp_path: Path):
    """Behavioral test: asus-preflight.sh fails when RBTA_MODEL_VERSION is empty in .env."""
    if os.name != "posix":
        pytest.skip("Bash script behavioral execution requires POSIX environment (Linux CI)")

    preflight_script = REPO_ROOT / "scripts" / "deploy" / "asus-preflight.sh"
    env_file = tmp_path / ".env"
    env_file.write_text("RBTA_API_KEY=ci-valid-key-placeholder\nRBTA_MODEL_VERSION=\n", encoding="utf-8")

    env = {**os.environ, "RBTA_ENV_FILE": str(env_file)}
    res = subprocess.run(
        ["bash", str(preflight_script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "preflight must exit non-zero when RBTA_MODEL_VERSION is empty"
    assert "RBTA_MODEL_VERSION is missing or empty" in res.stderr


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


def test_production_spa_static_serving(tmp_path: Path):
    """Verify production static SPA serving with assets mounting and client-route fallback."""
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)

    index_html = dist_dir / "index.html"
    index_html.write_text("<!DOCTYPE html><html><body><div id='root'>RBTA Dashboard</div></body></html>", encoding="utf-8")

    bundle_js = assets_dir / "index-test.js"
    bundle_js.write_text("console.log('rbta dashboard bundle');", encoding="utf-8")

    os.environ["RBTA_DASHBOARD_DIST"] = str(dist_dir)
    try:
        app = create_app()
        client = TestClient(app)

        # 1. Root redirect to /dashboard/
        root_resp = client.get("/", follow_redirects=False)
        assert root_resp.status_code == 307
        assert root_resp.headers["location"] == "/dashboard/"

        # 2. /dashboard/ serves index.html
        dash_resp = client.get("/dashboard/")
        assert dash_resp.status_code == 200
        assert "RBTA Dashboard" in dash_resp.text

        # 3. /dashboard/assets/index-test.js serves JS asset
        asset_resp = client.get("/dashboard/assets/index-test.js")
        assert asset_resp.status_code == 200
        assert "console.log('rbta dashboard bundle');" in asset_resp.text

        # 4. Nested SPA client routes fallback to index.html
        spa_resp = client.get("/dashboard/meta-alerts/101/raw-alerts/wazuh-alt-001")
        assert spa_resp.status_code == 200
        assert "RBTA Dashboard" in spa_resp.text

        # 5. Missing API route does NOT fallback to index.html, returns JSON 404
        api_resp = client.get("/api/v1/nonexistent-endpoint")
        assert api_resp.status_code == 404
        assert api_resp.headers.get("content-type") == "application/json"
    finally:
        os.environ.pop("RBTA_DASHBOARD_DIST", None)
