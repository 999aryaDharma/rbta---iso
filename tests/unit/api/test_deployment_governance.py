"""Governance and security tests for production deployment manifests, Dockerfile, and Compose."""

from pathlib import Path
import subprocess

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
