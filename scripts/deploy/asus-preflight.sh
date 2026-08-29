#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — ASUS Production Deployment Preflight Validator
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy/asus"

echo "=== [1/5] Resolving Environment Configuration ==="
ENV_FILE="${RBTA_ENV_FILE:-${DEPLOY_DIR}/.env}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: Environment configuration file not found at: ${ENV_FILE}" >&2
  echo "       Please create it by copying the template:" >&2
  echo "       cp ${ROOT_DIR}/.env.example ${ENV_FILE}" >&2
  exit 1
fi
echo "Environment file resolved: ${ENV_FILE}"

# Load environment variables securely
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

echo "=== [2/5] Validating Mandatory Fail-Closed Variables ==="
if [ -z "${RBTA_API_KEY:-}" ]; then
  echo "ERROR: RBTA_API_KEY is missing or empty in ${ENV_FILE}." >&2
  echo "       Production deployment requires a secure, non-empty API key." >&2
  exit 1
fi
echo "RBTA_API_KEY: configured (non-empty)"

if [ -z "${RBTA_MODEL_VERSION:-}" ]; then
  echo "ERROR: RBTA_MODEL_VERSION is missing or empty in ${ENV_FILE}." >&2
  echo "       Production deployment requires an explicit model version (no automatic fallback)." >&2
  exit 1
fi
echo "RBTA_MODEL_VERSION: ${RBTA_MODEL_VERSION}"

echo "=== [3/5] Validating Model Artifact Registry ==="
MODELS_DIR="${DEPLOY_DIR}/models"
if [ ! -d "${MODELS_DIR}/${RBTA_MODEL_VERSION}" ]; then
  echo "ERROR: Model directory '${MODELS_DIR}/${RBTA_MODEL_VERSION}' does not exist." >&2
  echo "       Please stage the immutable model bundle before deploying." >&2
  exit 1
fi

python "${SCRIPT_DIR}/validate_model.py" --models-dir "${MODELS_DIR}" --version "${RBTA_MODEL_VERSION}"

echo "=== [4/5] Validating State Directory Permissions ==="
STATE_DIR="${DEPLOY_DIR}/state"
mkdir -p "${STATE_DIR}"

# Validate write accessibility on host state directory
TEST_FILE="${STATE_DIR}/.preflight_write_test_$$"
if ! touch "${TEST_FILE}" 2>/dev/null; then
  echo "ERROR: Host state directory '${STATE_DIR}' is not writable by current user." >&2
  echo "       Remediation (run as root on host):" >&2
  echo "       sudo chown -R 10001:10001 ${STATE_DIR}" >&2
  echo "       sudo chmod 0750 ${STATE_DIR}" >&2
  exit 1
fi
rm -f "${TEST_FILE}"
echo "Host state directory write check: OK"

echo "=== [5/5] Validating Container Engine & Compose Specification ==="
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker command not found." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose not available." >&2; exit 1; }

docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" config --quiet
echo "Docker Compose manifest: VALID"

echo ""
echo ">>> PREFLIGHT CHECK COMPLETED: ALL DEPLOYMENT PREREQUISITES SATISFIED <<<"
