#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — ASUS Production Deployment Preflight Validator
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy/asus"

echo "=== [1/6] Resolving Environment Configuration ==="
ENV_FILE="${RBTA_ENV_FILE:-${DEPLOY_DIR}/.env}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: Environment configuration file not found at: ${ENV_FILE}" >&2
  echo "       Please create it by copying the template:" >&2
  echo "       cp ${DEPLOY_DIR}/.env.example ${ENV_FILE}" >&2
  exit 1
fi
echo "Environment file resolved: ${ENV_FILE}"

# Load environment variables safely
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

echo "=== [2/6] Validating Mandatory Fail-Closed Variables ==="
if [ -z "${RBTA_API_KEY:-}" ]; then
  echo "ERROR: RBTA_API_KEY is missing or empty in ${ENV_FILE}." >&2
  exit 1
fi
echo "✓ RBTA_API_KEY: configured (non-empty)"

if [ -z "${RBTA_MODEL_VERSION:-}" ]; then
  echo "ERROR: RBTA_MODEL_VERSION is missing or empty in ${ENV_FILE}." >&2
  exit 1
fi
echo "✓ RBTA_MODEL_VERSION: ${RBTA_MODEL_VERSION}"

echo "=== [3/6] Validating Host Port Configuration & Collision ==="
if [ -z "${RBTA_HOST_PORT:-}" ]; then
  echo "ERROR: RBTA_HOST_PORT is required in ${ENV_FILE} (default port 8000 is disallowed)." >&2
  exit 1
fi

if ! [[ "${RBTA_HOST_PORT}" =~ ^[0-9]+$ ]] || [ "${RBTA_HOST_PORT}" -lt 1024 ] || [ "${RBTA_HOST_PORT}" -gt 65535 ]; then
  echo "ERROR: RBTA_HOST_PORT must be an integer between 1024 and 65535, got '${RBTA_HOST_PORT}'." >&2
  exit 1
fi
echo "✓ RBTA_HOST_PORT: ${RBTA_HOST_PORT} (valid range)"

# Check if port is already bound by an unrelated process
if command -v ss >/dev/null 2>&1; then
  if ss -tulpn | grep -q ":${RBTA_HOST_PORT} "; then
    echo "WARNING: Port ${RBTA_HOST_PORT} appears to be in use. Ensure existing container will be cleanly replaced."
  fi
elif command -v netstat >/dev/null 2>&1; then
  if netstat -tuln | grep -q ":${RBTA_HOST_PORT} "; then
    echo "WARNING: Port ${RBTA_HOST_PORT} appears to be in use. Ensure existing container will be cleanly replaced."
  fi
fi

echo "=== [4/6] Validating Host Directory Paths & Permissions ==="
STATE_DIR="${RBTA_STATE_HOST_DIR:-${ROOT_DIR}/state}"
MODELS_DIR="${RBTA_MODEL_HOST_DIR:-${ROOT_DIR}/models}"
REPLAY_DIR="${RBTA_REPLAY_HOST_DIR:-${ROOT_DIR}/data/replay}"

mkdir -p "${STATE_DIR}"
echo "✓ State directory: ${STATE_DIR}"

if [ ! -d "${MODELS_DIR}/${RBTA_MODEL_VERSION}" ]; then
  echo "ERROR: Model directory '${MODELS_DIR}/${RBTA_MODEL_VERSION}' does not exist." >&2
  exit 1
fi
echo "✓ Model registry directory: ${MODELS_DIR}/${RBTA_MODEL_VERSION}"

if [ ! -d "${REPLAY_DIR}" ]; then
  echo "ERROR: Replay directory '${REPLAY_DIR}' does not exist." >&2
  exit 1
fi
echo "✓ Replay archive directory: ${REPLAY_DIR}"

echo "=== [5/6] Validating Container Engine & Compose Specification ==="
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker command not found." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose not available." >&2; exit 1; }

docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" config --quiet
echo "✓ Docker Compose manifest: VALID"

echo "=== [6/6] Checking Source Mode Invariant ==="
SOURCE_MODE="${RBTA_SOURCE_MODE:-DEFERRED}"
if [ "${SOURCE_MODE}" != "DEFERRED" ] && [ "${SOURCE_MODE}" != "LIVE" ]; then
  echo "ERROR: RBTA_SOURCE_MODE must be 'DEFERRED' or 'LIVE', got '${SOURCE_MODE}'." >&2
  exit 1
fi
echo "✓ Source mode: ${SOURCE_MODE}"

echo ""
echo ">>> PREFLIGHT CHECK COMPLETED: ALL DEPLOYMENT PREREQUISITES SATISFIED <<<"
