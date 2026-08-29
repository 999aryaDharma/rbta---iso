#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — ASUS Production Deployment Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy/asus"
ENV_FILE="${RBTA_ENV_FILE:-${DEPLOY_DIR}/.env}"

echo "=== [Phase 1/6] Running Deployment Preflight Checks ==="
bash "${SCRIPT_DIR}/asus-preflight.sh"

# Source configuration
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

STATE_DIR="${RBTA_STATE_HOST_DIR:-${ROOT_DIR}/state}"
MODELS_DIR="${RBTA_MODEL_HOST_DIR:-${ROOT_DIR}/models}"
REPLAY_DIR="${RBTA_REPLAY_HOST_DIR:-${ROOT_DIR}/data/replay}"
IMAGE_TAG="${RBTA_IMAGE_TAG:?RBTA_IMAGE_TAG is required}"

echo "=== [Phase 2/6] Verifying Code SHA Provenance & Ancestry ==="
CODE_SHA="${RBTA_CODE_SHA:-}"
if [ -z "${CODE_SHA}" ] && [ -f "${ROOT_DIR}/.agents/campaign/STATE.json" ]; then
  # Extract tested code sha from campaign state
  CODE_SHA=$(grep -oE '"code_sha_tested": "[^"]+"' "${ROOT_DIR}/.agents/campaign/STATE.json" | cut -d '"' -f4 || true)
fi
if [ -z "${CODE_SHA}" ]; then
  CODE_SHA=$(git -C "${ROOT_DIR}" rev-parse HEAD)
fi

echo "Verifying git ancestry of CODE_SHA=${CODE_SHA}..."
if ! git -C "${ROOT_DIR}" merge-base --is-ancestor "${CODE_SHA}" HEAD; then
  echo "ERROR: CODE_SHA ${CODE_SHA} is not an ancestor of current HEAD." >&2
  exit 1
fi
echo "✓ CODE_SHA ancestry verified: ${CODE_SHA}"

export RBTA_CODE_SHA="${CODE_SHA}"
export RBTA_BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

echo "=== [Phase 3/6] Building Production Container Image ==="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" build

# Verify container image label provenance
IMG_LABEL=$(docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision"}}' "rbta-service:${IMAGE_TAG}" 2>/dev/null || true)
if [ -n "${IMG_LABEL}" ] && [ "${IMG_LABEL}" != "${CODE_SHA}" ]; then
  echo "ERROR: Built image label revision '${IMG_LABEL}' does not match CODE_SHA '${CODE_SHA}'" >&2
  exit 1
fi
echo "✓ Container image provenance verified: rbta-service:${IMAGE_TAG}"

echo "=== [Phase 4/6] Running Containerized Pre-Start Validation as UID 10001 ==="
docker run --rm \
  --user "10001:10001" \
  -v "${MODELS_DIR}:/app/artifacts/models:ro" \
  -v "${REPLAY_DIR}:/app/data/replay:ro" \
  -v "${STATE_DIR}:/app/data/runtime:rw" \
  "rbta-service:${IMAGE_TAG}" \
  python scripts/deploy/validate_model.py --models-dir /app/artifacts/models --version "${RBTA_MODEL_VERSION}"
echo "✓ Model artifact and filesystem permissions verified inside container"

echo "=== [Phase 5/6] Starting RBTA Service ==="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" up -d

echo "=== [Phase 6/6] Probing Health & Running Read-Only Smoke Gates ==="
PORT="${RBTA_HOST_PORT:?RBTA_HOST_PORT is required}"
MAX_RETRIES=30
RETRY_COUNT=0

echo -n "Waiting for /ready probe..."
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if curl -s -f "http://127.0.0.1:${PORT}/ready" >/dev/null 2>&1; then
    echo " OK!"
    break
  fi
  echo -n "."
  sleep 1
  RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo " FAILED: /ready returned non-200 or timed out"
  docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" logs --tail=50
  exit 1
fi

echo "Running full read-only smoke suite..."
bash "${SCRIPT_DIR}/smoke.sh"

echo ""
echo "=========================================================================="
echo ">>> ASUS DEPLOYMENT SUCCESSFUL: All Gates and Observability Checks Passed <<<"
echo "=========================================================================="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" ps
