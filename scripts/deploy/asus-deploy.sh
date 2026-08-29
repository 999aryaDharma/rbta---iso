#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — ASUS Production Deployment Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy/asus"
ENV_FILE="${RBTA_ENV_FILE:-${DEPLOY_DIR}/.env}"

echo "=== [Phase 1/4] Running Deployment Preflight Checks ==="
bash "${SCRIPT_DIR}/asus-preflight.sh"

echo "=== [Phase 2/4] Building Production Container Image ==="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" build

echo "=== [Phase 3/4] Starting RBTA Service ==="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" up -d

echo "=== [Phase 4/4] Probing Health & Readiness Gates ==="
PORT="${RBTA_HOST_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
READY_URL="http://127.0.0.1:${PORT}/ready"

MAX_RETRIES=30

# 1. Liveness Gate (/health)
echo -n "Waiting for /health (process liveness)..."
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" exec -T rbta-service python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >/dev/null 2>&1; then
    echo " OK!"
    break
  fi
  echo -n "."
  sleep 1
  RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo " FAILED to reach /health"
  docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" logs --tail=50
  exit 1
fi

# 2. Readiness Gate (/ready)
echo -n "Waiting for /ready (model bundle loaded & pipeline ready)..."
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" exec -T rbta-service python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" >/dev/null 2>&1; then
    echo " OK!"
    break
  fi
  echo -n "."
  sleep 1
  RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo " FAILED: /ready returned 503 or timed out (active model not loaded properly)"
  docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" logs --tail=50
  exit 1
fi

echo ""
echo "=========================================================================="
echo ">>> DEPLOYMENT SUCCESSFUL: Health and Readiness Probes Both Passed. <<<"
echo "=========================================================================="
docker compose --env-file "${ENV_FILE}" -f "${DEPLOY_DIR}/compose.yml" ps
