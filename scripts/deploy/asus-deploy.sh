#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — ASUS Production Deployment Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/deploy/asus"

echo "=== [1/6] Preflight Checks ==="
command -v docker >/dev/null 2>&1 || { echo "Error: docker is required but not installed." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Error: docker compose is required." >&2; exit 1; }

echo "=== [2/6] Preparing Deployment Host Directories ==="
mkdir -p "${DEPLOY_DIR}/state" "${DEPLOY_DIR}/models"
# Ensure non-root app user (10001:10001) can write to state directory
chmod -R 775 "${DEPLOY_DIR}/state" || true

echo "=== [3/6] Validating Compose Configuration ==="
docker compose -f "${DEPLOY_DIR}/compose.yml" config --quiet

echo "=== [4/6] Building Production Container Image ==="
docker compose -f "${DEPLOY_DIR}/compose.yml" build

echo "=== [5/6] Starting RBTA Service ==="
docker compose -f "${DEPLOY_DIR}/compose.yml" up -d

echo "=== [6/6] Probing Health & Readiness ==="
PORT="${RBTA_HOST_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
READY_URL="http://127.0.0.1:${PORT}/ready"

MAX_RETRIES=30
RETRY_COUNT=0

echo -n "Waiting for /health..."
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if docker compose -f "${DEPLOY_DIR}/compose.yml" exec -T rbta-service python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >/dev/null 2>&1; then
    echo " OK!"
    break
  fi
  echo -n "."
  sleep 1
  RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo " FAILED to reach /health"
  docker compose -f "${DEPLOY_DIR}/compose.yml" logs --tail=50
  exit 1
fi

echo "Deployment Successful!"
docker compose -f "${DEPLOY_DIR}/compose.yml" ps
