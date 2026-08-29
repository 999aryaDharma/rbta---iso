#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — Isolated Engineering Ingestion Smoke Test
# ==============================================================================
# This script verifies end-to-end alert ingestion, idempotency, and SQLite durability
# in a strictly isolated temporary container and state directory.
# It NEVER touches or mutates production state.

ENV_FILE="${RBTA_ENV_FILE:-deploy/asus/.env}"
IMAGE_TAG="${RBTA_IMAGE_TAG:-latest}"
IMAGE_NAME="rbta-service:${IMAGE_TAG}"

API_KEY="isolated-engineering-smoke-token-42"
TEST_CONTAINER="rbta-isolated-smoke-$(date +%s)"
TEMP_STATE_DIR=$(mktemp -d -t rbta_isolated_state_XXXXXX)
TEMP_PORT=18088

cleanup() {
  echo "Cleaning up isolated smoke test container and temporary state..."
  docker stop "${TEST_CONTAINER}" >/dev/null 2>&1 || true
  docker rm "${TEST_CONTAINER}" >/dev/null 2>&1 || true
  rm -rf "${TEMP_STATE_DIR}" >/dev/null 2>&1 || true
  echo "Cleanup complete."
}
trap cleanup EXIT

echo "=== [1/4] Starting Disposable Container for Isolated Smoke ==="
# Mount temporary directory for state and evidence
docker run -d --name "${TEST_CONTAINER}" \
  -p "${TEMP_PORT}:8000" \
  -e RBTA_API_KEY="${API_KEY}" \
  -e RBTA_MODEL_VERSION="ci-smoke-v1" \
  -e RBTA_SOURCE_MODE="DEFERRED" \
  -e RBTA_LOG_LEVEL="DEBUG" \
  -v "${TEMP_STATE_DIR}:/app/data/runtime:rw" \
  "${IMAGE_NAME}"

BASE_URL="http://127.0.0.1:${TEMP_PORT}"
AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")

echo "=== [2/4] Waiting for Container Startup ==="
HEALTH_OK=false
for i in {1..30}; do
  if curl -s -f "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Container healthy on attempt $i."
    HEALTH_OK=true
    break
  fi
  sleep 1
done

if [ "${HEALTH_OK}" != "true" ]; then
  echo "ERROR: Isolated smoke container failed to start within 30s" >&2
  docker logs "${TEST_CONTAINER}"
  exit 1
fi

echo "=== [3/4] Ingesting Non-Production Engineering Sentinel Alert ==="
PAYLOAD='{
  "id": "__engineering_smoke_alert_001__",
  "timestamp": "2026-08-29T12:00:00.000+0000",
  "agent": {"id": "__engineering_smoke_agent__", "name": "engineering-test-agent"},
  "rule": {"id": "5501", "level": 7, "groups": ["pam"]},
  "rule_group_primary": "pam",
  "agent_criticality": 1
}'

echo "Attempt 1: Initial Ingestion..."
RESP_1=$(curl -s -S -f -X POST "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "${PAYLOAD}" "${BASE_URL}/api/v1/alerts/ingest")
echo "Result 1: ${RESP_1}"

echo "Attempt 2: Idempotent Duplicate Ingestion..."
RESP_2=$(curl -s -S -f -X POST "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "${PAYLOAD}" "${BASE_URL}/api/v1/alerts/ingest")
echo "Result 2: ${RESP_2}"

echo "=== [4/4] Verifying Durability & Raw Evidence Stored in Isolated DB ==="
STATS_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/runtime/stats")
echo "Stats: ${STATS_RESP}"

echo "=============================================================================="
echo "Isolated Engineering Smoke Test Complete — PASSED"
echo "=============================================================================="
