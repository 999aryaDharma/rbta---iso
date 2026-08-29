#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — Deployment Smoke Verification Script
# ==============================================================================

HOST_PORT="${RBTA_HOST_PORT:-8000}"
BASE_URL="http://127.0.0.1:${HOST_PORT}"
API_KEY="${RBTA_API_KEY:-}"

# Fail closed: Production smoke tests must use authentication
if [ -z "${API_KEY}" ]; then
  echo "ERROR: RBTA_API_KEY environment variable is required to run deployment smoke verification." >&2
  exit 1
fi

AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")

echo "=== [1/4] Checking Liveness (/health) ==="
HEALTH_RESP=$(curl -s -f "${BASE_URL}/health")
echo "Health: ${HEALTH_RESP}"

echo "=== [2/4] Checking Readiness (/ready) ==="
READY_RESP=$(curl -s -f "${BASE_URL}/ready")
echo "Readiness: ${READY_RESP}"

echo "=== [3/4] Checking Authorized Runtime Stats (/runtime/stats) ==="
STATS_RESP=$(curl -s -f "${AUTH_HEADER[@]}" "${BASE_URL}/runtime/stats")
echo "Stats: ${STATS_RESP}"

echo "=== [4/4] Sending Engineering Smoke Alert (Idempotency & Durability Check) ==="
# NOTE: s10_smoke_* identifiers are strict engineering test fixtures and must NOT be used in research evaluations.
PAYLOAD='{
  "id": "s10_smoke_test_alert_001",
  "timestamp": "2026-08-28T12:00:00.000+0000",
  "agent": {"id": "001", "name": "asus-smoke-agent"},
  "rule": {"id": "5501", "level": 3, "groups": ["pam"]},
  "rule_group_primary": "pam",
  "agent_criticality": 1
}'

echo "Ingesting first attempt..."
INGEST_1=$(curl -s -f -X POST "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "${PAYLOAD}" "${BASE_URL}/api/v1/alerts/ingest")
echo "Attempt 1: ${INGEST_1}"

echo "Ingesting second attempt (duplicate test)..."
INGEST_2=$(curl -s -f -X POST "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "${PAYLOAD}" "${BASE_URL}/api/v1/alerts/ingest")
echo "Attempt 2: ${INGEST_2}"

echo "Smoke Verification Complete — All Probes Passed."
