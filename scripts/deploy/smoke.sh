#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RBTA + Isolation Forest — Read-Only Production Deployment Smoke Verification
# ==============================================================================
# This script performs STRICTLY READ-ONLY observability, static asset, and health checks.
# It NEVER mutates production telemetry, raw evidence, or outbox state.

ENV_FILE="${RBTA_ENV_FILE:-deploy/asus/.env}"
if [ -f "${ENV_FILE}" ]; then
  # Deterministically extract PORT, API_KEY, MODEL_VERSION without arbitrary shell eval
  if [ -z "${RBTA_HOST_PORT:-}" ]; then
    VAL=$(grep -E '^[[:space:]]*RBTA_HOST_PORT=' "${ENV_FILE}" | tail -n 1 | cut -d '=' -f2- | tr -d ' "\r\n')
    if [ -n "${VAL}" ]; then RBTA_HOST_PORT="${VAL}"; fi
  fi
  if [ -z "${RBTA_API_KEY:-}" ]; then
    VAL=$(grep -E '^[[:space:]]*RBTA_API_KEY=' "${ENV_FILE}" | tail -n 1 | cut -d '=' -f2- | tr -d ' "\r\n')
    if [ -n "${VAL}" ]; then RBTA_API_KEY="${VAL}"; fi
  fi
  if [ -z "${RBTA_MODEL_VERSION:-}" ]; then
    VAL=$(grep -E '^[[:space:]]*RBTA_MODEL_VERSION=' "${ENV_FILE}" | tail -n 1 | cut -d '=' -f2- | tr -d ' "\r\n')
    if [ -n "${VAL}" ]; then RBTA_MODEL_VERSION="${VAL}"; fi
  fi
fi

if [ -z "${RBTA_HOST_PORT:-}" ]; then
  echo "ERROR: RBTA_HOST_PORT is required (no default port 8000 allowed)." >&2
  exit 1
fi

if [ -z "${RBTA_API_KEY:-}" ]; then
  echo "ERROR: RBTA_API_KEY environment variable is required to run smoke verification." >&2
  exit 1
fi

BASE_URL="http://127.0.0.1:${RBTA_HOST_PORT}"
AUTH_HEADER=(-H "Authorization: Bearer ${RBTA_API_KEY}")

echo "=== [1/11] Checking Liveness Probe (/health) ==="
HEALTH_RESP=$(curl -s -S -f "${BASE_URL}/health")
echo "✓ Health: ${HEALTH_RESP}"

echo "=== [2/11] Checking Readiness Probe (/ready) ==="
READY_RESP=$(curl -s -S -f "${BASE_URL}/ready")
echo "✓ Readiness: ${READY_RESP}"

echo "=== [3/11] Verifying Unauthorized Rejection (/api/v1/auth/check) ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/v1/auth/check" || true)
if [ "${HTTP_CODE}" != "401" ]; then
  echo "ERROR: Expected 401 Unauthorized for unauthenticated request, got ${HTTP_CODE}" >&2
  exit 1
fi
echo "✓ Unauthenticated request correctly rejected with 401"

echo "=== [4/11] Verifying Authorized Access (/api/v1/auth/check) ==="
AUTH_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/api/v1/auth/check")
echo "✓ Auth verified: ${AUTH_RESP}"

echo "=== [5/11] Checking Runtime Statistics (/runtime/stats) ==="
STATS_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/runtime/stats")
echo "✓ Runtime Stats: ${STATS_RESP}"

echo "=== [6/11] Checking Production Dashboard Serving (/dashboard/) ==="
DASH_BODY=$(curl -s -S -f "${BASE_URL}/dashboard/")
if [[ "${DASH_BODY}" != *'id="root"'* ]]; then
  echo "ERROR: /dashboard/ did not return expected HTML entrypoint" >&2
  exit 1
fi
echo "✓ Dashboard HTML entrypoint served successfully"

echo "=== [7/11] Checking Static JS Bundle Serving ==="
JS_PATH=$(echo "${DASH_BODY}" | grep -oE '/dashboard/assets/index-[^"]+\.js' | head -n 1 || true)
if [ -n "${JS_PATH}" ]; then
  JS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${JS_PATH}")
  if [ "${JS_CODE}" != "200" ]; then
    echo "ERROR: Static asset ${JS_PATH} returned HTTP ${JS_CODE}" >&2
    exit 1
  fi
  echo "✓ Static asset ${JS_PATH} served successfully (200 OK)"
else
  echo "Notice: No explicit index-*.js script found in dashboard HTML, skipped asset fetch"
fi

echo "=== [8/11] Checking Nested SPA Route Fallback ==="
SPA_BODY=$(curl -s -S -f "${BASE_URL}/dashboard/meta-alerts/101/raw-alerts/wazuh-sample-001")
if [[ "${SPA_BODY}" != *'id="root"'* ]]; then
  echo "ERROR: Nested SPA route did not return fallback HTML" >&2
  exit 1
fi
echo "✓ Nested SPA route served successfully"

echo "=== [9/11] Checking Replay Datasets Discovery (/api/v1/replay/datasets) ==="
DATASETS_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/api/v1/replay/datasets")
echo "✓ Datasets: ${DATASETS_RESP}"

echo "=== [10/11] Checking System Metadata Truth (/api/v1/dashboard/system) ==="
SYS_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/api/v1/dashboard/system")
echo "✓ System Metadata: ${SYS_RESP}"
if [ -n "${RBTA_MODEL_VERSION:-}" ]; then
  if [[ "${SYS_RESP}" != *"${RBTA_MODEL_VERSION}"* ]]; then
    echo "ERROR: System metadata does not contain configured model_version '${RBTA_MODEL_VERSION}'" >&2
    exit 1
  fi
  echo "✓ Model version truth verified in system metadata"
fi

echo "=== [11/11] Checking Integrations Truthful Status (/api/v1/dashboard/integrations) ==="
INTEG_RESP=$(curl -s -S -f "${AUTH_HEADER[@]}" "${BASE_URL}/api/v1/dashboard/integrations")
echo "✓ Integrations Status: ${INTEG_RESP}"

echo "=============================================================================="
echo "Production Smoke Verification Complete — ALL 11 READ-ONLY PROBES PASSED"
echo "=============================================================================="
