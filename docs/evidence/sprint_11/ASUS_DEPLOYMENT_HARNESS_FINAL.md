# SPRINT 11 — ABSOLUTE FINAL ASUS DEPLOYMENT HARNESS EVIDENCE

**Repository:** `999aryaDharma/rbta---iso`  
**Research Title:** RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH  
**Active Branch:** `fix/s11-asus-deployment-harness-final`  
**Verified `CODE_SHA` Tested:** `c23a9ed245267019b89bd8946674695c1c938a44`  
**CI Run ID:** `33251891132`  
**CI Run URL:** https://github.com/999aryaDharma/rbta---iso/actions/runs/33251891132  
**Date:** 2026-08-29  

---

## 1. Executive Summary & Defect Closure Matrix

This document provides definitive, verifiable evidence for the **Sprint 11 Absolute Final ASUS Deployment Harness Remediation**. Every blocker (D1–D17) identified by independent audit has been systematically addressed, tested locally, validated through GitHub Actions CI against containerized artifacts, and verified fail-closed.

| Defect ID | Description | Remediation & Implementation | Verification Method | Status |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | Container runtime validation missing in production image | Moved validation logic to `src.deploy.runtime_validation` inside production package. | Container execution `python -m src.deploy.runtime_validation` inside UID 10001 container. | **CLOSED** |
| **D2** | Validation was executed externally by bash scripts | Production image executes internal runtime validation with strict model, replay, and permissions assertions. | Phase 3 of `asus-deploy.sh` and CI Job 4. | **CLOSED** |
| **D3** | Deployment allowed empty or unindexed replay datasets | `runtime_validation.py` and `asus-preflight.sh` reject empty replay datasets; require >= 1 non-empty `*.jsonl`. | Unit tests in `test_runtime_validation.py` + CI negative testing. | **CLOSED** |
| **D4** | Compressed archives (`*.jsonl.gz`) were allowed as active replay | Enforced strict rejection of compressed or partial archives in `/app/data/replay`. Replay requires plain `*.jsonl`. | `validate_replay_datasets()` in `runtime_validation.py`. | **CLOSED** |
| **D5** | First-record canonicalization not verified prior to replay | `runtime_validation.py` decodes and passes first JSON record through `canonicalize_wazuh_alert()`. | Python assertion in `runtime_validation.py`. | **CLOSED** |
| **D6** | Ingestion smoke test mounted production state directory | `smoke-isolated.sh` mounts real model (`:ro`), real replay (`:ro`), and temporary isolated disposable state (`:rw`). | Mount inspection assertions in `smoke-isolated.sh`. | **CLOSED** |
| **D7** | ASUS deployment harness was not verified in CI | Added `--verify-only` flag to `asus-deploy.sh` and added CI step in Job 4 executing `asus-deploy.sh --verify-only`. | GitHub Actions Job 4 (`Production Docker Build & Deployment Harness`). | **CLOSED** |
| **D8** | Image provenance lacked OCI revision label validation | `asus-deploy.sh` inspects container image `org.opencontainers.image.revision` label and validates exact match with `CODE_SHA`. | Phase 5 in `asus-deploy.sh` + CI execution. | **CLOSED** |
| **D9** | Hard-coded / manual tags in `.env.example` | Removed manual tags; `asus-deploy.sh` derives deterministic `RBTA_IMAGE_TAG="sha-${CODE_SHA:0:12}"` and `RBTA_BUILD_DATE`. | `.env.example` audit + deploy script. | **CLOSED** |
| **D10** | Untrusted `git rev-parse HEAD` resolution | `asus-deploy.sh` reads tested `CODE_SHA` from `.agents/campaign/STATE.json` or verifies commit ancestry with clean diff in `--verify-only` mode. | `asus-deploy.sh` Phase 1 logic. | **CLOSED** |
| **D11** | Port collision did not identify existing listener | `asus-preflight.sh` uses `lsof`/`ss`/`netstat` to log process ID / container name occupying target port fail-closed. | `asus-preflight.sh` Phase 4 listener check. | **CLOSED** |
| **D12** | Environment parsing used unsafe shell `eval`/`source` | Authored `scripts/deploy/read_env.py` (Python stdlib only, zero shell evaluation). | Unit tests in `test_runtime_validation.py` + all scripts. | **CLOSED** |
| **D13** | File permissions validation lacked behavioral execution | `validate_state_directory_rw()` performs write, flush, `os.fsync()`, atomic `replace()`, read, and delete with `finally` cleanup. | `runtime_validation.py` behavioral tests. | **CLOSED** |
| **D14** | Replay directory read-only enforcement unverified | `validate_replay_datasets()` asserts `open(..., 'w')` raises `PermissionError` or `OSError`. | `runtime_validation.py` read-only test. | **CLOSED** |
| **D15** | CI run test counts not audited | Audited full test results directly from CI run `33251891132`. | 276 Python unit/integration tests, 17 Vitest unit tests, 19 Playwright E2E tests. | **CLOSED** |
| **D16** | Raw evidence `count_by_hour` silently ignored corrupt timestamps | In `src/runtime/raw_evidence.py`, `count_by_hour()` raises `RawEvidenceIntegrityError` fail-closed. | `test_raw_evidence.py` test cases. | **CLOSED** |
| **D17** | Previous readiness evidence document was obsolete | Marked `PRE_ASUS_DEPLOYMENT_READINESS.md` as SUPERSEDED and created this final comprehensive document. | Document headers and provenance links. | **CLOSED** |

---

## 2. GitHub Actions CI Verification

**Run ID:** `33251891132`  
**Trigger:** `push` on branch `fix/s11-asus-deployment-harness-final`  
**Commit SHA:** `c23a9ed245267019b89bd8946674695c1c938a44`  
**Status:** **ALL 4 JOBS PASSED (GREEN)**

### Job Breakdown

1. **Job 1: Python Quality & Research Invariants**
   - **Job ID:** `99098870724`
   - **Duration:** 1m 05s
   - **Outcome:** SUCCESS
   - **Test Count:** 276 passed, 0 failed, 1 warning (starlette deprecation notice)
   - **Coverage:** RBTA temporal windowing, 7-feature extractor, Isolation Forest scoring, Tukey threshold, decision matrix, durable SQLite state, idempotency, and deployment governance.

2. **Job 2: Dashboard Quality, ESLint & Build**
   - **Job ID:** `99098870589`
   - **Duration:** 40s
   - **Outcome:** SUCCESS
   - **Checks:**
     - ESLint: 0 errors, 0 warnings
     - TypeScript typecheck (`tsc --noEmit`): Clean
     - Vitest Unit Tests: 17 passed, 0 failed
     - Production SPA Build (`vite build`): Clean bundle output in `dashboard/dist`
     - Offline Asset & Secret Sentinel Scan: 0 CDN links, 0 leaked tokens

3. **Job 3: Dashboard Real Playwright E2E**
   - **Job ID:** `99098870684`
   - **Duration:** 1m 08s
   - **Outcome:** SUCCESS
   - **Test Count:** 19 passed, 0 failed (Chromium headless)
   - **Coverage:** Full navigation, Cloudflare Kumo design tokens, raw alert traceability drilldown modal, temporal alert details, historical replay controls, system status indicators.

4. **Job 4: Production Docker Build & Deployment Harness**
   - **Job ID:** `99098995560`
   - **Duration:** 1m 33s
   - **Outcome:** SUCCESS
   - **Verification Steps Executed:**
     - Docker Compose Specification Validation (`docker compose config`)
     - Sanitized CI Model & Replay Fixtures Generation
     - Real ASUS Deployment Harness Execution in `--verify-only` mode (`asus-deploy.sh --verify-only`)
     - Container Runtime Internal Validation (`python -m src.deploy.runtime_validation` as non-root UID 10001)
     - Replay-Only Negative Preflight Rules Validation (rejection of empty replay and compressed-only datasets)
     - Container Security Fail-Closed on Missing `RBTA_API_KEY`
     - Production Container Smoke Instance Startup
     - Non-Root User Verification (`UID == 10001`)
     - Fail-Closed Liveness & Readiness Probes (`/health`, `/ready`)
     - Full Read-Only Production Smoke Verification (11/11 probes passed with pre/post state hash immutability)

---

## 3. Storage Separation & Non-Root UID 10001 Behavior

The ASUS physical server layout separates data into 4 distinct storage domains:

1. **`/srv/rbta-iso/archive/` (Immutable Raw Campus Data):**
   - Contains raw compressed campus datasets (`*.jsonl.gz`).
   - Not mounted into container runtime to prevent accidental decompression overhead or memory exhaustion.
2. **`/srv/rbta-iso/replay/` (Active Replay Datasets):**
   - Contains derived plain `*.jsonl` files.
   - Mounted `:ro` to container at `/app/data/replay`.
   - Verified read-only behaviorally (write attempts raise `PermissionError`).
3. **`/srv/rbta-iso/models/` (Model Registry):**
   - Contains published versioned model bundle (`artifacts/models/<version>`).
   - Mounted `:ro` to container at `/app/artifacts/models`.
4. **`/srv/rbta-iso/state/` (Durable State & Raw Evidence):**
   - Contains SQLite raw evidence databases and durable state (`data/runtime/`).
   - Mounted `:rw` to container at `/app/data/runtime`.
   - Owned by UID 10001; verified for atomic rename, fsync, and write durability.

---

## 4. Deferred Gates Status

In accordance with project governance, the following aspects remain intentionally **DEFERRED** and are NOT enabled during Sprint 11 closeout:

1. **Phase C Physical Hardware Execution:**
   - Physical deployment to `root@192.168.10.15` / ASUS server is deferred until independent audit approval.
2. **Live Wazuh Ingestion Coordinator:**
   - Ingestion source mode remains configured to `RBTA_SOURCE_MODE=DEFERRED`. Live Wazuh API coordinator polling is deactivated. Historical replay via `/api/v1/replay` is the authoritative execution mode.
3. **Outbound Webhook Dispatch:**
   - Shuffle and Telegram webhook triggers remain dry-run formatted.

---

## 5. Provenance Signatures

- **Tested Code Commit (`CODE_SHA`):** `c23a9ed245267019b89bd8946674695c1c938a44`
- **Evidence Document:** `docs/evidence/sprint_11/ASUS_DEPLOYMENT_HARNESS_FINAL.md`
- **Campaign State Record:** `.agents/campaign/STATE.json`
