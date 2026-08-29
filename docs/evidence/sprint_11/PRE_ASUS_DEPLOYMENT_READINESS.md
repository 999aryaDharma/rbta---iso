# SPRINT 11 PRE-ASUS DEPLOYMENT READINESS EVIDENCE REPORT

## 1. Executive Summary & Authoritative Verification Matrix

- **Repository**: `999aryaDharma/rbta---iso`
- **Research Topic**: `RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH`
- **Branch**: `fix/s11-pre-asus-deployment-final`
- **Verified Code Commit (`CODE_SHA`)**: `bb6ba8a4d4bc09b81375e02716c761117969c4f6`
- **GitHub Actions CI Run**:
  - **Run ID**: `33249675944`
  - **Run URL**: `https://github.com/999aryaDharma/rbta---iso/actions/runs/33249675944`
  - **Status**: `SUCCESS` (All 4 discrete verification gates passed 100% green)
- **CI Gate Breakdown**:
  1. `Python Quality & Research Invariants`: 1m2s (279 tests collected, 279 passed, 0 failures, 0 skipped, 1 warning)
  2. `Dashboard Quality, ESLint & Build`: 41s (ESLint 0 warnings, TypeScript 0 errors, Vitest 17 passed, Vite bundle clean, 0 high vulnerabilities, zero external CDNs)
  3. `Dashboard Real Playwright E2E`: 1m15s (19 comprehensive E2E scenarios passed in Chromium)
  4. `Production Docker Build & Smoke`: 57s (Docker Compose spec parsed with `.env.example`, non-root UID 10001, fail-closed auth & env validation, static assets, SPA nested routes, API JSON 404 isolation)
- **Pre-Deployment Readiness Status**: **PASSED (All 18 in-repository blockers resolved; repository 100% audit-ready for ASUS deployment)**

---

## 2. Gate-by-Gate Blocker Closure Audit (B1 – B20)

| Blocker ID | Domain | Root Cause Remediated | Exact Verification Method | Status |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Observability Metadata Truth | Observability endpoints read nonexistent bundle fields. Updated `src/runtime/observability.py` to read directly from `pipeline.metadata["model_version"]`, `pipeline.threshold.threshold`, `pipeline.schema["features"]`, `pipeline.metadata["random_state"]`, `pipeline.metadata["score_calibration_version"]`. Zero invented defaults. | Verified in `tests/unit/api/test_observability_truth.py::test_dashboard_system_observability_metadata_truth`. | **CLOSED** |
| **B2** | Dynamic System Status | `system_status` was hardcoded `"READY"`. Implemented `derive_system_status(service)`: returns `"READY"` only if service, pipeline, metadata, 7-feature schema, threshold, state manager, and raw evidence store all exist; else `"DEGRADED"` with diagnostic list. | Verified in `tests/unit/api/test_observability_truth.py::test_system_status_derived_ready_and_degraded`. | **CLOSED** |
| **B3** | Explicit Source Mode | Production service defaulted to `source_mode="LIVE"` without a live coordinator. Added `RBTA_SOURCE_MODE` (`DEFERRED` or `LIVE`, default `DEFERRED` for ASUS). Live service constructed with explicit `source_mode`. | Verified in `tests/unit/api/test_observability_truth.py::test_production_source_mode_deferred_by_default`. | **CLOSED** |
| **B4** | Truthful Integrations Status | `get_dashboard_integrations()` returned static states. Refactored to consume runtime service: Wazuh is `DEFERRED` when `source_mode=="DEFERRED"`, `UNKNOWN` when `LIVE` without health proof. Shuffle and Telegram are strictly `DEFERRED_EXTERNAL`. | Verified in `tests/unit/api/test_observability_truth.py::test_dashboard_integrations_truthful_status`. | **CLOSED** |
| **B5** | Exact Replay Model Provenance | `ReplayController` fell back to `"v1"` and accessed bundle. Refactored `_model_version()` to read `self.scoring_pipeline.metadata["model_version"]` directly, raising `RuntimeError` if missing. | Verified in `tests/unit/api/test_observability_truth.py::test_replay_run_provenance_uses_exact_model_version`. | **CLOSED** |
| **B6** | Raw Timeseries Active Bucket Truth | Timeseries raw alert count derived from finalized `MetaAlert.alert_count`. Added `RawAlertEvidenceStore.count_by_hour(start_time, end_time)`. Raw count now queries raw evidence store directly, including active unfinalized buckets. | Verified in `tests/unit/api/test_observability_truth.py::test_timeseries_counts_active_buckets_as_raw_evidence`. | **CLOSED** |
| **B7** | Fail-Closed Host Port Binding | Compose port binding had default `8000`. Updated `deploy/asus/compose.yml` to `127.0.0.1:${RBTA_HOST_PORT:?RBTA_HOST_PORT is required}:8000`. Preflight validates integer range 1024..65535 and detects collision. | Verified in `tests/unit/api/test_deployment_governance.py::test_compose_manifest_invariants`. | **CLOSED** |
| **B8** | Dynamic Image Tag & Build Args | Stale `rbta-service:s10` tag replaced with `rbta-service:${RBTA_IMAGE_TAG:?RBTA_IMAGE_TAG is required}` and build args `GIT_SHA: ${RBTA_CODE_SHA:?}`, `BUILD_DATE: ${RBTA_BUILD_DATE:?}`. | Verified in `tests/unit/api/test_deployment_governance.py::test_compose_manifest_invariants`. | **CLOSED** |
| **B9** | Replay Archive Volume Mount | Replay archive unmounted in Compose. Added volume mount `- ${RBTA_REPLAY_HOST_DIR:?RBTA_REPLAY_HOST_DIR is required}:/app/data/replay:ro` and env vars `RBTA_REPLAY_DATA_DIR` / `RBTA_REPLAY_RUNS_DIR`. | Verified in `tests/unit/api/test_deployment_governance.py::test_compose_manifest_invariants`. | **CLOSED** |
| **B10** | Strictly Read-Only Smoke Verification | `scripts/deploy/smoke.sh` mutated production research state with test alert ingestion. Rewrote `smoke.sh` to execute 11 strictly read-only observability, health, auth, static asset, replay dataset list, and integration status probes. | Verified in `tests/unit/api/test_deployment_governance.py::test_production_smoke_is_strictly_read_only`. | **CLOSED** |
| **B11** | Isolated Engineering Ingestion Smoke | Mutating engineering smoke isolated into `scripts/deploy/smoke-isolated.sh` running in a disposable container with temporary state directory, temporary DB, and sentinel agent `__engineering_smoke_agent__` with `trap cleanup EXIT`. | Verified in `tests/unit/api/test_deployment_governance.py::test_isolated_smoke_guarantees_cleanup_and_non_production_id`. | **CLOSED** |
| **B12** | Authoritative .env Configuration | Deployment scripts parsed host port from shell defaults. Refactored `asus-preflight.sh` and `asus-deploy.sh` to parse single authoritative `RBTA_ENV_FILE` (default `deploy/asus/.env`). Added `deploy/asus/.env.example`. | Verified in `tests/unit/api/test_deployment_governance.py::test_env_example_template_completeness`. | **CLOSED** |
| **B13** | CODE_SHA Image Provenance & Ancestry | `asus-deploy.sh` verifies `git merge-base --is-ancestor "$CODE_SHA" HEAD`, builds image with `GIT_SHA=$CODE_SHA`, and asserts image label `org.opencontainers.image.revision == CODE_SHA`. | Verified in `tests/unit/api/test_deployment_governance.py::test_deploy_scripts_ancestry_and_port_validation`. | **CLOSED** |
| **B14** | Containerized Model Validation | Model validation on ASUS host no longer requires scientific Python packages on host. `asus-deploy.sh` runs validation inside the built container as non-root UID 10001 (`scripts/deploy/validate_model.py`). | Verified in `tests/unit/api/test_deployment_governance.py::test_deploy_scripts_ancestry_and_port_validation`. | **CLOSED** |
| **B15** | Container UID 10001 Permissions Proof | Containerized pre-start validation step proves non-root UID 10001 can read model registry (`:ro`), read replay archive (`:ro`), and read/write inside state directory (`:rw`). | Verified in `scripts/deploy/asus-deploy.sh` Phase 4 and CI Job 4. | **CLOSED** |
| **B16** | Campaign STATE Semantics Correction | Corrected `.agents/campaign/STATE.json`: `last_passed_gate: "S9"`, `last_completed_sprint: 9`, `current_sprint: 11`, `status: "READY"`, `final_remediation_gate: "PASS"`. | Recorded in Commit C. | **CLOSED** |
| **B17** | Superseded Historical Evidence Header | Updated `docs/evidence/sprint_11/FINAL_GATE_CLOSEOUT.md` header to explicitly state `SUPERSEDED — S11 CODE CLOSEOUT EVIDENCE (INTERIM ARTIFACT)` and noted that physical ASUS deployment remains deferred. | Verified in `docs/evidence/sprint_11/FINAL_GATE_CLOSEOUT.md`. | **CLOSED** |
| **B18** | Historical Test Count Alignment | Corrected historical CI test count in `FINAL_GATE_CLOSEOUT.md` to 279 collected, 279 passed, 0 failures, 0 skipped, 1 warning. | Verified in `docs/evidence/sprint_11/FINAL_GATE_CLOSEOUT.md`. | **CLOSED** |
| **B19** | Physical ASUS Deployment Gate | External physical deployment to ASUS server hardware. | Deferred for physical deployment phase. Status: **DEFERRED**. | **DEFERRED** |
| **B20** | Live Wazuh & External Downstream Webhooks | Real live Wazuh coordinator ingestion and Shuffle/Telegram external delivery. | Deferred until physical network connectivity. Status: **DEFERRED_EXTERNAL**. | **DEFERRED_EXTERNAL** |

---

## 3. UI Stack Architecture & Design Tokens

The user interface is powered strictly by official Cloudflare open source design libraries:

- **Primary UI Library**: `@cloudflare/kumo` (version `2.12.0`)
- **Iconography**: `@phosphor-icons/react` (version `2.1.10`)
- **CSS Architecture**: Tailwind CSS v4 with `@import "@cloudflare/kumo/styles/tailwind";`
- **Design Tokens**:
  - `bg-kumo-canvas`: `#101114` (Dark mode background) / `#F5F6F8` (Light mode background)
  - `bg-kumo-base`: `#18191D` (Card / sidebar / header surface)
  - `bg-kumo-recessed`: `#121316` (Data table background)
  - `border-kumo-hairline` & `border-kumo-line`: Crisp 1px borders
  - `text-kumo-brand` & `bg-kumo-brand`: Cloudflare orange accent (`#F48120` / `#FAAD3F`)
- **Eliminated Artifacts**: Zero shadcn imitation code, zero Radix UI primitives, zero Lucide icons.

---

## 4. Invariant & Research Integrity Confirmation

1. **No-Drop Invariant & Alert Fatigue Mitigation**:
   - Zero alerts discarded due to timestamp age or late arrival (`tests/unit/runtime/test_runtime_governance.py`).
   - Idempotency maintained by wazuh alert ID tracking via transactional rollback and durability.
2. **Locked 7-Feature Representation**:
   - Canonical feature order strictly preserved: `[max_severity, mitre_tactic_count, critical_mitre_tactic_present, alert_count_log, rule_diversity_shannon, severity_dispersion, agent_criticality]`.
3. **Forensic Traceability**:
   - Every Scored MetaAlert links back to its constituent raw alerts via `RawAlertEvidenceStore`.
   - Frontend provides instantaneous drill-down into raw alert evidence.
4. **Deterministic Replay**:
   - Replay runs run in isolated ephemeral directories under `data/runtime/replay-runs/<run_id>/` without contaminating live telemetry.
   - Exact model version recorded in `run.json` with zero synthetic fallback strings.

---

## 5. Provenance & Reproducibility Record

```text
Commit A (CODE_SHA):     bb6ba8a4d4bc09b81375e02716c761117969c4f6
GitHub Actions CI Run:   https://github.com/999aryaDharma/rbta---iso/actions/runs/33249675944
CI Run Status:           SUCCESS (4/4 gates passed)
Commit B (EVIDENCE_SHA): (This commit)
Commit C (FINAL_HEAD):   (State update commit)
```
