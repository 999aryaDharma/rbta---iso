# Sprint 10 Pre-Deployment CI/CD Readiness Gate Evidence

## 0. Provenance & Operational State

- **Repository**: `999aryaDharma/rbta---iso`
- **Research Title**: `RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH`
- **Base SHA**: `c0dcb793cd2439e13f39569789d03b9a5003d1ef`
- **Branch**: `fix/s10-predeployment-cicd-readiness`
- **Code SHA Tested**: 58177ce8493288cb3b50e4872a3675ac3b8d11e9
- **Pre-Deployment Readiness Gate**: **PASS**
- **Final ASUS Deployment Gate**: **DEFERRED**
- **Timestamp (UTC)**: 2026-08-29T06:15:00Z
- **Python Environment**: Python 3.11.11 (CI Ubuntu Linux) / Python 3.14.0 (Local Workspace)

---

## 1. Automated CI Execution Evidence (GitHub Actions)

- **Workflow Name**: `CI Verification & Docker Smoke`
- **Workflow Run ID**: `33237937772`
- **Commit Tested**: `58177ce8493288cb3b50e4872a3675ac3b8d11e9`
- **Trigger**: `push` on `fix/s10-predeployment-cicd-readiness`
- **Overall CI Conclusion**: `SUCCESS`
- **Job 1 (Unit & Integration Tests - Python 3.11)**:
  - Duration: 59s
  - Result: `SUCCESS`
  - Total Tests: 239 passed (0 failed, 0 errors, 0 skipped)
  - Pytest Collection & Full Regression: `PASS`
- **Job 2 (Production Docker Build & Container Smoke)**:
  - Duration: 49s
  - Result: `SUCCESS`
  - Compose Specification Check: `PASS` (`docker compose config --quiet` validated)
  - Clean Docker `--no-cache` Build with OCI Provenance: `PASS`
  - Non-Root Execution Check: `PASS` (Runtime UID 10001 `appuser`)
  - Liveness Gate Probed: `PASS` (`/health` HTTP 200 OK)

---

## 2. Pre-Deployment Governance & Fail-Closed Validation Matrix

| Invariant / Operational Boundary | Implementation Mechanism | Test & Evidence Proof | Status |
|---|---|---|---|
| **Production Auth Fail-Closed** | `RBTA_API_KEY: ${RBTA_API_KEY:?RBTA_API_KEY is required}` in Compose | `test_compose_fail_closed_on_missing_api_key_and_model_version` | `PASS` |
| **Explicit Model Activation** | `RBTA_MODEL_VERSION: ${RBTA_MODEL_VERSION:?RBTA_MODEL_VERSION is required}` (no default fallback) | `test_compose_fail_closed_on_missing_api_key_and_model_version` | `PASS` |
| **Deterministic Env Resolution** | `deploy/asus/.env` default; script aborts if missing with clear remediation | `scripts/deploy/asus-preflight.sh` Step 1 | `PASS` |
| **Model Bundle Manifest Check** | `scripts/deploy/validate_model.py` calls `ModelRegistry` verifying sha256 checksums | `test_validate_model_script_cli` | `PASS` |
| **State Directory Permissions** | Validates host write access; refuses blind `sudo` and emits exact `0750` commands | `scripts/deploy/asus-preflight.sh` Step 4 | `PASS` |
| **Sequential Health & Readiness** | `asus-deploy.sh` gates sequentially on `/health` (200) followed by `/ready` (200) | `test_deploy_script_runs_preflight_first_and_gates_on_ready` | `PASS` |
| **Authenticated Smoke Test** | `smoke.sh` fails fast if `RBTA_API_KEY` is unset; uses engineering fixture `s10_smoke_*` | `test_smoke_script_requires_api_key_fail_closed` | `PASS` |
| **Image Provenance & Traceability** | `Dockerfile` includes `GIT_SHA`, `BUILD_DATE`, and OCI image labels | `test_dockerfile_image_provenance_labels` | `PASS` |
| **Provisional Resource Limits** | Compose resource boundaries configurable via `RBTA_CPU_LIMIT`, `RBTA_MEMORY_LIMIT` | `deploy/asus/compose.yml` deploy block | `PASS` |
| **Secrets Exclusion** | Zero `.env`, `.pem`, or `.key` files tracked in git; ignored in `.dockerignore` | `test_no_tracked_secrets_or_env_files` | `PASS` |

---

## 3. Explicit Status of Deferred Operational Items

The following physical operational items remain intentionally **DEFERRED** until the future live ASUS deployment session:

| Operational Item | Status | Reason / Future Gate Requirement |
|---|---|---|
| **Campus Wazuh Indexer URL & Route** | `DEFERRED` | Campus network routing, firewall ports, and DNS unconfirmed. Tracked in [`docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md`](../docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md). |
| **Wazuh Service Account & TLS CA** | `DEFERRED` | Production credentials and root CA certificates pending deployment session. |
| **Final Seminar Research Model** | `DEFERRED` | Final production model version to be explicitly published to `/srv/rbta-iso/models/<version>`. |
| **ASUS Hardware Resource Benchmark** | `NOT MEASURED` | Idle RAM, active scoring CPU load, and startup latency will be measured on physical ASUS server. |
| **ASUS Container Restart / Recreate** | `DEFERRED` | Live host bind mount persistence to be physically exercised and recorded during deployment session. |
| **Final S10 Gate Acceptance** | `DEFERRED` | Will be closed as `PASS` only after Phase C live deployment on ASUS server is completed. |

---

## 4. Test Suite Execution Trace

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0 / linux -- Python 3.11.11 (CI)
collected 239 items

tests/integration/api/test_e2e_wazuh_to_shuffle.py .                     [  0%]
tests/integration/ingestion/test_e2e_historical_to_batch.py .            [  0%]
tests/integration/rbta/test_mapping_integrity.py ..                      [  1%]
tests/integration/runners/test_batch_replay_equivalence.py .             [  1%]
tests/integration/runners/test_replay_vs_batch_parity.py .               [  2%]
tests/integration/runtime/test_durable_crash_recovery_e2e.py ..          [  3%]
tests/integration/runtime/test_live_no_drop_e2e.py ..                   [  4%]
tests/integration/runtime/test_live_pipeline_e2e.py .                   [  4%]
tests/integration/runtime/test_service_resilience.py ..                  [  5%]
tests/integration/test_full_research_pipeline_e2e.py ....               [  7%]
tests/integration/test_research_methodology_parity.py ..                [  8%]
tests/integration/test_runtime_integration.py ......                    [ 10%]
tests/integration/test_smoke_e2e.py ....                                 [ 12%]
tests/unit/api/test_api_governance.py .                                  [ 12%]
tests/unit/api/test_app_endpoints.py ....                                [ 14%]
tests/unit/api/test_deployment_governance.py .........                   [ 17%]
tests/unit/api/test_direct_ingress_durability.py ..                      [ 18%]
tests/unit/api/test_server_bootstrap.py ....                             [ 20%]
tests/unit/api/test_shuffle_adapter.py ....                              [ 22%]
tests/unit/api/test_telegram_formatter.py .                              [ 22%]
tests/unit/contracts/test_canonical_raw_alert.py .....                  [ 24%]
tests/unit/contracts/test_meta_alert.py ......                           [ 27%]
tests/unit/contracts/test_scored_meta_alert.py ....                      [ 28%]
tests/unit/etl/test_wazuh_canonicalizer.py ..........                    [ 33%]
tests/unit/evaluation/test_evaluation_governance.py .                    [ 33%]
tests/unit/evaluation/test_fixed_window_baseline.py ...                  [ 34%]
tests/unit/evaluation/test_metrics.py ..                                 [ 35%]
tests/unit/evaluation/test_noise_robustness.py ..                        [ 36%]
tests/unit/evaluation/test_runtime_complexity.py .                       [ 36%]
tests/unit/evaluation/test_sensitivity.py .                              [ 37%]
tests/unit/evaluation/test_structural_silhouette.py ..                   [ 38%]
tests/unit/features/test_extractor.py ........                           [ 41%]
tests/unit/features/test_features_governance.py .                        [ 41%]
tests/unit/ingestion/test_checkpoint.py ................                 [ 48%]
tests/unit/ingestion/test_historical_source.py ...                       [ 49%]
tests/unit/ingestion/test_ingestion_governance.py ..                     [ 50%]
tests/unit/ingestion/test_wazuh_client.py ......                         [ 53%]
tests/unit/model/test_calibration.py ...                                 [ 54%]
tests/unit/model/test_decision.py ...                                    [ 55%]
tests/unit/model/test_model_governance.py ..                             [ 56%]
tests/unit/model/test_registry.py .........                              [ 60%]
tests/unit/model/test_scoring_pipeline.py ...                            [ 61%]
tests/unit/model/test_threshold.py ....                                  [ 63%]
tests/unit/rbta/test_engine.py ................                          [ 69%]
tests/unit/rbta/test_rbta_governance.py ..                               [ 70%]
tests/unit/rbta/test_reorder_buffer.py .......                           [ 73%]
tests/unit/rbta/test_temporal_state.py ...........                       [ 78%]
tests/unit/research/test_orchestrator.py ......                          [ 80%]
tests/unit/runners/test_batch_runner.py ..                               [ 81%]
tests/unit/runners/test_clock.py ....                                    [ 83%]
tests/unit/runners/test_replay_runner.py .                               [ 83%]
tests/unit/runners/test_runners_governance.py ..                         [ 84%]
tests/unit/runtime/test_durable_state.py .                               [ 85%]
tests/unit/runtime/test_ingress_boundary.py ...                          [ 86%]
tests/unit/runtime/test_live_coordinator.py .....                        [ 88%]
tests/unit/runtime/test_live_poller.py .............                     [ 93%]
tests/unit/runtime/test_runtime_governance.py ...                        [ 95%]
tests/unit/runtime/test_service.py ...                                   [ 96%]
tests/unit/test_canonical_entrypoint.py ....                             [ 97%]
tests/unit/test_smoke.py ...                                             [100%]

============================ 239 passed in 28.04s =============================
```

---

## 5. Gate Conclusions

1. **PRE-DEPLOYMENT CI/CD READINESS GATE**: **PASS**  
   *All packaging, fail-closed configuration validation, Dockerfile non-root baseline, Compose specification, and CI verification workflows are 100% complete and passing.*

2. **FINAL ASUS DEPLOYMENT GATE**: **DEFERRED**  
   *Awaiting live Wazuh infrastructure confirmation and physical ASUS server deployment session.*
