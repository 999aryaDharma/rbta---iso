# Sprint 10 Final Pre-Deployment CI/CD Readiness Gate Evidence

## 0. Provenance & Operational State

- **Repository**: `999aryaDharma/rbta---iso`
- **Research Title**: `RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH`
- **Base SHA**: `34b6ecdd07fa2d97051ddd88b9bd7c8c0d4a3e7d`
- **Branch**: `fix/s10-predeployment-final-readiness`
- **Code SHA Tested**: 3c8ac078955d7f00c798ef5d0a33cb7e65a0edae
- **Pre-Deployment Final Readiness Gate**: **PASS**
- **Final ASUS Deployment Gate**: **DEFERRED**
- **Timestamp (UTC)**: 2026-08-29T06:33:00Z
- **Python Environment**: Python 3.11.16 (CI Ubuntu Linux) / Python 3.14.0 (Local Workspace)

---

## 1. Automated CI Execution Evidence (GitHub Actions)

- **Workflow Name**: `CI Verification & Docker Smoke`
- **Workflow Run ID**: `33238647321`
- **Commit Tested**: `3c8ac078955d7f00c798ef5d0a33cb7e65a0edae`
- **Trigger**: `push` on `fix/s10-predeployment-final-readiness`
- **Overall CI Conclusion**: `SUCCESS`
- **Job 1 (Unit & Integration Tests - Python 3.11)**:
  - Duration: 56s
  - Result: `SUCCESS`
  - Total Tests: 247 passed (0 failed, 0 errors, 0 skipped on Linux CI)
  - Pytest Collection & Full Regression: `PASS`
- **Job 2 (Production Docker Build & Container Smoke)**:
  - Duration: 41s
  - Result: `SUCCESS`
  - Compose Specification Check: `PASS` (`docker compose config --quiet` validated)
  - Clean Docker `--no-cache` Build with OCI Provenance: `PASS`
  - Non-Root Execution Check: `PASS` (Runtime UID 10001 `appuser`)
  - Liveness Gate Probed: `PASS` (`/health` HTTP 200 OK)

---

## 2. Pre-Deployment Governance & Behavioral Validation Matrix

| Invariant / Operational Boundary | Implementation Mechanism | Test & Evidence Proof | Status |
|---|---|---|---|
| **One Canonical Directory Layout** | Root-relative `state/` and `models/` under `/srv/rbta-iso/` | `deploy/asus/compose.yml` bind mounts `../../state` and `../../models` | `PASS` |
| **State UID/GID 10001 Validation** | `scripts/deploy/validate_state_dir.py` validates POSIX `10001:10001` & owner write | `test_state_dir_validation_behavioral_wrong_uid_gid`, `test_state_dir_validation_behavioral_mock_uid_gid` | `PASS` |
| **Rejection of Insecure chmod 777** | Preflight refuses world-writable directories and emits `0750` remediation | `test_state_dir_validation_behavioral_rejects_world_writable` | `PASS` |
| **Behavioral API Key Fail-Closed** | `docker compose config` fails with nonzero exit when `RBTA_API_KEY` is missing | `test_compose_behavioral_fail_closed_missing_api_key` | `PASS` |
| **Behavioral Model Version Fail-Closed** | `docker compose config` fails with nonzero exit when `RBTA_MODEL_VERSION` is missing | `test_compose_behavioral_fail_closed_missing_model_version` | `PASS` |
| **Behavioral Compose Success** | `docker compose config` succeeds with exit 0 when mandatory variables are present | `test_compose_behavioral_success_with_required_vars` | `PASS` |
| **Behavioral Smoke Auth Fail-Closed** | `smoke.sh` exits nonzero before network call if `RBTA_API_KEY` is unset | `test_smoke_script_behavioral_auth_fail_closed` | `PASS` |
| **Behavioral Missing Env Check** | `asus-preflight.sh` exits nonzero when `RBTA_ENV_FILE` does not exist | `test_preflight_behavioral_missing_env` | `PASS` |
| **Model Bundle Manifest Check** | `scripts/deploy/validate_model.py` calls `ModelRegistry` verifying sha256 checksums | `test_validate_model_script_cli` | `PASS` |
| **Wazuh Checklist Consistency** | Scheduling fields in `WAZUH-LIVE-INTEGRATION-CHECKLIST.md` match canonical code defaults | Fast poll 5s, Recent recon 5m / 2 daily indices, Full recon 1h | `PASS` |
| **Image Provenance & Traceability** | `Dockerfile` includes `GIT_SHA`, `BUILD_DATE`, and OCI image labels | `test_dockerfile_image_provenance_labels` | `PASS` |
| **Non-Root Runtime** | Container execution restricted to UID `10001:10001` with `cap_drop: ALL` | Verified in CI Docker Smoke job | `PASS` |
| **Secrets Exclusion** | Zero `.env`, `.pem`, or `.key` files tracked in git; ignored in `.dockerignore` | `test_no_tracked_secrets_or_env_files` | `PASS` |

---

## 3. Explicit Status of Deferred Physical & Infrastructure Items

The following physical operational items remain intentionally **DEFERRED** until the future live ASUS deployment session:

| Operational Item | Status | Reason / Future Gate Requirement |
|---|---|---|
| **Campus Wazuh Indexer URL & Route** | `DEFERRED` | Campus network routing, firewall ports, and DNS unconfirmed. Tracked in [`docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md`](../docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md). |
| **Wazuh Service Account & TLS CA** | `DEFERRED` | Production credentials and root CA certificates pending deployment session. |
| **Final Seminar Research Model** | `DEFERRED` | Final production model version to be explicitly published to `/srv/rbta-iso/models/<version>`. |
| **ASUS Hardware Resource Benchmark** | `NOT MEASURED` | Idle RAM, active scoring CPU load, and startup latency will be measured on physical ASUS server. |
| **ASUS Container Restart / Recreate** | `DEFERRED` | Live host bind mount persistence to be physically exercised and recorded during deployment session. |
| **Physical ASUS Host Permissions** | `NOT YET PHYSICALLY VERIFIED` | Preflight logic is verified in CI; host directory permissions on ASUS will be verified upon setup. |
| **Final S10 Gate Acceptance** | `DEFERRED` | Will be closed as `PASS` only after Phase C live deployment on ASUS server is completed. |

---

## 4. Gate Conclusions

1. **PRE-DEPLOYMENT FINAL READINESS GATE**: **PASS**  
   *Canonical single-layout directory structure, UID/GID 10001 state ownership validator, behavioral fail-closed tests, Dockerfile provenance, and CI verification workflows are 100% complete and passing.*

2. **FINAL ASUS DEPLOYMENT GATE**: **DEFERRED**  
   *Awaiting live Wazuh infrastructure discovery and physical ASUS server deployment session.*
