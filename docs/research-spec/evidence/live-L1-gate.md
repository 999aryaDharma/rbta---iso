# Live prerequisites gate — 2026-09-28

Status: **PARTIAL — full regression gate not green; live deployment not verified**.

Branch: `prod/final-dashboard-demo`.
Base code SHA: `9de3438`. Review the resulting commit on `prod/final-dashboard-demo`.
Deployment target: campus VPS selected by researcher; OS and source access pending.

## Implemented and reproduced

| Root cause | Reproduction and correction |
| --- | --- |
| Dedup/history committed before JSON bucket snapshot | Two new failure-injection tests failed before the fix. Snapshot, new IDs, history, pending scoring, outbox and cursor now commit in one SQLite transaction. Additional tests cover transaction rollback and legacy JSON migration. |
| Evidence still buffered after durable ingress/shutdown | Two new tests failed before the fix. Evidence is flushed before durable core state and before auto-persist ingress mutation. |
| Poller treats timeout/failed shards/early termination as valid results; no cursor progress guard | Five new tests failed before the fix. Requests tolerate missing daily indices but prohibit partial search; incomplete results and non-advancing cursors fail closed. |
| TLS CA env ignored and server error body exposed | Two new tests failed before the fix. CA/verify env is read, invalid boolean rejected, response bodies omitted from exceptions. |
| Envelope metadata changes duplicate fingerprint | Two new tests failed before the fix. Version 2 excludes named transport fields; version 1 records are verified and compared without rewriting their hashes. |
| Service mutation overlaps between threads | Controlled overlap test failed before the fix. Reentrant lock now covers service mutations and persistence. No claim of interprocess exclusion or event-time ordering. |
| Oversized API request raises AttributeError | Existing test failed: installed Starlette lacks `HTTP_413_CONTENT_TOO_LARGE`. Use compatible `HTTP_413_REQUEST_ENTITY_TOO_LARGE`; all seven endpoint tests pass. |
| Three frontend assertions use obsolete labels | Vitest failed against unchanged current components. Updated expected labels to existing `Action: SUPPRESS` and Indonesian replay text; product UI unchanged. |

## Changed files

Source:

- `src/runtime/durable_state.py`
- `src/runtime/service.py`
- `src/runtime/raw_evidence.py`
- `src/runtime/json_safe.py`
- `src/runtime/live_source.py`
- `src/ingestion/wazuh_client.py`
- `src/api/app.py`

Tests:

- `tests/unit/runtime/test_durable_state.py`
- `tests/unit/runtime/test_service.py`
- `tests/unit/runtime/test_raw_evidence.py`
- `tests/unit/runtime/test_live_poller.py`
- `tests/unit/ingestion/test_wazuh_client.py`
- `tests/unit/api/test_direct_ingress_durability.py`
- `dashboard/src/components/shared/DecisionBadge.test.tsx`
- `dashboard/src/features/replay/CurrentMetaAlertCard.test.tsx`

Documentation:

- `docs/deployment/LIVE-TOPOLOGY-DECISION.md`
- `docs/deployment/LIVE-STATE-RECOVERY.md`
- `docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md`
- This evidence record.
- Local ignored `AGENTS.md`: campus VPS target recorded, native Lenovo development retained.

## Commands and observed results

Commands run from repository root in PowerShell unless noted:

```powershell
& 'C:/Users/User/miniconda3/envs/ML/python.exe' -m pytest tests/unit/runtime tests/unit/ingestion tests/unit/api/test_direct_ingress_durability.py tests/integration/runtime tests/integration/runners -q -p no:cacheprovider
```

**122 passed**, one `python_multipart` pending-deprecation warning, 34.52s.
Includes batch/replay equivalence and live reconciliation/restart coverage.

```powershell
& 'C:/Users/User/miniconda3/envs/ML/python.exe' -m pytest tests/unit/api/test_app_endpoints.py -q -p no:cacheprovider
```

**7 passed**, one pending-deprecation warning, 10.11s.

```powershell
& 'C:/Users/User/miniconda3/envs/ML/python.exe' -m pytest -q -p no:cacheprovider --tb=short
```

Final full backend run: **345 passed, 2 failed**, one pending-deprecation warning,
76.76s. Failures:

1. `tests/unit/deploy/test_local_launcher.py::test_preserves_wsl_mount_path`:
   legacy Docker helper uses Windows `Path.resolve()` even when passed
   `platform="linux"`, yielding `D:\mnt\...` instead of `/mnt/...`.
   Historical Docker source/test left unchanged.
2. `tests/unit/test_smoke.py::test_third_party_dependencies_import`:
   installed Matplotlib extension compiled against NumPy 1.x fails with the
   environment's NumPy 2.5.3. No packages, environment, or model artifacts changed.

These source/test files were verified unchanged from HEAD. The earlier full run
also failed the HTTP 413 case; that case was fixed and passes in the final run.

Frontend, from `dashboard`:

```powershell
npm run lint
npm run typecheck
npm test -- --maxWorkers=2
npm run build
```

Lint and typecheck passed. The initial sandboxed test invocation failed during
Vite startup with `spawn EPERM`; retried outside the sandbox. The first unsandboxed
run produced 35 passed / 3 failed due to the obsolete assertions listed above.
Final run: **38 passed in 13 files**, 32.26s. Production build passed in 6.02s
(with TypeScript verification in the build command). No product UI source changed.

```powershell
npm run test:e2e -- --workers=2
```

Chromium was available. **19 passed, 5 failed**, 58.0s. Failures in
`dashboard/e2e/dashboard.spec.ts`:

- Test 2, line 540: expected `Security Analytics Overview` is absent after login.
- Test 3, line 549: expected `Escalated Incidents` text is absent.
- Test 13, line 672: expected `RUNNING` is absent after starting replay.
- Test 20, line 767: all-datasets start button remains disabled; click timed out.
- Test 23, line 819: `max_severity` is absent in the feature inspector.

These are assertion/flow failures, not a missing browser. E2E spec and product
UI source remain unchanged from HEAD. The suite uses mocked backend responses;
it does not verify the new live transport against a real server. Further replay
fixture/flow diagnosis is required before declaring the overall gate green.

`git diff --check` passed with the repository's normal line-ending configuration.
Git warned about LF/CRLF conversion. A diagnostic run overriding `core.autocrlf`
reported CR characters as whitespace; no repository Git settings were changed.
Research core, model, runners and ReplayController diffs were empty. Git status
also warned about inaccessible old pytest cache directories.

## Remaining work / claim boundaries

- Worker, interprocess single writer, persistent live run/model pin, read-only API,
  `/api/v1/live/status`, context switcher and live notification dispatcher are not
  implemented in this prerequisite change.
- Full-retention reconciliation remains enabled; streaming/memory optimization is
  deferred pending ordered-consumption and partial-cycle recovery design/tests.
- No change to research core, replay controller/runner, model artifacts, thresholds,
  synthetic labels, or evaluation results.
- CPU/memory capacity and real TLS/RBAC/index mappings on VPS are unverified.
- No Wazuh network request, remote deployment, Docker launch, notification send,
  or real-data shadow run was performed.
- No throughput or power-failure durability claim is inferred from unit tests.
- Campus integration remains **BLOCKED_EXTERNAL**. Gate L1 as originally proposed
  is not complete, and the system is not claimed live-ready.
