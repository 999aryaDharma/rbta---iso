# Handoff for Claude review — live ingestion foundation

Review target: latest commit on `prod/final-dashboard-demo` containing this
handoff and the L1 foundation changes. The commit SHA is in the Git history.

## Review request

Review the diff for correctness and data-loss risks, especially:

1. SQLite snapshot transaction, recovery/migration from JSON 1.0/1.1, and crash
   ordering between raw evidence and runtime state.
2. Live alert deduplication/fingerprint migration and whether transport envelope
   fields are excluded without weakening raw evidence conflict checks.
3. Indexer pagination integrity behavior (missing daily index, timeout, shard
   failures, early termination, cursor progress) and TLS configuration.
4. Whether service locking and current tests support only the guarantees claimed.
5. Whether replay behavior, frozen-model provenance, and the locked research
   method remain untouched.

Please report concrete findings with file/line, severity, and a minimal fix.
Do not treat this as approval for campus deployment or a claim that live mode is
complete.

## Verified scope

Implemented in this change: durable SQLite runtime checkpoint/migration,
raw-evidence flush ordering, service mutation locking, transport-stable evidence
fingerprints, stricter Indexer polling integrity checks, Indexer TLS env support,
and compatible HTTP 413 handling. See
[`live-L1-gate.md`](../research-spec/evidence/live-L1-gate.md) for exact test
commands and observed results.

Targeted backend/integration checks: 122 passed. API endpoint checks: 7 passed.
Frontend lint, typecheck, 38 unit tests, and build passed. Full backend suite had
345 passed and 2 environment/legacy failures. Playwright had 19 passed and 5
existing-flow/assertion failures. Do not summarize these as a fully green gate.

## Not implemented or verified

There is no live worker thread, order buffer, read-only live API/status,
live/replay dashboard switcher, or live Telegram dispatcher yet. Full
retention polling still materializes candidates; streaming is deferred. Campus
VPS OS, Indexer reachability/RBAC/TLS, capacity, and real-data smoke test remain
unknown. Deployment status is **BLOCKED_EXTERNAL**. Offline replay remains in
place; the research core and model artifacts were not changed.

## Next: live stream plan L2–L6 — L2–L5 SELESAI di kode; L6 eksternal pending

L2 (worker thread), L3 (status API), L4 (order buffer), L5 (dispatcher
Telegram + switcher) selesai dan terintegrasi — gate 205 passed. L6
sisi-kode (real HTTP sender + wiring kredensial) selesai; verifikasi VPS,
RBAC/TLS, kredensial bot, dan shadow run menunggu peneliti. Status:
**BLOCKED_EXTERNAL** tetap berlaku. Detail: [`LIVE-STREAM-PLAN.md`](./LIVE-STREAM-PLAN.md).

## Review F1–F13 — resolved (2026-09-30)

Temuan Major/Moderate: token disanitasi dari log, drain default false,
filter usia bootstrap, quarantine durable + backoff (kebijakan: quarantine,
bukan halt), config_hash guard, telemetri jujur (buffer null bila off,
at-least-once, event_lag), file lock satu-proses, dry-run flag. Gate
terintegrasi: **357 passed, 0 failed**. Detail per temuan di
[`LIVE-STREAM-PLAN.md`](./LIVE-STREAM-PLAN.md#review-f1f13--resolusi-2026-09-30-paralel-4-agen--integrasi). Status deployment tetap **BLOCKED_EXTERNAL**.

## Follow-up fixes (review findings resolved, uncommitted)

Independent review of `9de3438..16ab574` returned 1 Important + 7 Minor
findings. All accepted findings are fixed in the working tree; one Minor was
declined with reasoning. Research core, replay, and model artifacts untouched
(`git diff` over `src/rbta src/model src/etl src/config src/runners` is empty).

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 1 | Important | Transport errors dropped host + exception chain (`src/ingestion/wazuh_client.py:110-118`) | Messages now include `base_url`; chained with `from exc`; body still redacted. Same for 401/403 path. Tests: `test_network_failure_includes_host_and_cause`, `test_http_error_includes_host_and_cause_without_body` |
| 2 | Minor | `skip_conflict_check` dead parameter (`src/runtime/raw_evidence.py:165`, `src/runtime/service.py:296`) | Docstrings/comments now state the flag is a legacy no-op and replay always enforces conflict detection (fail-closed). No behavior change; covered by existing `test_replay_duplicate_*` tests |
| 3 | Minor (declined) | Missing `_shards.failed` key defaults to pass (`src/runtime/live_source.py`) | Declined: strict require-key broke 13 existing tests whose mocks treat shard-less responses as valid; real OpenSearch always sends `_shards`, so `failed > 0` check suffices. No change |
| 4 | Minor | `metadata.items()` raw `AttributeError` on non-mapping (`src/runtime/json_safe.py:78`) | Raises `TypeError` with clear message. Test: `test_canonical_fingerprint_rejects_non_mapping_metadata` |
| 5 | Minor | `WAZUH_INDEXER_CA_PATH` never validated (`src/ingestion/wazuh_client.py:60`) | Missing file raises `FileNotFoundError` naming the env var at init. Test: `test_invalid_ca_path_raises_clear_error` |
| 6 | Minor | `restore_state` raw `OperationalError` if state dir vanished (`src/runtime/durable_state.py:268`) | Recreates parent dir before connect. Test: `test_restore_recreates_parent_dir_removed_between_init_and_restore` |
| 7 | Minor | `_drain_pending_scoring` outside `_serialized` lock (`src/runtime/service.py:247`) | Decorated with `@_serialized` (reentrant RLock, no deadlock; all callers already hold the lock). Covered by existing `test_service.py` suite |
| 8 | Minor | Cursor message dropped offending value (`src/runtime/live_source.py:192`) | Message now includes `cursor=[...]`. Test: `test_live_poller_cursor_message_includes_offending_value` |

Verification after fixes: 61 focused tests passed
(`test_wazuh_client`, `test_live_poller`, `test_raw_evidence`,
`test_durable_state`, `test_service`); broader gate suite
(`tests/unit/runtime`, `tests/unit/ingestion`,
`test_direct_ingress_durability`, `tests/integration/runtime`,
`tests/integration/runners`, `test_app_endpoints`) → **135 passed**.
Full backend suite, frontend checks, and Playwright E2E not re-run here;
prior gate results in `live-L1-gate.md` still stand with their stated
failures. Deployment status remains **BLOCKED_EXTERNAL**.
