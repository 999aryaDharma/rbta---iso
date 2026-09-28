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

There is no live worker, interprocess writer lock, read-only live API/status,
live/replay dashboard switcher, or live outbox dispatcher in this change. Full
retention polling still materializes candidates; streaming is deferred. Campus
VPS OS, Indexer reachability/RBAC/TLS, capacity, and real-data smoke test remain
unknown. Deployment status is **BLOCKED_EXTERNAL**. Offline replay remains in
place; the research core and model artifacts were not changed.
