# Live/replay state recovery contract

## Runtime checkpoint schema 1.2

`finalized_history.sqlite3` now holds the authoritative `runtime_snapshot` row in
the same transaction as newly seen alert IDs and finalized history. Its payload
contains active buckets, per-agent temporal state, meta counter, pending scoring,
outbox and source checkpoint. No research algorithm or model is changed.

`state.json` remains a compatibility/inspection mirror. If mirror publication
fails after the SQLite commit, recovery uses SQLite and a warning is emitted.
Do not treat the mirror alone as a backup or read it as current worker status.

Existing schema 1.0/1.1 JSON is loaded when no SQLite snapshot exists. The first
successful checkpoint writes schema 1.2 without resetting buckets or IDs. A 1.2
mirror without its SQLite snapshot is rejected. Migration cannot reconstruct
alerts lost by a crash in the older implementation; do not claim otherwise.

## Evidence and fingerprints

Auto-persist ingress flushes raw evidence before mutating the core. Buffered replay
keeps its existing batching, but every durable checkpoint/shutdown flushes evidence
before committing runtime state. Evidence is allowed to be ahead of the core after
a failure; retry can safely process it. Core commits must not precede evidence.

New evidence rows use `fingerprint_version=2`. The hash excludes only named
transport fields: `source_index`, `source_document_id`, `source_sort`, `fetched_at`,
`source_mode`, `timestamp_received`, `opensearch_index`, `opensearch_document_id`.
Transport metadata is still stored. All other canonical content remains protected.

Existing rows are version 1. On duplicate ingestion their old hash is verified
against stored content before normalized comparison. Their historical hash and
content are not overwritten. Corrupt/unsupported legacy evidence fails closed.

## Upgrade, backup and rollback

Stop all processes that write the affected runtime before upgrade or backup.
Back up the complete runtime directory and use SQLite's backup facility or a
consistent stopped-process filesystem backup, including any WAL files. Keep the
model artifact version and raw evidence alongside the runtime backup.

Roll back application code together with its matching pre-upgrade runtime backup;
do not run old code against a 1.2 SQLite snapshot and then resume new code. New code
prefers SQLite, while old code writes only JSON. This mixed-version operation can
make the two representations diverge.

Each live run must have its own state directory/database, separate from replay
workspaces. The in-process RLock serializes service mutations; it does not provide
interprocess exclusion or impose event-time ordering on concurrent submissions.
Interprocess ownership remains a worker-stage requirement.

## Indexer transport

TLS verification defaults to enabled. `WAZUH_INDEXER_CA_PATH` supplies the campus
CA bundle. `WAZUH_INDEXER_VERIFY_TLS` accepts `true` or `false`; keep `true` for
deployment. Explicit constructor settings override environment configuration.

Search requests tolerate missing daily indices and request no partial results.
Timeouts, early termination, failed shards and non-advancing full-page cursors
raise integrity errors. An HTTP 401/403 or malformed response is not an empty poll.
Indexer error response bodies are omitted from authentication/HTTP exceptions.

See the [OpenSearch Search API](https://docs.opensearch.org/latest/api-reference/search-apis/search/)
for search parameter semantics. Actual behavior against the campus version still
requires an authorized smoke test.
