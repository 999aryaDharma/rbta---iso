# Hybrid Lenovo–ASUS Runtime Design

Date: 2026-09-07  
Branch: `prod/final-dashboard-demo`

## Objective

Run the same RBTA + Isolation Forest demonstration on either a Windows Lenovo laptop with Docker Desktop or the existing Linux ASUS server. The host paths differ, but the container paths, model contract, API behavior, replay semantics, and evaluation definitions remain identical.

The Lenovo corpus is a directory of daily Wazuh exports named like `wazuh-alerts-4.x-2026.04.02.jsonl`, accompanied by `.meta` sidecars. Only `.jsonl` and `.jsonl.gz` files are alert inputs. `.meta`, manifests, archives, and partial files are not parsed as alerts.

## Runtime Topology

Both targets expose the service only on loopback and mount three stable container paths:

| Purpose | Container path | Lenovo host | ASUS host |
|---|---|---|---|
| Replay corpus | `/app/data/replay` (read-only) | `D:/KAMPUS/SKRIPSI/wazuh-data-2026/indexer-export` | `/srv/rbta-iso/replay` |
| Model registry | `/app/artifacts/models` (read-only) | Configurable local directory | `/srv/rbta-iso/models` |
| Runtime state | `/app/data/runtime` (read-write) | Docker named volume | `/srv/rbta-iso/state` |

The local named volume avoids UID and Windows bind-mount permission failures while preserving state across container restarts. The source dataset and model remain immutable mounts on both targets.

## Local Operator Workflow

A standard-library Python launcher provides `up`, `down`, `status`, `logs`, and `index` commands. It:

1. Reads `deploy/local/.env` without shell evaluation.
2. Normalizes native Windows drive paths for Docker Compose; WSL users supply `/mnt/d/...`.
3. Validates the replay directory, supported files, model version, API key, port, Docker, and Compose.
4. Starts the hardened container.
5. Waits for readiness.
6. Starts a background dataset catalog refresh and displays progress.
7. Prints the Demo URL.

The launcher never copies or modifies the replay corpus.

## Dataset Catalog

The current synchronous catalog reads and hashes every event during the initial dataset-list request. That is unsuitable for a directory containing many daily files. The replacement has two paths:

- Fast listing: stat files and reuse valid cached manifests. Uncached or changed files are returned as `pending` without reading their contents.
- Background refresh: inspect, canonicalize, count, timestamp-bound, and SHA-256 each supported file in a worker thread. Status reports total, completed, current filename, failures, and timestamps.

Starting a single dataset may inspect just that dataset synchronously. Starting `ALL` requires a complete, current catalog so the UI cannot freeze while implicitly scanning the whole corpus. The Demo page exposes indexing state and a deliberate refresh action.

Cache writes are atomic and serialized because the replay worker and catalog worker can overlap. Dataset files stay read-only.

## Research Corpus Input

The canonical research orchestrator accepts either one `.jsonl`/`.jsonl.gz` file or a directory. Directory discovery is non-recursive, lexically deterministic, and limited to supported alert inputs. Each alert is canonicalized with provenance-aware error messages.

The full April–August directory is one chronological corpus, not one training set. After global timestamp ordering and duplicate-ID rejection, the existing leakage-safe split remains:

- 60% reference training
- 20% threshold calibration
- 20% held-out evaluation

The run manifest records every source file, its size, SHA-256, and a combined corpus digest. This makes Lenovo and ASUS runs comparable even when host paths differ.

## API and UI

New authenticated replay endpoints:

- `GET /api/v1/replay/datasets/catalog-status`
- `POST /api/v1/replay/datasets/refresh`

The dataset list adds `inspection_status` (`cached` or `pending`). The Demo page explains why indexing is required, shows progress, and disables full-corpus replay until the catalog is current. Existing single-file replay remains available.

## Safety and Failure Behavior

- Path traversal protections remain unchanged.
- Replay and model mounts are read-only.
- The API remains loopback-only behind the shared demo key.
- Empty or unsupported directories fail before deployment.
- Invalid JSON and canonicalization errors identify file and line.
- Duplicate Wazuh IDs across daily files fail the research run to prevent temporal leakage.
- A failed catalog file is reported without killing the API; `ALL` remains blocked until resolved.
- `main` is not modified; all changes and the final push target only `prod/final-dashboard-demo`.

## Verification

Verification covers unit tests for Windows/WSL path handling, deterministic discovery, gzip input, sidecar exclusion, corpus provenance, non-blocking catalog behavior, refresh status, and runtime validation. It also covers API contracts, frontend lint/typecheck/unit tests/build, backend full suite/coverage, Compose rendering where Docker is available, and a final diff/ref audit before push.
