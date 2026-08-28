# RBTA Production Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing RBTA + Isolation Forest codebase into a research-valid, production-ready dual-mode system supporting batch research, historical Wazuh replay, and live Wazuh ingestion without duplicating Research Core logic.

**Architecture:** All sources normalize into one `CanonicalRawAlert`, then flow through one stateful `RBTAEngine`, one seven-feature extractor, and one model runtime. Batch mode may train/calibrate artifacts explicitly; replay/live mode may only load immutable artifacts. Historical Wazuh uses daily PIT + `search_after`; live Wazuh uses campus collector push or approved private connectivity with overlap + dedup.

**Tech Stack:** Python 3.11+, pandas/numpy for research analysis only, scikit-learn, joblib, pytest, HTTP client with explicit timeouts, PostgreSQL for durable operational state, FastAPI-compatible REST boundary if API layer is implemented, Docker for ASUS deployment.

**Spec:** `docs/research-spec/00-SOURCE-OF-TRUTH.md` through `docs/research-spec/12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`

## Global Constraints

- Seminar report and `00-SOURCE-OF-TRUTH.md` override legacy code/comments.
- EMA/ETW state is local per `agent_id`, never global.
- Warm-up is 100 events per agent; alpha is 0.10.
- `delta_t = base_delta_t * EMA_gap / baseline_gap`, clipped to 0.5x–1.5x base.
- No watermark `late_drop` in primary research path.
- Exactly one RBTA bucket type: `(agent_id, rule_group_primary)`.
- Exactly seven IF features in the order defined by `03-FEATURE-ENGINEERING-SPEC.md`.
- `IsolationForest(n_estimators=200, contamination="auto")`.
- RobustScaler required.
- Tukey threshold is `Q3 + 1.5*IQR` with no clamp to 1.0.
- Ground truth/synthetic attack labels must not influence model parameters.
- Historical/replay/live all use Wazuh event timestamp for EMA/RBTA.
- Replay/live must not fit model/scaler/calibration/threshold.
- One transport request per page, not one request per alert.
- No hardcoded workstation path, credential, model result, threshold, ARR, throughput, or success claim.
- Each sprint is gated. Do not start sprint N+1 until sprint N evidence is recorded and all tests for N pass.

---

# Sprint 0 — Repository Baseline and Test Harness

## Objective

Create a reproducible Python project and protect current behavior before algorithm refactor.

### Task 0.1 — Establish project manifest

**Files:**
- Create: `pyproject.toml`
- Modify: `.gitignore`

- [ ] Declare supported Python version and runtime/test dependencies.
- [ ] Configure pytest in `pyproject.toml`.
- [ ] Ensure `.env`, secrets, model staging files, local DB, logs, caches are ignored.
- [ ] Run `python -m pytest --collect-only` and record output.
- [ ] Commit: `build: add reproducible python project config`

### Task 0.2 — Create test layout

**Files:**
- Create: `tests/unit/`
- Create: `tests/integration/`
- Create: `tests/fixtures/wazuh/`

- [ ] Add a smoke test proving imports work.
- [ ] Add representative saved raw Wazuh hit fixtures matching real API shapes already observed.
- [ ] Do not store credentials or private tokens in fixtures.
- [ ] Run `pytest -q`.
- [ ] Commit: `test: establish rbta test harness`

### Task 0.3 — Preserve baseline evidence

- [ ] Record current branch/commit.
- [ ] Record current primary entry points and legacy outputs.
- [ ] Save only non-sensitive sample output needed for before/after comparison.
- [ ] Do not modify research algorithm in Sprint 0.

**Gate S0:** project installs, tests collect/run, no secrets committed.

---

# Sprint 1 — Canonical Contracts and Domain Configuration

## Objective

Make file, API, archive, and live sources produce one canonical contract.

### Task 1.1 — Centralize domain constants

**Files:**
- Create: `src/config/domain.py`
- Test: `tests/unit/config/test_domain.py`

Produces single definitions for:

```text
AGENT_CRITICALITY
GROUP_SEVERITY_WEIGHT
CRITICAL_MITRE_TACTICS
```

- [ ] Write tests for known mappings and default policy.
- [ ] Move definitions from legacy modules without changing semantics authorized by source-of-truth.
- [ ] Replace imports gradually; do not leave duplicate authoritative maps.
- [ ] Commit: `refactor: centralize research domain constants`

### Task 1.2 — Define canonical DTOs

**Files:**
- Create: `src/contracts/raw_alert.py`
- Create: `src/contracts/meta_alert.py`
- Create: `src/contracts/scored_meta_alert.py`
- Test: `tests/unit/contracts/`

Required `CanonicalRawAlert` fields:

```text
wazuh_alert_id
timestamp
agent_id
agent_name
rule_group_primary
rule_level
rule_id
mitre_tactics
srcip
agent_criticality
metadata
```

- [ ] Enforce timezone-aware parse at source boundary, normalized consistently.
- [ ] Validate severity 0–15.
- [ ] Preserve source metadata separately from research fields.
- [ ] Commit: `feat: define canonical alert contracts`

### Task 1.3 — Build canonical Wazuh parser

**Files:**
- Create: `src/etl/wazuh_canonicalizer.py`
- Test: `tests/unit/etl/test_wazuh_canonicalizer.py`

Must support:

```text
raw Wazuh object
OpenSearch hit {_index,_id,_source,sort}
MITRE nested rule.mitre.*
MITRE flattened rule.mitre_tactics / rule.mitre_techniques
```

- [ ] Write failing test for each shape first.
- [ ] Ensure `wazuh_alert_id` comes from Wazuh alert `id`, not OpenSearch `_id`.
- [ ] Preserve `_id` in metadata.
- [ ] Test file/API parity: same logical alert -> equal CanonicalRawAlert research fields.
- [ ] Commit: `feat: canonicalize wazuh alert sources`

**Gate S1:** one canonical DTO, no duplicated authoritative domain mapping, parser parity passes.

---

# Sprint 2 — Agent-Local EMA and RBTA Core

## Objective

Replace legacy global adaptive window, late-drop, and Bucket B with correct stateful Research Core.

### Task 2.1 — Implement per-agent temporal state

**Files:**
- Create: `src/rbta/temporal_state.py`
- Test: `tests/unit/rbta/test_temporal_state.py`

Required state:

```text
last_timestamp
warmup_event_count
warmup_gaps
baseline_gap
ema_gap
current_delta_t
```

- [ ] Test first event.
- [ ] Test warm-up ends at event 100 per agent.
- [ ] Test alpha=0.10.
- [ ] Test proportional formula.
- [ ] Test lower/upper clip.
- [ ] Test Agent A events cannot mutate Agent B state.
- [ ] Commit: `feat: implement agent local elastic time window`

### Task 2.2 — Isolate reorder buffer

**Files:**
- Create: `src/rbta/reorder_buffer.py`
- Test: `tests/unit/rbta/test_reorder_buffer.py`

- [ ] Preserve valid events.
- [ ] No `late_drop` result exists.
- [ ] Drain returns all buffered events in deterministic order.
- [ ] Commit: `refactor: isolate lossless reorder buffer`

### Task 2.3 — Implement one RBTA engine

**Files:**
- Create: `src/rbta/engine.py`
- Test: `tests/unit/rbta/test_engine.py`

Interface:

```python
process(alert: CanonicalRawAlert) -> list[MetaAlert]
drain() -> list[MetaAlert]
flush_idle(event_time) -> list[MetaAlert]
```

- [ ] Bucket key exactly `(agent_id, rule_group_primary)`.
- [ ] Merge uses local agent `current_delta_t`.
- [ ] Max bucket duration 60 minutes.
- [ ] `start_time=min`, `end_time=max` for out-of-order safety.
- [ ] Aggregate rule distribution, severity distribution, MITRE tactic set, source alert IDs.
- [ ] Remove CompoundMetaAlert/Bucket B from primary dependency graph.
- [ ] Commit: `feat: implement single bucket rbta engine`

### Task 2.4 — Mandatory integrity tests

- [ ] `sum(alert_count) == valid processed raw count` after drain.
- [ ] No source alert ID appears in two final meta-alerts.
- [ ] Same input deterministic across repeated runs.
- [ ] Out-of-order fixture loses zero events.
- [ ] Commit: `test: prove rbta mapping integrity`

**Gate S2:** EMA isolation, no-event-loss, single-bucket and mapping integrity all pass.

---

# Sprint 3 — Exact Seven-Feature Extractor

## Objective

Replace all legacy feature vectors with exactly seven features from the source-of-truth.

### Task 3.1 — Create extractor

**Files:**
- Create: `src/features/extractor.py`
- Test: `tests/unit/features/test_extractor.py`

Single constant:

```python
FEATURE_COLUMNS = [
  "max_severity",
  "mitre_tactic_count",
  "critical_mitre_tactic_present",
  "alert_count_log",
  "rule_diversity_shannon",
  "severity_dispersion",
  "agent_criticality",
]
```

- [ ] Singleton entropy=0.
- [ ] Same-rule repeated entropy=0.
- [ ] Balanced two-rule entropy approximately 1 normalized.
- [ ] Singleton severity dispersion=0.
- [ ] Duplicate MITRE tactics count once.
- [ ] Critical tactic flag correct.
- [ ] `alert_count_log == log1p(count)`.
- [ ] Output exactly seven columns in fixed order.
- [ ] Missing required aggregate raises clear error, no silent zero-fill.
- [ ] Commit: `feat: implement canonical seven feature extractor`

### Task 3.2 — Remove duplicate feature logic

- [ ] Legacy IF and RBTA code no longer calculates feature vector independently.
- [ ] Search repo for old active feature lists and remove/archive primary imports.
- [ ] Commit: `refactor: remove legacy feature duplication`

**Gate S3:** all seven feature tests pass and only one active `FEATURE_COLUMNS` exists.

---

# Sprint 4 — Isolation Forest Training and Immutable Artifacts

## Objective

Make training explicit and inference reproducible for single-event streaming.

### Task 4.1 — Model core

**Files:**
- Create: `src/models/isolation_forest.py`
- Test: `tests/unit/models/test_isolation_forest.py`

- [ ] RobustScaler.
- [ ] IF exactly 200 trees.
- [ ] contamination exactly `"auto"`.
- [ ] fixed random state.
- [ ] no ground-truth code path.
- [ ] Commit: `feat: implement research isolation forest model`

### Task 4.2 — Score calibration

**Files:**
- Create: `src/models/score_calibration.py`
- Test: `tests/unit/models/test_score_calibration.py`

- [ ] Fit calibration only from training/reference score set.
- [ ] Single score transform uses stored parameters.
- [ ] Degenerate reference fails artifact generation.
- [ ] No per-request min/max.
- [ ] Commit: `feat: add persistent anomaly score calibration`

### Task 4.3 — Tukey and decision

- [ ] Compute `Q3 + 1.5*IQR` without clamp.
- [ ] Implement 4-quadrant Decision Matrix.
- [ ] FP gate uses `mitre_tactic_count == 0`.
- [ ] Commit: `feat: implement threshold and decision engine`

### Task 4.4 — Artifact registry

**Files:**
- Create: `src/models/registry.py`
- Test: `tests/unit/models/test_registry.py`

Write/load exactly:

```text
isolation_forest.joblib
robust_scaler.joblib
score_calibration.json
threshold.json
feature_schema.json
metadata.json
```

- [ ] Atomic staging->publish.
- [ ] Feature schema compatibility validation.
- [ ] Roundtrip same vector -> same score.
- [ ] Missing artifact -> readiness failure.
- [ ] Commit: `feat: add versioned model artifact registry`

**Gate S4:** artifact roundtrip passes; single-event inference does not collapse to 0.5; zero live fit operations.

---

# Sprint 5 — Historical Wazuh Indexer Source

## Objective

Implement efficient, resumable historical acquisition from Wazuh Indexer.

### Task 5.1 — HTTP client boundary

**Files:**
- Create: `src/sources/base.py`
- Create: `src/sources/wazuh/client.py`
- Test: `tests/unit/sources/wazuh/test_client.py`

- [ ] Explicit connect/read timeout.
- [ ] TLS verification configurable but production defaults secure.
- [ ] Credentials from env/config only.
- [ ] Redacted logging.
- [ ] Retry only transient status/network failures.
- [ ] 401/403 fail-fast.
- [ ] Commit: `feat: add wazuh indexer client`

### Task 5.2 — Daily index discovery

- [ ] Discover real indices; missing dates valid.
- [ ] Sort daily indices ascending.
- [ ] Do not construct nonexistent dates as guaranteed source.
- [ ] Commit: `feat: discover historical wazuh indices`

### Task 5.3 — PIT + search_after iterator

**Files:**
- Create: `src/sources/wazuh/historical.py`
- Test: `tests/unit/sources/wazuh/test_historical.py`

- [ ] PIT per daily index only.
- [ ] Reject partial PIT.
- [ ] Sort exactly `@timestamp ASC, id ASC`.
- [ ] Page size default 500.
- [ ] Yield individual events from pages.
- [ ] `search_after` uses last hit exact `sort`.
- [ ] PIT closes in `finally` success/error.
- [ ] Commit: `feat: add resumable historical wazuh source`

### Task 5.4 — Historical checkpoint

- [ ] Persist index + last_sort + count + last Wazuh ID.
- [ ] Resume with new PIT after restart.
- [ ] Dedup prevents overlap double-processing.
- [ ] Commit: `feat: checkpoint historical wazuh export`

**Gate S5:** multi-page fixture exports once, resumes without duplicate, closes PIT, skips missing date safely.

---

# Sprint 6 — Batch and Replay Runners

## Objective

Prove both execution modes use the same core.

### Task 6.1 — Batch runner

**Files:**
- Create: `src/runners/batch.py`
- Test: `tests/integration/test_batch_runner.py`

- [ ] Uses CanonicalRawAlert and RBTAEngine.
- [ ] Explicit drain at EOF.
- [ ] Training is explicit command, not implicit side-effect.
- [ ] Commit: `feat: add research batch runner`

### Task 6.2 — Replay stream runner

**Files:**
- Create: `src/runners/stream.py`
- Create: `src/runners/replay_clock.py`
- Test: `tests/integration/test_replay_runner.py`

- [ ] Supports 1x/10x/100x/MAX.
- [ ] Sleep uses event-time gap / speed.
- [ ] RBTA always sees original timestamps.
- [ ] Loads model artifacts only.
- [ ] Commit: `feat: add historical replay stream runner`

### Task 6.3 — Equivalence proof

- [ ] Same canonical fixture through batch and MAX replay.
- [ ] Assert equal final meta-alerts.
- [ ] Assert equal seven features.
- [ ] Assert equal scores/decisions with same artifact.
- [ ] Commit: `test: prove batch replay core equivalence`

**Gate S6:** batch-replay equivalence passes.

---

# Sprint 7 — Live Ingestion and Durable Runtime State

## Objective

Support real new Wazuh alerts when campus agents reconnect.

### Task 7.1 — State store interface

**Files:**
- Create: `src/state/interface.py`
- Create: `src/state/postgres.py`
- Test: `tests/integration/state/`

Persist:

```text
dedup IDs
source checkpoint
per-agent temporal state
active buckets
finalized meta-alerts
outbox
```

- [ ] Recovery does not reset state.
- [ ] Same Wazuh ID cannot mutate core twice.
- [ ] Commit: `feat: add durable stream state store`

### Task 7.2 — Live Indexer polling source

**Files:**
- Create: `src/sources/wazuh/live.py`
- Test: `tests/unit/sources/wazuh/test_live.py`

- [ ] No long-lived PIT for live tail.
- [ ] Poll recent indices with configurable overlap.
- [ ] Default start: 5s interval, 5m overlap, 500 page.
- [ ] Dedup before core mutation.
- [ ] Handle date rollover.
- [ ] Late-indexed event inside overlap is processed.
- [ ] Commit: `feat: add live wazuh index polling source`

### Task 7.3 — Campus collector ingress contract

**Files:**
- Create: `src/operational/api.py` or equivalent service boundary.
- Test: `tests/integration/test_ingress_idempotency.py`

- [ ] Single-alert authenticated ingress.
- [ ] Optional batch ingress for collector recovery.
- [ ] 202 accepted for new event.
- [ ] Duplicate returns idempotent success without core mutation.
- [ ] Invalid schema 400.
- [ ] Commit: `feat: add idempotent wazuh collector ingress`

### Task 7.4 — Idle flush + restart recovery

- [ ] Idle bucket finalization tested.
- [ ] Restart reloads same active EMA/buckets.
- [ ] Controlled shutdown drains according to documented policy.
- [ ] Commit: `feat: recover stateful rbta stream runtime`

**Gate S7:** overlap/dedup, restart recovery, idle flush, and no duplicate mutation pass.

---

# Sprint 8 — Evaluation Refactor

## Objective

Implement exactly the seminar evaluation, independent of operational transport.

- [ ] Sensitivity values: 1,5,10,15,20,30,45,60 with adaptive OFF.
- [ ] Final RBTA adaptive per-agent ON.
- [ ] Fixed tumbling baseline per report.
- [ ] ARR from actual run.
- [ ] Noise robustness 0,5,10,20,30%, preserving valid agent pair.
- [ ] Runtime across 8 subset sizes; actual throughput and R².
- [ ] IF structural validity: Silhouette on scaled seven-feature space.
- [ ] 100 same-proportion random partitions.
- [ ] Report null mean/std/min/max, percentile, z, empirical p.
- [ ] Remove synthetic attack/ground-truth primary evaluation imports.
- [ ] No hardcoded claims.

**Gate S8:** evaluation artifacts are reproducible and every reported number is generated by a run.

---

# Sprint 9 — Operational API, Outbox, Shuffle

## Objective

Expose scored results safely without putting research logic in Shuffle.

- [ ] `/health` liveness.
- [ ] `/ready` checks state store + model/scaler/calibration/threshold/schema.
- [ ] `/runtime/stats` operational counters.
- [ ] Meta-alert detail/trace endpoints.
- [ ] Durable outbox for scored events.
- [ ] Stable `event_id`/`meta_id` downstream idempotency.
- [ ] Shuffle adapter uses REST/OpenAPI contract.
- [ ] Shuffle does not implement EMA/RBTA/features/IF.
- [ ] Telegram formatting stays downstream.

**Gate S9:** Wazuh-like fixture -> RBTA -> IF -> outbox -> Shuffle stub exactly once.

---

# Sprint 10 — ASUS Deployment and CI

## Objective

Run safely as a lightweight research analytics service on ASUS without hosting Wazuh itself.

- [ ] Create production Dockerfile with non-root user.
- [ ] Create compose/deployment manifest for app + PostgreSQL if selected.
- [ ] Persistent volumes only for state/artifacts/log policy.
- [ ] Resource limits documented and benchmarked.
- [ ] Secrets external to image/repo.
- [ ] Healthcheck uses `/health`; deployment readiness uses `/ready`.
- [ ] Graceful shutdown tested.
- [ ] CI runs lint/type checks if adopted + full pytest relevant suites.
- [ ] Artifact build/deploy does not include research raw secrets/data unintentionally.

**Gate S10:** clean ASUS deploy from repository + documented env succeeds and survives restart.

---

# Sprint 11 — Dashboard and Demonstration Layer

## Objective

Visualize the algorithm without changing it.

Recommended dashboard data:

```text
raw alert rate
meta-alert rate
alert reduction rate
active agents
active buckets
per-agent EMA/baseline/current delta-t
latest scored meta-alerts
replay progress + historical event time + speed
model version
source mode
```

- [ ] Dashboard reads operational state/API only.
- [ ] No decision/model calculation in frontend.
- [ ] Replay pause/resume does not change event timestamp.
- [ ] Reset starts explicit new run.

**Gate S11:** demo can replay archive deterministically and show real downstream Shuffle/Telegram execution.

---

# Final Verification

Agent must run and attach evidence for:

```text
pytest full suite
EMA isolation tests
no-event-loss tests
seven-feature schema tests
artifact roundtrip tests
historical pagination/resume tests
live overlap/dedup tests
batch-replay equivalence tests
restart recovery tests
research end-to-end tests
operational Wazuh-like -> outbox test
```

Repository search must show no active primary imports/usages of:

```text
CompoundMetaAlert
late_drop
compute_dynamic_contamination
old 11-feature FEATURE_COLS
synthetic scenario A/B/C in primary runner
ground_truth parameterization of IF
```

## Commit Discipline

Each task should end in a small reviewable commit. Do not produce one giant refactor commit.

Recommended prefixes:

```text
build:
test:
refactor:
feat:
fix:
docs:
```

## Agent Stop Rule

If source-of-truth and implementation plan appear inconsistent, **STOP** and report the exact contradiction with file/section references. Do not invent a compromise.

If infrastructure credential/network access is missing, implement against fixtures/stubs and mark only that external integration evidence as blocked. Do not weaken security or expose Wazuh Indexer publicly to make a test pass.