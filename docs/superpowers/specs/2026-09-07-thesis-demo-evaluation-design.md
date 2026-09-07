# Thesis Demo and Evaluation Design

**Status:** Approved continuation scope  
**Target branch:** `prod/final-dashboard-demo` only  
**Primary audience:** thesis examiners, supervisors, SOC practitioners, and the researcher operating the demo

## Goal

Turn the existing deterministic replay page into a defensible thesis demonstration that exposes live RBTA and frozen-model Isolation Forest behavior, runs the complete research evaluation protocol after replay, and explains every result without claiming attack-detection accuracy.

## Non-negotiable research boundaries

- An alert is a Wazuh rule output, not proof of an attack.
- ARR measures reduction of triage units, not detection accuracy.
- Isolation Forest produces a prioritization score, not a ground-truth class.
- Silhouette and permutation results are internal structural evidence only.
- `contamination="auto"` is model configuration metadata; it does not control escalation in this pipeline.
- External Wazuh, Shuffle, and Telegram delivery remain explicitly deferred until separately verified.
- The official research dataset, the golden demo dataset, code SHA, parser version, model version, and random seed must be independently identifiable.

## Architecture

### 1. Dataset provenance and fast discovery

`ReplayDatasetCatalog` owns dataset inspection and a sidecar manifest cache. It accepts `.jsonl` and `.jsonl.gz`, computes SHA-256 over the exact source bytes, counts valid non-empty records, records size and timestamp range, and invalidates cached metadata when size or modification time changes. Listing datasets uses cached manifests; replay start reuses the same metadata instead of scanning the file a second time.

Each replay run copies the selected manifest fields into `run.json`. The UI labels a dataset as `golden`, `research`, or `unclassified` only when this is explicitly supplied by manifest metadata; it never guesses from the filename.

### 2. Live evaluation snapshot

`ReplayEvaluationTracker` receives each accepted raw alert and each finalized scored meta-alert. It maintains bounded/O(1) counters and online statistics:

- processed and evidence-preserved alerts;
- finalized and active meta-alerts;
- live ARR;
- throughput and elapsed time;
- action and decision distributions;
- above-threshold count and rate;
- anomaly-score count, min, max, mean, and fixed-bin histogram;
- source-member accounting and traceability coverage;
- model/reference provenance and explicit interpretation boundary.

The live tracker never refits the scaler, model, calibration, or threshold. It exposes a versioned DTO through replay status so frontend calculations remain limited to presentation.

### 3. Post-replay evaluation job

An `EvaluationJobController` can start only after a replay is `COMPLETED` or `STOPPED` with enough valid events. It reads canonical evidence from that isolated run, executes phases in a worker thread, publishes progress after each phase, and writes an atomic `evaluation.json` artifact.

Phases:

1. eight static delta-t sensitivity points: 1, 5, 10, 15, 20, 30, 45, 60 minutes;
2. three aggregation variants: global time-only fixed, contextual fixed, contextual adaptive RBTA;
3. ARR and context-purity comparison for all variants;
4. robustness at 0%, 5%, 10%, 20%, and 30% injected low-severity/no-MITRE noise using identical seeded scenarios across variants;
5. runtime evaluation over eight increasing subsets with warm-up, five measured repetitions, median and IQR; sorting/preparation and engine time reported separately;
6. frozen-model score distribution and decision distribution;
7. observed binary ESCALATE/non-ESCALATE Silhouette against 100 equal-proportion seeded permutations;
8. artifact/provenance summary and permitted interpretation text.

Cancellation is cooperative between phases. A failed phase produces `ERROR` with the phase and message while preserving completed partial results. Starting a second job while one is active returns a conflict.

### 4. Fair aggregation comparison

Context purity is defined as the proportion of meta-alerts containing exactly one `(agent_id, rule_group_primary)` pair. Context contamination is its complement. The contextual fixed baseline uses the same contextual key as RBTA but a static window; therefore:

- time-only fixed vs contextual fixed isolates the value of contextual grouping;
- contextual fixed vs contextual adaptive isolates the value of temporal adaptation.

Noise evaluation reuses the same injected alert stream per noise rate for all three variants.

### 5. Runtime and persistence correctness

- Raw evidence insertion count changes only when SQLite actually inserts a unique row.
- Evidence fingerprints cover the complete canonical evidence payload in both replay and live modes; replay may omit stored original envelopes but must not weaken canonical integrity.
- Outbox stores only actionable `ESCALATE` items. All scored items remain queryable from indexed history.
- Finalized history is paged/queryable from SQLite and only a bounded recent cache remains in memory.
- Deduplication keys move to an indexed SQLite ledger for replay-sized workloads; JSON checkpoints retain only active aggregation state, counters, pending scoring, and transport offsets.
- Container and API hardening include security headers, request-size bounds, rate limiting for protected control endpoints, constant-time API-key comparison, and read-only root filesystem where runtime write mounts permit it.

### 6. Demo information architecture

The sidebar label becomes **Demo** and the page title becomes **Thesis Demonstration**. The route may remain `/replay` for compatibility, with `/demo` as the canonical route and `/replay` redirecting.

The page uses five sections in presentation order:

1. **Research question and claim boundary** — the problem, the exact safe claim, and dataset/model provenance.
2. **Run controls** — dataset, speed, progress, lifecycle, and a preflight readiness checklist.
3. **Live pipeline** — raw → canonical → RBTA → seven features → frozen IF → Tukey/decision → deferred output.
4. **Live evidence** — ARR, throughput, decisions, threshold behavior, traceability, score distribution, and current meta-alert.
5. **Complete evaluation** — phase progress, aggregation ablation, sensitivity, noise, runtime, structural IF result, limitations, and export artifact link.

Every metric includes a one-sentence definition and a “what this does not prove” note where misuse is likely. English technical identifiers remain intact, while explanatory copy is Indonesian for the thesis audience.

### 7. Frontend performance and dependency safety

Routes are lazy-loaded so chart/table-heavy pages are separate chunks. React Router is upgraded to the patched v7 line and route/navigation APIs are migrated without changing URLs or behavior. Automated tests cover schema parsing, routing, the Demo label, live metrics, job states, and interpretation copy.

## API contracts

- `GET /api/v1/replay/datasets` returns cached provenance manifests.
- `GET /api/v1/replay/status` includes `telemetry.evaluation_live`.
- `POST /api/v1/replay/evaluation/start` starts the post-replay job for the active `run_id`.
- `GET /api/v1/replay/evaluation/status` returns phase progress and results.
- `POST /api/v1/replay/evaluation/cancel` requests cooperative cancellation.
- `GET /api/v1/replay/evaluation/artifact` downloads the completed JSON artifact for the active run.

Existing authenticated replay endpoints remain compatible.

## Verification

- Unit tests for manifest cache invalidation, gzip reading, duplicate evidence count, full canonical fingerprinting, actionable-only outbox, online live metrics, context purity, contextual baseline, matched noise scenarios, and repeated runtime statistics.
- API tests for auth, invalid lifecycle transitions, progress/status DTOs, artifact availability, rate limiting, and security headers.
- Frontend tests for Zod contracts, Demo navigation, live metric explanations, evaluation states, and claim-boundary copy.
- Full Python suite, coverage, frontend lint/typecheck/unit tests/build, Playwright flow, Docker build/smoke, `git diff --check`, and dependency audit.

## Thesis output

The repository will contain a Chapter IV–V framework mapping every table/figure to a generated artifact. Numerical result slots stay explicitly marked as “populate from frozen official run”; no fabricated values are inserted.
