# Thesis Demo and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible thesis-demo workflow with correct persistence, live metrics, complete post-replay RBTA/Isolation Forest evaluation, examiner-friendly UI, and Chapter IV–V scaffolding.

**Architecture:** The existing replay controller remains the lifecycle owner. Focused catalog, tracking, and evaluation-job modules are injected into it; evaluation results are backend-authoritative and persisted per run. The SPA consumes versioned DTOs and presents a guided `/demo` route with lazy-loaded pages.

**Tech Stack:** Python 3.11+, FastAPI, SQLite, NumPy/Pandas/SciPy/scikit-learn 1.7.2, React 18, TypeScript, React Router 7, TanStack Query, Recharts, Vitest, Playwright, Docker.

**Spec:** `docs/superpowers/specs/2026-09-07-thesis-demo-evaluation-design.md`

## Global Constraints

- Work only on `prod/final-dashboard-demo`; never modify `main`.
- Preserve backend-authoritative calculations and frozen-model inference.
- Never present ARR or Silhouette as attack-detection accuracy.
- Preserve replay run isolation and raw-to-meta traceability.
- Keep external Wazuh, Shuffle, and Telegram status explicitly deferred.
- Use TDD for every behavior change and bug fix.

---

### Task 1: Correct evidence and durable runtime persistence

**Files:**
- Modify: `src/runtime/raw_evidence.py`
- Modify: `src/runtime/durable_state.py`
- Modify: `src/runtime/service.py`
- Modify: `src/rbta/engine.py`
- Test: `tests/unit/runtime/test_raw_evidence.py`
- Test: `tests/unit/runtime/test_service.py`
- Test: `tests/unit/runtime/test_durable_state.py`

**Interfaces:**
- Produces: accurate `RawAlertEvidenceStore.store(...) -> bool`, SQLite-backed dedup/history/outbox operations, bounded recent-history access.

- [ ] Add failing tests reproducing duplicate cached-count drift, weakened replay fingerprints, non-actionable outbox growth, and full-history restore.
- [ ] Run focused tests and confirm failures are caused by the audited behavior.
- [ ] Make SQLite row insertion authoritative, use complete canonical fingerprints, persist only actionable outbox entries, and page history from SQLite.
- [ ] Run focused and runtime integration tests.
- [ ] Commit the independently verified persistence correction.

### Task 2: Add dataset catalog and frozen provenance

**Files:**
- Create: `src/runtime/dataset_catalog.py`
- Modify: `src/runtime/replay_controller.py`
- Test: `tests/unit/runtime/test_dataset_catalog.py`
- Test: `tests/unit/runtime/test_replay_controller.py`

**Interfaces:**
- Produces: `ReplayDatasetCatalog.list()`, `open_text(name)`, and manifest DTO fields `sha256`, `total_events`, `timestamp_start`, `timestamp_end`, `compression`, `classification`, `cache_status`.

- [ ] Add failing tests for `.jsonl.gz`, cache reuse/invalidation, traversal rejection, and run provenance persistence.
- [ ] Implement streaming source inspection and atomic sidecar cache.
- [ ] Integrate catalog into dataset listing/start without a second full scan.
- [ ] Run focused replay tests and benchmark listing reuse.
- [ ] Commit dataset provenance support.

### Task 3: Make research evaluation fair and deterministic

**Files:**
- Create: `src/evaluation/contextual_fixed_baseline.py`
- Create: `src/evaluation/context_quality.py`
- Modify: `src/evaluation/noise_robustness.py`
- Modify: `src/evaluation/runtime_complexity.py`
- Modify: `src/research/orchestrator.py`
- Test: `tests/unit/evaluation/test_context_quality.py`
- Test: `tests/unit/evaluation/test_contextual_fixed_baseline.py`
- Test: `tests/unit/evaluation/test_noise_robustness.py`
- Test: `tests/unit/evaluation/test_runtime_complexity.py`
- Test: `tests/unit/research/test_orchestrator.py`

**Interfaces:**
- Produces: three-variant ablation records, context purity/contamination, matched-seed noise comparison, repeated runtime median/IQR and preparation timing.

- [ ] Add failing tests for context mixing, matched noise streams, repeat aggregation, and deterministic regression inputs.
- [ ] Implement contextual static aggregation and context-quality metrics.
- [ ] Expand noise evaluation across all variants using identical injected streams.
- [ ] Replace single micro-timing assertions with warm-up plus five-repeat median/IQR measurements.
- [ ] Update research artifacts and interpretation fields.
- [ ] Run the complete evaluation and orchestrator suites.
- [ ] Commit fair-evaluation changes.

### Task 4: Add replay live metrics and post-replay evaluation jobs

**Files:**
- Create: `src/runtime/replay_evaluation.py`
- Create: `src/runtime/evaluation_job.py`
- Modify: `src/runtime/replay_controller.py`
- Modify: `src/api/routes/replay.py`
- Test: `tests/unit/runtime/test_replay_evaluation.py`
- Test: `tests/unit/runtime/test_evaluation_job.py`
- Test: `tests/unit/api/test_dashboard_endpoints.py`

**Interfaces:**
- Produces: `evaluation_live` DTO and authenticated evaluation start/status/cancel/artifact endpoints.

- [ ] Add failing tests for online ARR/score statistics, frozen model provenance, lifecycle conflicts, phase progress, cancellation, and atomic artifacts.
- [ ] Implement O(1) live tracker and wire it to raw/scored events.
- [ ] Implement isolated post-replay job with explicit phases and error preservation.
- [ ] Add API endpoints and stable DTOs.
- [ ] Run focused runtime/API tests followed by all backend tests.
- [ ] Commit evaluation orchestration.

### Task 5: Harden API, deployment, dependencies, and bundle loading

**Files:**
- Modify: `src/api/app.py`
- Modify: `src/api/auth.py`
- Modify: `deploy/asus/compose.yml`
- Modify: `dashboard/package.json`
- Modify: `dashboard/package-lock.json`
- Modify: `dashboard/src/app/App.tsx`
- Test: `tests/unit/api/test_api_governance.py`
- Test: `tests/unit/api/test_app_endpoints.py`

**Interfaces:**
- Produces: security-header middleware, bounded control request rate, constant-time key comparison, patched Router 7 setup, route-level code splitting.

- [ ] Add failing API tests for security headers, request limits, timing-safe auth behavior, and control throttling.
- [ ] Implement scoped middleware without rate-limiting health probes.
- [ ] Enable read-only root filesystem with explicit writable runtime mounts.
- [ ] Upgrade React Router to 7.18.3 and migrate the router setup.
- [ ] Lazy-load feature pages and verify per-route chunks.
- [ ] Run dependency audit, frontend unit/lint/typecheck/build, and API suite.
- [ ] Commit hardening and dependency remediation.

### Task 6: Build the examiner-oriented Demo UI

**Files:**
- Create: `dashboard/src/features/demo/LiveEvaluationPanel.tsx`
- Create: `dashboard/src/features/demo/ResearchBoundaryCard.tsx`
- Create: `dashboard/src/features/demo/PostReplayEvaluation.tsx`
- Modify: `dashboard/src/features/replay/ReplayPage.tsx`
- Modify: `dashboard/src/components/shared/Sidebar.tsx`
- Modify: `dashboard/src/api/replay.ts`
- Modify: `dashboard/src/api/schemas.ts`
- Modify: `dashboard/e2e/dashboard.spec.ts`
- Test: matching Vitest files under `dashboard/src/features/demo/`

**Interfaces:**
- Consumes: backend dataset provenance, `evaluation_live`, and post-replay evaluation status.
- Produces: canonical `/demo` route and guided five-section thesis flow.

- [ ] Add failing schema/component/navigation tests for the new contracts and Indonesian explanation copy.
- [ ] Rename navigation and route while preserving `/replay` redirect compatibility.
- [ ] Implement preflight, live metrics, charts/tables, interpretations, loading/error/empty states, and artifact export.
- [ ] Refine shared spacing, hierarchy, responsive layout, and accessibility across touched pages.
- [ ] Run Vitest, lint, typecheck, build, and Playwright.
- [ ] Commit the Demo UI.

### Task 7: Align documentation and prepare Chapters IV–V

**Files:**
- Modify: `README.md`
- Modify: `docs/research-spec/05-EVALUATION-SPEC.md`
- Modify: `docs/research-spec/15-DASHBOARD-AND-DEMONSTRATION-SPEC.md`
- Create: `docs/thesis/BAB-IV-V-KERANGKA.md`
- Create: `docs/demo/SIDANG-RUNBOOK.md`

**Interfaces:**
- Produces: one source of truth for deployed/demo status, artifact-to-table mapping, safe claims, and a 5–7 minute runbook.

- [ ] Update terminology and remove contradictions around contamination, FPR, timezone, and external integrations.
- [ ] Document exact commands for frozen official evaluation and golden replay rehearsal.
- [ ] Write Chapter IV–V headings, required evidence, figure/table captions, interpretation prompts, limitations, and conclusion mapping without invented results.
- [ ] Run placeholder/contradiction scan and `git diff --check`.
- [ ] Commit the documentation set.

### Task 8: Final verification and release evidence

**Files:**
- Create: `docs/evidence/thesis-demo/VERIFICATION.md`

- [ ] Run the full Python suite with coverage and record counts.
- [ ] Run frontend audit, unit tests, lint, typecheck, production build, and Playwright.
- [ ] Build and smoke-test the Docker image and ASUS compose configuration when Docker is available.
- [ ] Run an engineering golden replay and complete post-replay evaluation; validate generated schemas and traceability.
- [ ] Run `git diff --check`, secret scan, and confirm `main` was never checked out or modified.
- [ ] Record exact SHA, model version, dataset hash, commands, environment, and any unavailable verification honestly.
