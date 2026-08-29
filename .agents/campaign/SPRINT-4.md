# Sprint 4 Contract — Isolation Forest and Immutable Artifacts

## Objective

Make training explicit and inference reproducible for single-event streaming.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `03-FEATURE-ENGINEERING-SPEC.md`, `04-ISOLATION-FOREST-SPEC.md`, `06-CODEBASE-REFACTOR-SPEC.md`, `07-IMPLEMENTATION-CHECKLIST.md`, `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Implement the shared model runtime, calibration, Tukey/decision logic, and artifact registry according to the spec.

Locked model requirements include RobustScaler, `IsolationForest(n_estimators=200, contamination="auto")`, fixed random state, no ground-truth parameterization, calibration fitted only from training/reference scores, no per-request min/max, Tukey `Q3 + 1.5*IQR` without clamp, the exact Decision Matrix and False Positive Gate, and feature-schema compatibility checks.

Registry must write/load exactly the artifact set required by the sprint plan and publish atomically from staging.

Required proof includes artifact roundtrip in a new load context: the same feature vector with the same artifact yields the same score/decision; missing/incompatible artifacts fail readiness; single-event inference does not fit scaler/model/calibration/threshold and does not collapse via request-local normalization.

## Gate S4

Artifact roundtrip passes, single-event inference is reproducible and non-degenerate, full regression passes, and governance proves zero live/replay fit operations.
