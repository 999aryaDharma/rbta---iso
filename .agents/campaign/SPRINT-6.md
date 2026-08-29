# Sprint 6 Contract — Batch and Replay Runners

## Objective

Prove batch research and replay streaming use the same Research Core and immutable model artifacts.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `06-CODEBASE-REFACTOR-SPEC.md`, `08-OPERATIONAL-INTEGRATION-SPEC.md`, `10-DUAL-MODE-RUNTIME-SPEC.md`, `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Implement batch runner, replay stream runner, and replay clock. Batch/replay must consume `CanonicalRawAlert`, the same `RBTAEngine`, the same seven-feature extractor, and the same model artifact inference path.

Replay supports 1x, 10x, 100x, and MAX. Sleep is based only on original event-time gap divided by speed; RBTA always receives the original timestamp. Replay loads immutable artifacts only and never trains/fits/calibrates.

Mandatory equivalence fixture: same canonical input + same artifacts through batch and MAX replay must produce equal final meta-alerts, equal seven-feature vectors, and equal scores/decisions.

## Gate S6

Batch/MAX-replay equivalence passes end-to-end, no duplicated Research Core exists, and full regression passes.
