# Sprint 9 Contract — Operational API, Durable Outbox, Shuffle

## Objective

Expose scored results safely while keeping Shuffle and downstream notification layers free of research logic.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `08-OPERATIONAL-INTEGRATION-SPEC.md`, `10-DUAL-MODE-RUNTIME-SPEC.md`, `11-RUNTIME-STATE-SPEC.md`, `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Provide `/health` liveness; `/ready` that verifies required durable state and complete compatible model/scaler/calibration/threshold/schema artifacts; `/runtime/stats`; meta-alert detail/trace endpoints; durable outbox; stable downstream `event_id`/`meta_id` idempotency; REST/OpenAPI Shuffle adapter; downstream Telegram formatting only.

Shuffle must never implement EMA, RBTA, feature extraction, IF scoring/training, threshold calibration, or decision methodology.

Mandatory end-to-end fixture: Wazuh-like input -> canonical alert -> RBTA -> seven features -> immutable IF inference/decision -> durable outbox -> Shuffle stub exactly once, including retry/idempotency behavior.

Do not start ASUS deployment or dashboard work.

## Gate S9

The complete Wazuh-like-to-Shuffle-stub path is exactly-once at the downstream boundary, readiness accurately fails on missing/incompatible dependencies, no research logic exists in Shuffle, and full regression passes.
