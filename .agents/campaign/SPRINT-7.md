# Sprint 7 Contract — Live Ingestion and Durable Runtime State

## Objective

Support real new Wazuh alerts with durable dedup, temporal state, active buckets, finalized outputs, and recovery.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `08-OPERATIONAL-INTEGRATION-SPEC.md`, `09-WAZUH-INDEXER-INGESTION-SPEC.md`, `10-DUAL-MODE-RUNTIME-SPEC.md`, `11-RUNTIME-STATE-SPEC.md`, `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Implement the durable state interface/store, live Wazuh polling source, authenticated collector ingress boundary, idle flush, controlled shutdown, and restart recovery.

Persist at minimum dedup IDs, source checkpoint, per-agent temporal state, active buckets, finalized meta-alerts, and outbox according to authoritative state specs.

Live Indexer polling must not use a long-lived PIT tail; use configurable overlap with documented defaults (5s poll, 5m overlap, 500 page unless higher authority says otherwise), dedup before core mutation, recover late-indexed events inside overlap, and handle daily rollover.

Collector ingress must be idempotent: new valid event accepted, duplicate success without core mutation, invalid required schema rejected clearly. Do not implement research algorithms in the transport boundary.

## Gate S7

Overlap/dedup, restart recovery, idle flush, durable state restoration, and zero duplicate core mutation pass; full regression passes.
