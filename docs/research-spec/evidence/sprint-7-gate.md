# Sprint 7 — Live Ingestion & Durable Runtime State Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-6-dual-runners`
* **Sprint Branch:** `refactor/sprint-7-live-state`
* **Base Commit SHA:** `847667f70b7405cb6b23cb1ee2d354b3ee428c9b` (`847667f`)
* **Code Commit Tested:** `ec33c3f96f998390480b01ae041a859ef3d111bf` (`ec33c3f`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Ingestion & Durable State Architecture

| Requirement / Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Durable State Store** | Atomic snapshot serialization and crash-recovery for seen alert IDs, `_meta_id_counter`, `AgentTemporalState`, `_active_buckets`, `source_checkpoint`, and `outbox` | `src/runtime/durable_state.py` |
| **Live Indexer Poller** | Non-PIT polling with configurable overlap window (default 5m) and interval (default 5s); dedup before engine mutation; daily index rollover support | `src/runtime/live_source.py` |
| **Collector Ingress Boundary** | Authenticated webhook ingress; canonical parsing; idempotency returning 200 without duplicate engine mutation; clear 400 rejection on invalid schema | `src/runtime/ingress.py` |
| **Live RBTA Service** | Stateful service coordinating ingestion, scoring, event-time idle bucket flush (`idle_gap > delta_t`), outbox queueing/ack, and controlled shutdown | `src/runtime/service.py` |

---

## 3. Files Created & Modified

### Core Modules
* `src/runtime/__init__.py`: Package interface.
* `src/runtime/durable_state.py`: `DurableStateManager`.
* `src/runtime/live_source.py`: `WazuhIndexerLivePoller`.
* `src/runtime/ingress.py`: `CollectorIngressBoundary`, `IngressResult`, and `IngressPayloadError`.
* `src/runtime/service.py`: `LiveRBTAService`.

### Tests
* `tests/unit/runtime/test_durable_state.py`: 1 unit test verifying save/restore of engine seen IDs, temporal states, and active buckets.
* `tests/unit/runtime/test_live_poller.py`: 1 unit test verifying overlap queries and pre-core deduplication.
* `tests/unit/runtime/test_ingress_boundary.py`: 3 unit tests verifying authentication, valid payload ingestion, duplicate idempotency, and malformed payload rejection.
* `tests/unit/runtime/test_service.py`: 2 unit tests verifying live alert scoring, idle flushing, outbox acknowledgment, controlled shutdown, and restart recovery.
* `tests/unit/runtime/test_runtime_governance.py`: 1 governance test ensuring zero duplicate research core declarations in runtime modules.

---

## 4. Test Verification Evidence

### Targeted Runtime Test Execution Output
```text
$ python -m pytest tests/unit/runtime/ -q
........                                                                 [100%]
8 passed in 2.24s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
148 tests collected in 1.98s

$ python -m pytest -q
....................................................................................................................................................

============================ 148 passed in 6.12s ==============================
```

---

## 5. Gate Decision

```text
Gate S7: PASS
Ready for Sprint 8: YES
```
