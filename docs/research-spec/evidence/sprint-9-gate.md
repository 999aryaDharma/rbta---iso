# Sprint 9 — Operational API, Durable Outbox & Shuffle Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-8-evaluation`
* **Sprint Branch:** `refactor/sprint-9-api-outbox-shuffle`
* **Base Commit SHA:** `c9c3deca48a314c46647225c56d7fe316827011d` (`c9c3dec`)
* **Code Commit Tested:** `58a24dd2eadbb40eac8b19eb2ecb0e17588592c8` (`58a24dd`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. API & Integration Architecture

| Requirement / Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Liveness & Readiness Probes** | `GET /health` returns liveness; `GET /ready` strictly verifies active published model bundle, schema compatibility, and runtime initialization (503 if missing) | `src/api/app.py` |
| **Runtime Aggregation Stats** | `GET /runtime/stats` returns seen IDs, active buckets count, outbox depth, and counter | `src/api/app.py` |
| **Authenticated Ingress Boundary** | `POST /api/v1/alerts/ingest` validates Bearer token, canonicalizes alert schema, rejects bad payload (400), and handles duplicates idempotently | `src/api/app.py`, `src/runtime/ingress.py` |
| **Durable Outbox & Acknowledgment** | `GET /api/v1/outbox` and `POST /api/v1/outbox/{meta_id}/ack` for reliable downstream delivery | `src/api/app.py`, `src/runtime/service.py` |
| **Shuffle SOAR Webhook Adapter** | `ShuffleWebhookForwarder` dispatches scored meta-alerts with retry and idempotent `X-Event-ID: rbta-meta-{meta_id}` header | `src/api/shuffle_adapter.py` |
| **Downstream Presentation Formatter** | `format_telegram_alert` generates presentation-only Markdown notifications without performing any research calculations | `src/api/telegram_formatter.py` |
| **Mandatory End-to-End Proof** | Complete flow from raw Wazuh-like alert $\to$ canonical alert $\to$ RBTA $\to$ 7 features $\to$ model inference $\to$ durable outbox $\to$ Shuffle webhook receiver exactly once | `tests/integration/api/test_e2e_wazuh_to_shuffle.py` |

---

## 3. Files Created & Modified

### Core Modules
* `src/api/__init__.py`: Package interface.
* `src/api/app.py`: FastAPI application factory `create_app` exposing `/health`, `/ready`, `/runtime/stats`, `/api/v1/alerts/ingest`, `/api/v1/outbox`, `/api/v1/outbox/{meta_id}/ack`, and `/api/v1/meta-alerts/{meta_id}`.
* `src/api/shuffle_adapter.py`: `ShuffleWebhookForwarder` with retry and idempotent headers.
* `src/api/telegram_formatter.py`: `format_telegram_alert`.
* `src/model/registry.py`: Added `get_active_version()` discovery method.

### Tests
* `tests/unit/api/test_app_endpoints.py`: 4 unit tests verifying health, ready failure (503), ready success (200), and ingest/outbox lifecycle.
* `tests/unit/api/test_shuffle_adapter.py`: 1 unit test verifying webhook forwarding and idempotent header delivery.
* `tests/unit/api/test_telegram_formatter.py`: 1 unit test verifying presentation-only Markdown formatting.
* `tests/unit/api/test_api_governance.py`: 1 governance test ensuring zero research/model logic in Shuffle and Telegram adapters.
* `tests/integration/api/test_e2e_wazuh_to_shuffle.py`: Mandatory end-to-end integration test proving exactly-once delivery from raw Wazuh alert to Shuffle webhook.

---

## 4. Test Verification Evidence

### Targeted API Test Execution Output
```text
$ python -m pytest tests/unit/api/ tests/integration/api/ -q
........                                                                 [100%]
8 passed in 3.23s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
164 tests collected in 2.50s

$ python -m pytest -q
....................................................................................................................................................................

============================ 164 passed in 8.21s ==============================
```

---

## 5. End-to-End Exactly-Once Delivery Proof

From `tests/integration/api/test_e2e_wazuh_to_shuffle.py`:

1. **Ingest Event 1 (10:00, Severity 12, MITRE: Initial Access):** Accepted as new event, buffered into active bucket `("001", "pam")`.
2. **Retry Ingest Event 1 (duplicate):** Accepted as duplicate (`is_duplicate: True`), zero mutation to active bucket or seen IDs.
3. **Ingest Event 2 (10:20, 20-min gap > 15-min $\Delta t$):** Finalizes bucket 1, extracts 7 features, scores via reference model pipeline, queues ScoredMetaAlert into durable outbox.
4. **Outbox Retrieval:** Outbox contains exactly 1 item (`meta_id=1`).
5. **Shuffle Forwarding:** Forwarded with header `X-Event-ID: rbta-meta-1` and payload. Shuffle received exactly 1 webhook.
6. **Outbox Acknowledgment:** `POST /api/v1/outbox/1/ack` dequeues item.
7. **Post-Ack State:** Outbox is empty (`0 items`), durable state is persisted and consistent.

---

## 6. Campaign Scope Boundary

* **Sprints 0 through 9 Completed:** All research and core engineering deliverables are fully implemented, verified, and audited.
* **Sprint 10 (Deployment) & Sprint 11 (Live Dashboard):** Excluded from this research campaign as specified in master rules.

---

## 7. Gate Decision

```text
Gate S9: PASS
Research Campaign (Sprints 0–9): COMPLETED
```
