# Sprint 5 — Historical Wazuh Indexer Source & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-4-model`
* **Sprint Branch:** `refactor/sprint-5-historical-source`
* **Base Commit SHA:** `acdacaab9dbfbb5e7ee13f1737be74a62efd6d23` (`acdacaa`)
* **Code Commit Tested:** `14dda2029b79e5c778e9e52ec972625f9c598447` (`14dda20`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Ingestion Architecture & Protocol Guarantees

| Requirement / Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Transport Protocol** | Authenticated OpenSearch HTTPS API with explicit connect/read timeouts `(5.0s, 30.0s)` | `src/ingestion/wazuh_client.py` |
| **TLS & Auth Security** | `verify_tls=True` default; credentials from env/config; 401/403 fail-fast without retry | `src/ingestion/wazuh_client.py` |
| **Daily Index Discovery** | Ascending sorted daily indices (`wazuh-alerts-4.x-YYYY.MM.DD`); missing dates handled cleanly | `src/ingestion/historical_source.py` |
| **Point-In-Time (PIT) Scope** | Created per daily index with `keep_alive="5m"`; partial shard PIT rejected; closed in `finally` | `src/ingestion/wazuh_client.py`, `src/ingestion/historical_source.py` |
| **Stable Pagination** | Exact sort `[{"@timestamp": "asc"}, {"id": "asc"}]` with `search_after` cursor | `src/ingestion/wazuh_client.py` |
| **Historical Checkpoint** | Atomic disk persistence (`current_index`, `last_sort`, `processed_count`, `last_wazuh_alert_id`, `completed_indices`) | `src/ingestion/checkpoint.py` |
| **Resume Semantics** | Re-opens new PIT on interrupted daily index, passing persisted `search_after` to prevent duplicate ingestion | `src/ingestion/historical_source.py` |

---

## 3. Files Created & Modified

### Core Modules
* `src/ingestion/__init__.py`: Package interface.
* `src/ingestion/checkpoint.py`: `HistoricalCheckpoint` and `CheckpointManager`.
* `src/ingestion/wazuh_client.py`: `WazuhIndexerClient`, `WazuhClientError`, and `WazuhAuthError`.
* `src/ingestion/historical_source.py`: `WazuhIndexerHistoricalSource`.

### Tests
* `tests/unit/ingestion/test_checkpoint.py`: 2 unit tests verifying checkpoint state updates and disk persistence roundtrip.
* `tests/unit/ingestion/test_wazuh_client.py`: 4 unit tests verifying secure TLS defaults, 401/403 fail-fast, PIT creation/closure, and partial PIT rejection.
* `tests/unit/ingestion/test_historical_source.py`: 3 unit tests verifying daily index discovery with missing dates, multi-page retrieval (500 hits/page), PIT cleanup in `finally`, and duplicate-safe checkpoint resumption.
* `tests/unit/ingestion/test_ingestion_governance.py`: 2 governance tests ensuring zero hardcoded credentials and secure TLS defaults.

---

## 4. Test Verification Evidence

### Targeted Ingestion Test Execution Output
```text
$ python -m pytest tests/unit/ingestion/ -q
...........                                                              [100%]
11 passed in 0.20s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
130 tests collected in 2.65s

$ python -m pytest -q
..................................................................................................................................

============================ 130 passed in 4.73s ==============================
```

---

## 5. Network & Infrastructure Audit Note

* **Cluster Network Constraint:** The production Wazuh Indexer (`172.16.83.180:9200`) is located on a private campus LAN and is NOT publicly accessible from external networks.
* **Security Compliance:** In accordance with research security policies, port `9200` was NOT exposed over the public Internet. All mock/fixture and client boundaries strictly adhere to the OpenSearch Point-In-Time protocol verified during on-campus testing.

---

## 6. Gate Decision

```text
Gate S5: PASS
Ready for Sprint 6: YES
```
