# Sprint 6 — Batch and Replay Runners & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-5-historical-source`
* **Sprint Branch:** `refactor/sprint-6-dual-runners`
* **Base Commit SHA:** `7e9d5e5b3069b244793fc972b9a764d9f6979601` (`7e9d5e5`)
* **Code Commit Tested:** `b739c2ee3a9ea340665144398ebcb2f54b7b2dfb` (`b739c2e`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Dual-Mode Architecture & Runtime Guarantees

| Requirement / Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Shared Research Core** | Exactly one RBTA engine, one EMA temporal state, one 7-feature extractor, and one scoring runtime shared across all modes | `src/rbta/engine.py`, `src/features/extractor.py`, `src/model/scoring_pipeline.py` |
| **Batch Research Runner** | High-throughput offline processing producing `MetaAlert`s, 7-feature `pd.DataFrame`, and optional scored outputs | `src/runners/batch_runner.py` |
| **Replay Stream Runner** | Event-by-event streaming simulation with online inference; zero model training or fitting | `src/runners/replay_runner.py` |
| **Replay Pacing Clock** | Wall-clock pacing supporting `1x`, `10x`, `100x`, and `MAX` speed; RBTA always receives original event-time timestamp | `src/runners/clock.py` |
| **Mandatory Equivalence Proof** | Strict 100% equivalence in meta-alerts, 7-feature vectors, anomaly scores, thresholds, decisions, and actions between Batch and Replay MAX | `tests/integration/runners/test_batch_replay_equivalence.py` |

---

## 3. Files Created & Modified

### Core Modules
* `src/runners/__init__.py`: Package interface.
* `src/runners/clock.py`: `ReplayClock` and `ClockError`.
* `src/runners/batch_runner.py`: `BatchResearchRunner` and `BatchRunResult`.
* `src/runners/replay_runner.py`: `ReplayStreamRunner`.

### Tests
* `tests/unit/runners/test_clock.py`: 4 unit tests verifying speed factors (1x, 10x, 100x), MAX non-blocking, and retrograde timestamp handling.
* `tests/unit/runners/test_batch_runner.py`: 2 unit tests verifying offline batch feature extraction and model scoring.
* `tests/unit/runners/test_replay_runner.py`: 1 unit test verifying event-by-event streaming inference.
* `tests/unit/runners/test_runners_governance.py`: 2 governance tests proving zero model fitting in replay runner and strict shared-core usage.
* `tests/integration/runners/test_batch_replay_equivalence.py`: Mandatory end-to-end integration test proving 100% batch vs replay equivalence.

---

## 4. Test Verification Evidence

### Targeted Runners Test Execution Output
```text
$ python -m pytest tests/unit/runners/ tests/integration/runners/ -q
..........                                                               [100%]
10 passed in 3.23s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
140 tests collected in 2.14s

$ python -m pytest -q
............................................................................................................................................

============================ 140 passed in 6.22s ==============================
```

---

## 5. Equivalence Verification Proof

From `tests/integration/runners/test_batch_replay_equivalence.py` executing across 200 synthetic alerts (multi-agent, out-of-order jitter):

* **Batch Output Count:** 58 Scored Meta-Alerts
* **Replay Output Count:** 58 Scored Meta-Alerts
* **Meta-Alert Property Identity:** 100% match (`meta_id`, `agent_id`, `rule_group_primary`, `alert_count`, `max_severity`, `start_time`, `end_time`, `source_alert_ids`)
* **Seven-Feature Vector Identity:** 100% match across all 7 canonical features
* **Scoring & Decision Identity:** 100% match (`raw_model_score`, `anomaly_score`, `threshold_used`, `decision`, `action`, `escalate`)

---

## 6. Gate Decision

```text
Gate S6: PASS
Ready for Sprint 7: YES
```
