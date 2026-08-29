# Sprint 2 — RBTA Core & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-0-1-foundation`
* **Sprint Branch:** `refactor/sprint-2-rbta-core`
* **Base Commit SHA:** `86e13598f047f350d49b66c70c3847f5b0738580` (`86e1359`)
* **Code Commit Tested:** `ad8f3c3ca28d79a000e3a8d2c2f23678e3850ae2` (`ad8f3c3`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Research Methodology Locks Implemented

| Parameter / Rule | Authoritative Value / Behavior | Implementation Location |
| :--- | :--- | :--- |
| **EMA Smoothing Factor ($\alpha$)** | Exactly `0.10` | `src/config/research.py` |
| **Warmup Horizon** | First `100` events per agent | `src/rbta/temporal_state.py` |
| **Baseline Statistic** | Arithmetic mean of available local forward gaps during warmup | `src/rbta/temporal_state.py` |
| **Adaptive Update Timing** | Starts at event `101` per agent | `src/rbta/temporal_state.py` |
| **ETW Ratio Formula** | $\text{ratio} = \frac{\text{EMA\_gap}}{\text{baseline\_gap}}$ | `src/rbta/temporal_state.py` |
| **ETW Bounds** | $[0.5 \times \Delta t_{\text{base}}, 1.5 \times \Delta t_{\text{base}}]$ | `src/config/research.py` |
| **RBTA Bucket Key** | Exactly `(agent_id, rule_group_primary)` | `src/rbta/engine.py` |
| **Max Bucket Duration** | $60$ minutes ($3600$ seconds) | `src/config/research.py` |
| **Failure-Atomicity** | Transactional processing: failed process attempts cause zero state mutation and do not poison seen alert IDs | `src/rbta/engine.py` |
| **Zero-Baseline Policy** | Terminal invalid state if warmup baseline is $\le 0$; never extends warmup beyond 100 events | `src/rbta/temporal_state.py` |
| **Fixed Mode Policy** | `adaptive=False` maintains constant $\Delta t = \Delta t_{\text{base}}$ without requiring an adaptive baseline | `src/rbta/temporal_state.py` |
| **Residual Retrograde Handling** | Monotonic event-time; no negative EMA gaps; non-mergeable residual $\to$ immediate singleton | `src/rbta/temporal_state.py`, `src/rbta/engine.py` |
| **Ingress Idempotency** | Second occurrence of `wazuh_alert_id` $\to$ zero mutation / no-op | `src/rbta/engine.py` |

---

## 3. Files Created & Modified

### Research Configuration
* `src/config/research.py`: Methodology locks (`EMA_ALPHA=0.10`, `WARMUP_EVENT_TARGET=100`, `MAX_BUCKET_DURATION=60m`, bounds `0.5x..1.5x`).
* `src/config/__init__.py`: Exported research constants.

### Core Modules
* `src/rbta/__init__.py`: Public package interface.
* `src/rbta/temporal_state.py`: `AgentTemporalState` and `TemporalStateError` with snapshotting, terminal invalid failure state, and baseline-independent fixed mode.
* `src/rbta/reorder_buffer.py`: `LosslessReorderBuffer` ($O(n \log k)$ bounded min-heap with monotonic arrival sequence tie-breaker).
* `src/rbta/engine.py`: `RBTAEngine`, `_ActiveBucket`, and `RBTAInvariantError` with transactional snapshot-and-commit failure-atomic processing.

### Tests
* `tests/unit/rbta/test_temporal_state.py`: 12 unit tests covering warmup, adaptation, clamps, terminal invalid baseline, and fixed mode.
* `tests/unit/rbta/test_reorder_buffer.py`: 7 unit tests covering bounded reordering, ties, idempotency, and lossless conservation.
* `tests/unit/rbta/test_engine.py`: 16 unit tests covering single-bucket aggregation, boundary merges/splits, idempotency, failure-atomicity, retry semantics, and fixed mode.
* `tests/unit/rbta/test_rbta_governance.py`: 2 unit tests verifying zero forbidden legacy tokens in `src/rbta/` and instance isolation.
* `tests/integration/rbta/test_mapping_integrity.py`: 2 integration tests proving 100% event conservation, multiplicity-1 mapping integrity, and deterministic reproducibility.

---

## 4. Test Verification Evidence

### Targeted Test Suite Execution Output
```text
$ python -m pytest tests/unit/rbta/test_temporal_state.py -q
............                                                             [100%]
12 passed in 0.04s

$ python -m pytest tests/unit/rbta/test_reorder_buffer.py -q
.......                                                                  [100%]
7 passed in 0.05s

$ python -m pytest tests/unit/rbta/test_engine.py -q
................                                                         [100%]
16 passed in 0.06s

$ python -m pytest tests/unit/rbta/test_rbta_governance.py -q
..                                                                       [100%]
2 passed in 0.04s

$ python -m pytest tests/integration/rbta -q
..                                                                       [100%]
2 passed in 0.08s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
88 tests collected in 0.46s

$ python -m pytest -q
........................................................................................

============================= 88 passed in 1.89s ==============================
```

### Module Breakdown
* `tests/integration/rbta/test_mapping_integrity.py`: 2 passed
* `tests/unit/config/test_domain.py`: 9 passed
* `tests/unit/config/test_domain_governance.py`: 2 passed
* `tests/unit/contracts/test_immutability.py`: 3 passed
* `tests/unit/contracts/test_meta_alert.py`: 5 passed
* `tests/unit/contracts/test_raw_alert.py`: 7 passed
* `tests/unit/contracts/test_scored_meta_alert.py`: 4 passed
* `tests/unit/etl/test_json_orches.py`: 1 passed
* `tests/unit/etl/test_wazuh_canonicalizer.py`: 15 passed
* `tests/unit/rbta/test_engine.py`: 16 passed
* `tests/unit/rbta/test_rbta_governance.py`: 2 passed
* `tests/unit/rbta/test_reorder_buffer.py`: 7 passed
* `tests/unit/rbta/test_temporal_state.py`: 12 passed
* `tests/unit/test_smoke.py`: 3 passed

---

## 5. Mapping Integrity & Behavioral Proof

From the end-to-end integration test (`tests/integration/rbta/test_mapping_integrity.py`):

```text
Ingress Records:                        239
Duplicate Ingress Records:                1
Processed Unique Raw Alerts:            238
Sum of MetaAlert.alert_count:           238
Unique Source Alert IDs in MetaAlerts:  238
Missing Alert IDs:                        0
Duplicate Alert Memberships:              0
Duration Invariant Violations (>60m):     0
Negative Duration Violations (<0s):       0
```

### Manual Mathematics Check
* **Baseline Calculation:** 100 events with 10.0s inter-arrival gaps $\to \text{baseline\_gap} = 10.0\text{s}$, $\text{ema\_gap}_0 = 10.0\text{s}$.
* **Adaptive Update (Event 101, gap = 5.0s):**
  $$\text{ema\_gap}_1 = 0.10 \times 5.0 + 0.90 \times 10.0 = 9.5\text{s}$$
  $$\text{ratio} = \frac{9.5}{10.0} = 0.95$$
  $$\Delta t_{\text{current}} = 600\text{s} \times 0.95 = 570.0\text{s} = 9.5\text{ minutes}$$
* **Result:** PASS. Verified in unit tests and integration tests.

---

## 6. Repository Governance & Legacy Isolation

* **Forbidden legacy tokens in `src/rbta/`:** `0 matches` (`HIGH_FREQ`, `LOW_FREQ`, `SHRINK_RATE`, `EXPAND_RATE`, `late_drop`, `CompoundMetaAlert`, `ground_truth`, `is_synthetic`, `IsolationForest`).
* **Workstation paths in `src/rbta/` and evidence:** `0 matches` (`D:\`, `C:\`, `file:///D:`, `file:///C:`).
* **Remaining legacy code in repository:** Legacy `src/engine/rbta_core.py` and `src/engine/feature_engineering.py` remain untouched and are NOT imported by `src/rbta/`. Full deprecation and replacement will follow in Sprint 3 (`SevenFeatureExtractor`) and Sprint 4 (`IsolationForestPipeline`).

---

## 7. Gate Decision

```text
Gate S2: PASS
Ready for Sprint 3: YES
```
