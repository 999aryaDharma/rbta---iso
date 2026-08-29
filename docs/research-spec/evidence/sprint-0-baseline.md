# Sprint 0 — Repository Baseline & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Baseline Record  
**Date:** 2026-08-28  

---

## 1. Repository Baseline & Traceability

* **Branch:** `refactor/sprint-0-1-foundation` (branched from `main` @ `189c7f7`)
* **Baseline Commit SHA:** `189c7f7`
* **Code Commit Tested:** `bcb24ac`
* **Python Runtime:** Python 3.14.0 (Windows AMD64, compatible with `>=3.11` specification)
* **Test Framework:** pytest 9.0.3 (active) / pytest 9.1.1 (`.venv-gate`)

---

## 2. Legacy Primary Entry Points

Before refactoring, the repository contained the following operational and research entry points:

1. **`main.py`**: Legacy orchestrator executing a 14-step pipeline based on prior Landauer alignment.
2. **`src/etl/preprocessing_01.py`**: Batch CSV loader (`load_and_prepare`), filtering corrupt rows, mapping columns to internal names without pre-sorting.
3. **`src/etl/json_orches.py`**: Batch JSONL parser extracting raw Wazuh alerts into intermediate CSVs (`rbta_ready_ALL.csv`).
4. **`src/engine/rbta_core.py`**: Legacy RBTA v5 algorithm (`run_rbta`) implementing `OutOfOrderBuffer` (min-heap), global `ElasticWindow`, `Watermark`, `MetaAlert` (Bucket A), and `CompoundMetaAlert` (Bucket B).
5. **`src/engine/fixed_window_baseline.py`**: Fixed tumbling window baseline comparison without contextual grouping.
6. **`src/engine/feature_engineering.py`**: 11-feature vector extractor (`enrich_features`) computing `alert_velocity`, `rule_concentration`, `severity_spread`, and `deviation_from_baseline`.
7. **`src/engine/isolation_forest.py`**: Legacy Isolation Forest pipeline (`run_pipeline`) with batch-level min/max score normalization, dynamic contamination, and 4-quadrant decision matrix.
8. **`src/evaluation/metrics.py`**: Metrics evaluation suite (sensitivity analysis, elbow method, ARR per rule group, FPR vs reduction, runtime complexity proof, scenario evaluations).
9. **`src/evaluation/attack_injector.py`**: Synthetic attack injector for scenarios A, B, and C with label propagation.
10. **`src/evaluation/robustness.py`**: Noise injection test suite evaluating noise absorption across varying noise rates.
11. **`src/streaming/alert_stream_simulator.py`**: Historical CSV replay simulator with accelerated replay clock.

---

## 3. Legacy Pipeline Summary

```text
CSV/JSON Input
  ↓
load_and_prepare() [preprocessing_01.py]
  ↓ (optional)
run_injection() [attack_injector.py - Scenarios A/B/C]
  ↓
sensitivity_analysis() [metrics.py]
  ↓
run_rbta(delta_t) [rbta_core.py - Global ElasticWindow, Bucket A + Bucket B, Watermark late_drop]
  ↓
add_if_features() + enrich_features() [feature_engineering.py - 11 HIDS features]
  ↓ (optional)
propagate_labels() [attack_injector.py]
  ↓
run_fixed_window() [fixed_window_baseline.py]
  ↓
compute_arr_per_group() [metrics.py]
  ↓
run_pipeline() [isolation_forest.py - RobustScaler, IF(n=200), dynamic contamination, current-batch min/max, clamped Tukey IQR]
  ↓
compute_fpr_vs_reduction() + noise_robustness_test() + runtime_complexity_proof()
  ↓
Scenario A/B Reports + Visualizations
```

---

## 4. Legacy Contracts and Output Shapes

1. **Preprocessed Raw Alert (Pandas DataFrame):**
   * Columns: `timestamp`, `agent_id`, `agent_name`, `rule_groups`, `rule_level`, `srcip`, `srcip_type`, `rule_id`, `criticality_score`, `has_mitre`, `has_critical_mitre`.
2. **RBTA Meta-Alert (Bucket A):**
   * Columns: `meta_id`, `parent_meta_id`, `agent_id`, `agent_name`, `rule_groups`, `start_time`, `end_time`, `duration_sec`, `alert_count`, `max_severity`, `attacker_count`, `rule_group_severity_enc`, `agent_criticality`, `hour_of_day`, `unique_rules_triggered`, `mitre_hit_count`, `external_threat_count`, `internal_src_count`, `attacker_ips`, `severity_dist`, `rule_id_dist`, `rule_group_dist`, `mitre_tactic`, `wazuh_alert_ids`.
3. **Compound Meta-Alert (Bucket B - To be removed):**
   * Columns: `compound_id`, `agent_id`, `agent_name`, `window_id`, `window_start`, `window_end`, `start_time`, `end_time`, `duration_sec`, `alert_count`, `max_severity`, `mitre_hit_count`, `n_rule_groups`, `attacker_count`, `attacker_ips`, `rule_group_dist`, `mitre_tactic`.
4. **Legacy 11-Feature Matrix (To be replaced by 7 Canonical Features):**
   * Features: `alert_count_log`, `max_severity`, `duration_sec`, `rule_group_severity_enc`, `agent_criticality`, `hour_of_day`, `alert_velocity`, `mitre_hit_count`, `rule_concentration`, `severity_spread`, `deviation_from_baseline`.
5. **Scored Meta-Alert:**
   * Columns: Meta-alert columns + `anomaly_score`, `decision` (`CRITICAL`, `SUSPICIOUS`, `NOISE_HIGH`, `NOISE`, `CONTEXTUAL_ANOMALY`), `action` (`ESCALATE`, `DAILY_DIGEST`, `SUPPRESS`), `escalate` (0/1), and optional `ground_truth`.

---

## 5. Known Legacy Contradictions against Research Specifications

| Audit Finding | Legacy Behavior | Normative Specification (`00`–`14`) | Refactor Phase |
| :--- | :--- | :--- | :--- |
| **P0-01** | Final RBTA ran with `enable_adaptive=False` | Final RBTA MUST run with adaptive ETW enabled per agent | Sprint 2 |
| **P0-02** | Single global `ElasticWindow` state | Independent `AgentTemporalState` per `agent_id` | Sprint 2 |
| **P0-03** | Step-based adaptation (`HIGH_FREQ`, `LOW_FREQ`, 0.8x/1.2x) | Proportional formula: `base_dt * (EMA_gap / baseline_gap)` clipped to `[0.5x, 1.5x]` | Sprint 2 |
| **P0-04** | Watermark `late_drop` silently dropped late alerts | No `late_drop` policy; 100% valid parsed alerts must be processed | Sprint 2 |
| **P0-05** | Dual buckets (Bucket A + Bucket B Compound) | Single bucket: `(agent_id, rule_group_primary)` | Sprint 2 |
| **P0-06** | 11-feature HIDS vector | Exactly 7 canonical features in authoritative order | Sprint 3 |
| **P0-07** | Feature calculation duplicated across modules | Single `SevenFeatureExtractor` module | Sprint 3 |
| **P0-08** | Dynamic contamination computed from synthetic labels | Fully unsupervised `contamination="auto"`, zero ground-truth influence | Sprint 4 |
| **P0-09** | Tukey threshold clamped to 1.0 (`min(Q3 + 1.5*IQR, 1.0)`) | Pure Tukey IQR fence `Q3 + 1.5*IQR` without 1.0 clamping | Sprint 4 |
| **P0-10** | Live streaming score normalized with batch min/max | Calibration parameters persisted at training; loaded at inference | Sprint 4 |
| **P0-11** | Synthetic attack scenarios A/B/C in primary pipeline | Removed from primary research pipeline; structural validity evaluation used | Sprint 8 |
| **P0-12** | Missing native Wazuh Indexer PIT / `search_after` | Resumable `WazuhIndexerHistoricalSource` with daily PIT | Sprint 5 |
| **P0-13** | Workstation-specific hardcoded Windows paths (`D:\KAMPUS\...`) | External configuration via relative path / CLI / config files | Sprint 0–1 (Resolved) |
| **P1-07** | Domain constants duplicated across files | Single authoritative `src/config/domain.py` | Sprint 1 (Resolved) |

---

## 6. Clean Installation & Test Evidence

### Clean Virtual Environment Verification (`.venv-gate`)
* **Editable Installation:** PASS
* **Install Command:** `.venv-gate\Scripts\python -m pip install -e ".[dev]"`
* **Installed Project:** `rbta-wazuh-isolation-forest==0.1.0`

### Pytest Collection
```text
$ python -m pytest --collect-only -q
tests/unit/config/test_domain.py: 9
tests/unit/config/test_domain_governance.py: 2
tests/unit/contracts/test_immutability.py: 3
tests/unit/contracts/test_meta_alert.py: 5
tests/unit/contracts/test_raw_alert.py: 7
tests/unit/contracts/test_scored_meta_alert.py: 4
tests/unit/etl/test_json_orches.py: 1
tests/unit/etl/test_wazuh_canonicalizer.py: 15
tests/unit/test_smoke.py: 3

49 tests collected in 3.35s
```

### Pytest Execution
```text
$ python -m pytest -q
.................................................

49 passed in 2.03s
```
