# Sprint 8 — Research Evaluation Refactor & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-7-live-state`
* **Sprint Branch:** `refactor/sprint-8-evaluation`
* **Base Commit SHA:** `7a5d8f7663e6ef9fa95ae535b2e675fbba1f2249` (`7a5d8f7`)
* **Code Commit Tested:** `d51440810c02caa40811a4c9274df0e2bf077487` (`d514408`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Evaluation Methodology & Scientific Guarantees

| Requirement / Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Sensitivity Analysis** | Evaluates static $\Delta t \in \{1, 5, 10, 15, 20, 30, 45, 60\}$ minutes with `adaptive=False` (strict ceteris paribus) | `src/evaluation/sensitivity.py` |
| **Fixed Window Baseline** | Pure time-window tumbling baseline (calendar-only, non-contextual) to benchmark reduction | `src/evaluation/fixed_window_baseline.py` |
| **ARR Metric** | Exact reduction formula: $\text{ARR} = \frac{N_{\text{raw}} - N_{\text{meta}}}{N_{\text{raw}}} \times 100\%$ | `src/evaluation/metrics.py` |
| **Noise Robustness** | Injects realistic noise at $\{0\%, 5\%, 10\%, 20\%, 30\%\}$ rates; preserves valid `(agent_id, agent_name)` pairs, severity 1–4, zero MITRE tactics | `src/evaluation/noise_robustness.py` |
| **Runtime Complexity** | Evaluates 8 scaling subsets; measures throughput ($\text{alerts/ms}$); fits linear regression $N_{\text{alerts}} \to T_{\text{ms}}$ reporting actual $R^2$ | `src/evaluation/runtime_complexity.py` |
| **Structural Silhouette** | Silhouette on scaled 7-feature space for binary partition ($\text{ESCALATE}=1$, $\text{non-ESCALATE}=0$); 100 same-proportion permutations; reports null distribution, $z$-score, and empirical $p$-value | `src/evaluation/structural_silhouette.py` |
| **Zero Ground-Truth Contamination** | Legacy synthetic attack scenarios (A/B/C) and ground-truth metrics (PR-AUC, F1, F0.5) removed from primary evaluation pipeline | `src/evaluation/` |

---

## 3. Files Created & Modified

### Core Modules
* `src/evaluation/__init__.py`: Package interface.
* `src/evaluation/metrics.py`: `compute_arr` and `MetricsError`.
* `src/evaluation/fixed_window_baseline.py`: `run_fixed_window_baseline` and `FixedWindowResult`.
* `src/evaluation/sensitivity.py`: `run_delta_t_sensitivity_analysis` and `SensitivityResult`.
* `src/evaluation/noise_robustness.py`: `run_noise_robustness_evaluation` and `NoiseRobustnessResult`.
* `src/evaluation/runtime_complexity.py`: `run_runtime_complexity_evaluation` and `RuntimeComplexityResult`.
* `src/evaluation/structural_silhouette.py`: `run_structural_silhouette_evaluation` and `StructuralSilhouetteResult`.

### Legacy Files Removed
* `src/evaluation/attack_injector.py` (legacy synthetic attack scenarios removed)
* `src/evaluation/robustness.py` (replaced by authoritative `noise_robustness.py`)

### Tests
* `tests/unit/evaluation/test_metrics.py`: 2 unit tests verifying ARR calculation and boundary checks.
* `tests/unit/evaluation/test_fixed_window_baseline.py`: 1 unit test verifying non-contextual tumbling calendar windows.
* `tests/unit/evaluation/test_sensitivity.py`: 1 unit test verifying 8 static $\Delta t$ evaluations and elbow computation.
* `tests/unit/evaluation/test_noise_robustness.py`: 1 unit test verifying 5 noise rates, agent-name pairing, low severity, and degradation metrics.
* `tests/unit/evaluation/test_runtime_complexity.py`: 1 unit test verifying 8 scaling subsets, linear regression, and actual $R^2$ fitting.
* `tests/unit/evaluation/test_structural_silhouette.py`: 1 unit test verifying binary partition, 100 permutations, z-score, empirical p-value, and degenerate single-class handling.
* `tests/unit/evaluation/test_evaluation_governance.py`: 1 governance test ensuring zero synthetic attack ground-truth labels in primary evaluation modules.

---

## 4. Test Verification Evidence

### Targeted Evaluation Test Execution Output
```text
$ python -m pytest tests/unit/evaluation/ -q
........                                                                 [100%]
8 passed in 2.45s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
156 tests collected in 1.49s

$ python -m pytest -q
............................................................................................................................................................

============================ 156 passed in 6.97s ==============================
```

---

## 5. Gate Decision

```text
Gate S8: PASS
Ready for Sprint 9: YES
```
