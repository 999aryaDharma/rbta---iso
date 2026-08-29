# Sprint 3 — Exact Seven-Feature Extractor & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-2-rbta-core`
* **Sprint Branch:** `refactor/sprint-3-features`
* **Base Commit SHA:** `453e3c63148154101e4a1a6b0c29fe0270a6c0c4` (`453e3c6`)
* **Code Commit Tested:** `2a3ba53d838fdd38ee3991cff6cff0c1671cdb4c` (`2a3ba53`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Authoritative Feature Schema

The single authoritative feature vector is defined in `src/features/extractor.py`:

```python
FEATURE_COLUMNS: tuple[str, ...] = (
    "max_severity",
    "mitre_tactic_count",
    "critical_mitre_tactic_present",
    "alert_count_log",
    "rule_diversity_shannon",
    "severity_dispersion",
    "agent_criticality",
)
```

| Feature Name | Type | Mathematical / Domain Definition | Valid Range |
| :--- | :--- | :--- | :--- |
| **`max_severity`** | float | $\max(\text{rule\_level} \in \text{bucket})$ | $[0.0, 15.0]$ |
| **`mitre_tactic_count`** | float | $\text{len}(\text{mitre\_tactics\_unique})$ | $\ge 0.0$ |
| **`critical_mitre_tactic_present`** | float | $1.0$ if $\text{critical\_mitre\_present}$ else $0.0$ | $\{0.0, 1.0\}$ |
| **`alert_count_log`** | float | $\ln(1 + \text{alert\_count}) = \text{log1p}(\text{alert\_count})$ | $> 0.0$ |
| **`rule_diversity_shannon`** | float | $\frac{-\sum p_i \ln(p_i)}{\ln(k)}$ for $k > 1$, else $0.0$ | $[0.0, 1.0]$ |
| **`severity_dispersion`** | float | $\sqrt{\frac{\sum c_i (\text{sev}_i - \mu)^2}{N}}$ (population std), else $0.0$ for singletons | $\ge 0.0$ |
| **`agent_criticality`** | float | Domain asset score from centralized config | $[1.0, 4.0]$ |

---

## 3. Files Created & Modified

### Core Modules
* `src/features/__init__.py`: Package entrypoint exporting `SevenFeatureExtractor`, `FEATURE_COLUMNS`, and extraction helpers.
* `src/features/extractor.py`: Canonical `SevenFeatureExtractor`, `compute_rule_diversity_shannon`, and `compute_severity_dispersion`.

### Tests
* `tests/unit/features/test_extractor.py`: 12 unit tests verifying mathematical formulas, boundary conditions, edge cases, DataFrame output schema, and error handling.
* `tests/unit/features/test_feature_governance.py`: 1 governance test ensuring a single authoritative `FEATURE_COLUMNS` declaration in active source.

---

## 4. Test Verification Evidence

### Targeted Feature Test Execution Output
```text
$ python -m pytest tests/unit/features/ -q
.............                                                            [100%]
13 passed in 0.43s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
101 tests collected in 0.60s

$ python -m pytest -q
.....................................................................................................

============================ 101 passed in 2.08s ==============================
```

---

## 5. Repository Governance & Clean Architecture

* **Active `FEATURE_COLUMNS` declarations in `src/`:** Declared strictly in `src/features/extractor.py` and exported by `src/features/__init__.py`.
* **Zero silent fallback:** Missing required fields or non-finite values raise explicit `FeatureExtractionError`.
* **Zero circular dependencies:** `src/features` depends only on `src/contracts` and numerical standard libraries / pandas.

---

## 6. Gate Decision

```text
Gate S3: PASS
Ready for Sprint 4: YES
```
