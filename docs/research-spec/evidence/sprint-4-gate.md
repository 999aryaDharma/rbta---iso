# Sprint 4 — Isolation Forest Pipeline & Gate Evidence

**Research Title:** *RULE-BASED TEMPORAL AGGREGATION DAN ISOLATION FOREST UNTUK MITIGASI ALERT FATIGUE PADA LOG KEAMANAN SIEM WAZUH*  
**Document Type:** Normative Audit & Verification Record  
**Date:** 2026-08-28  

---

## 1. Traceability & Environment

* **Base Branch:** `refactor/sprint-3-features`
* **Sprint Branch:** `refactor/sprint-4-model`
* **Base Commit SHA:** `89e92cd712c4ef2db3b72aaefbfb92cf571ef99c` (`89e92cd`)
* **Code Commit Tested:** `570628d3b06767d61d55773702e9d7aaae109c07` (`570628d`)
* **Python Runtime:** Python 3.14.0 (Windows AMD64)
* **Test Framework:** pytest 9.0.3

---

## 2. Methodology Locks & Architecture

| Component | Authoritative Specification | Implementation |
| :--- | :--- | :--- |
| **Feature Scaler** | `RobustScaler` (fit on training reference only) | `src/model/scoring_pipeline.py` |
| **Estimator Config** | `IsolationForest(n_estimators=200, contamination="auto", random_state=42)` | `src/model/scoring_pipeline.py` |
| **Contamination Mode** | Strictly `"auto"`; zero ground truth or synthetic label dependencies | `src/model/scoring_pipeline.py` |
| **Raw Score Policy** | Oriented raw score: $\text{raw\_anomaly} = -\text{score\_samples}(X_{\text{scaled}})$ | `src/model/scoring_pipeline.py` |
| **Calibration Policy** | Stream-safe $\frac{\text{raw} - \text{min}}{\text{max} - \text{min}}$ derived from reference dataset | `src/model/calibration.py` |
| **Threshold Policy** | Tukey IQR outlier fence: $\theta = Q_3 + 1.5 \times \text{IQR}$ (unclamped) | `src/model/threshold.py` |
| **False Positive Gate** | $\text{sev} < 7 \land \text{count} < 5 \land \text{mitre} = 0 \land \text{score} \ge \theta \implies \text{CONTEXTUAL\_ANOMALY} \to \text{SUPPRESS}$ | `src/model/decision.py` |
| **Decision Matrix** | 4-quadrant evaluation combining anomaly score and rule severity | `src/model/decision.py` |
| **Artifact Lifecycle** | 6-file atomic staging publication bundle (`isolation_forest.joblib`, `robust_scaler.joblib`, `score_calibration.json`, `threshold.json`, `feature_schema.json`, `metadata.json`) | `src/model/registry.py` |

---

## 3. Files Created & Modified

### Core Modules
* `src/model/__init__.py`: Package interface.
* `src/model/calibration.py`: `ScoreCalibration` and `CalibrationError`.
* `src/model/threshold.py`: `TukeyThreshold`, `compute_tukey_threshold`, and `ThresholdError`.
* `src/model/decision.py`: `evaluate_decision` with False Positive Gate.
* `src/model/scoring_pipeline.py`: `ModelArtifactBundle`, `ScoringPipeline`, and `train_reference_pipeline`.
* `src/model/registry.py`: `ModelRegistry` with atomic staging publication and schema validation.

### Tests
* `tests/unit/model/test_calibration.py`: 3 unit tests verifying calibration scaling, degenerate bounds error, and roundtrip.
* `tests/unit/model/test_threshold.py`: 4 unit tests verifying Tukey IQR calculation, unclamped threshold, and error handling.
* `tests/unit/model/test_decision.py`: 3 unit tests verifying 4 Decision Matrix quadrants and False Positive Gate logic.
* `tests/unit/model/test_scoring_pipeline.py`: 3 unit tests verifying reference training, single-event parity with batch, and non-collapsing stream inference.
* `tests/unit/model/test_registry.py`: 3 unit tests verifying atomic staging publication, missing artifact validation, and schema mismatch fail-fast.
* `tests/unit/model/test_model_governance.py`: 2 governance tests ensuring zero dynamic contamination, zero ground truth usage, and zero live `fit` calls.

---

## 4. Test Verification Evidence

### Targeted Model Test Execution Output
```text
$ python -m pytest tests/unit/model/ -q
..................                                                       [100%]
18 passed in 5.71s
```

### Full Regression Test Suite Execution Output
```text
$ python -m pytest --collect-only -q
119 tests collected in 1.44s

$ python -m pytest -q
.......................................................................................................................

============================ 119 passed in 4.30s ==============================
```

---

## 5. Behavioral Proofs

1. **Single-Event Stream Reproducibility:** Single-event inference `pipeline.score_single(m)` produces scores, thresholds, decisions, and actions strictly identical to batch evaluation `pipeline.score_meta_alerts(metas)` without fitting or request-local normalization.
2. **Artifact Integrity:** Publishing and restoring artifact bundles via `ModelRegistry` preserves exact model estimator states, scaler parameters, calibration bounds, and threshold values.
3. **Fail-Fast Invariants:** Incompatible feature schemas or missing bundle artifacts fail readiness immediately without fallback.

---

## 6. Gate Decision

```text
Gate S4: PASS
Ready for Sprint 5: YES
```
