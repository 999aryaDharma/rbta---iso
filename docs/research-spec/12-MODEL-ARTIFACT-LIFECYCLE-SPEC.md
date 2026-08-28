# Model Artifact Lifecycle Specification

## Goal

Memisahkan secara tegas **training/calibration** dari **inference** agar batch research dan stream/live mode memakai model yang reproducible dan tidak melakukan fit ulang diam-diam.

## 1. Artifact Bundle

Setiap training menghasilkan directory versioned:

```text
artifacts/models/<model_version>/
├── isolation_forest.joblib
├── robust_scaler.joblib
├── score_calibration.json
├── threshold.json
├── feature_schema.json
└── metadata.json
```

Minimum recommended model version:

```text
rbta-if-YYYYMMDD-<short_hash>
```

## 2. `feature_schema.json`

Harus menyimpan exactly tujuh field dalam urutan tetap:

```json
{
  "schema_version": "1.0",
  "features": [
    "max_severity",
    "mitre_tactic_count",
    "critical_mitre_tactic_present",
    "alert_count_log",
    "rule_diversity_shannon",
    "severity_dispersion",
    "agent_criticality"
  ]
}
```

Runtime harus fail-fast jika feature list/order tidak cocok.

## 3. RobustScaler Lifecycle

Training:

```text
X_train -> RobustScaler.fit -> X_scaled
```

Inference:

```text
X_live -> existing_scaler.transform
```

Dilarang `fit_transform` pada live/replay request.

## 4. Isolation Forest Lifecycle

Training configuration:

```text
n_estimators = 200
contamination = "auto"
random_state = fixed
```

Model fit hanya pada explicit training command.

Inference hanya:

```text
model.score_samples(X_scaled)
```

## 5. Score Calibration

Current legacy implementation menormalisasi score menggunakan min/max dari batch yang sedang diproses. Ini tidak valid untuk single-event live inference.

Training/calibration run harus menyimpan transform yang membuat score live comparable dengan score training.

Policy v1:

```text
raw_anomaly = -model.score_samples(X_scaled)
cal_min = min(raw_anomaly on calibration/training reference)
cal_max = max(raw_anomaly on calibration/training reference)
normalized = (raw_anomaly - cal_min) / (cal_max - cal_min)
```

Do not silently clip before threshold calculation unless explicitly specified. Untuk UI range, optional display clipping boleh dilakukan pada layer presentation, tetapi decision engine harus menggunakan scoring policy version yang sama dengan training.

Degenerate calibration (`cal_max == cal_min`) adalah artifact generation failure, bukan fallback live `0.5`.

`score_calibration.json` minimum:

```json
{
  "version": "minmax-v1",
  "raw_min": 0.31,
  "raw_max": 0.72,
  "higher_is_more_anomalous": true
}
```

Angka contoh bukan hasil penelitian.

## 6. Tukey Threshold

Threshold dihitung pada normalized scores reference run:

```text
Q1
Q3
IQR = Q3-Q1
theta = Q3 + 1.5*IQR
```

Tidak di-clamp ke 1.0.

`threshold.json`:

```json
{
  "method": "tukey_iqr",
  "q1": 0.0,
  "q3": 0.0,
  "iqr": 0.0,
  "threshold": 0.0
}
```

Semua nilai harus berasal dari run aktual; contoh schema tidak boleh dicopy sebagai hasil.

## 7. `metadata.json`

Minimum:

```text
model_version
created_at_utc
training_run_id
git_commit
feature_schema_version
research_config_hash
python_version
sklearn_version
n_estimators
contamination
random_state
training_row_count
meta_alert_count
training_period_start
training_period_end
score_calibration_version
```

Jika temporal holdout dipakai, metadata juga menyimpan cutoff.

## 8. Atomic Artifact Publication

Training jangan menulis langsung ke directory artifact aktif.

Pattern:

```text
artifacts/models/.staging/<run_id>/
-> write all files
-> validate roundtrip
-> checksum
-> atomic rename/publish to version directory
```

Artifact set yang incomplete tidak boleh menjadi active model.

## 9. Runtime Model Registry

Runtime start:

```text
load metadata
validate feature schema
load scaler
load model
load calibration
load threshold
run smoke prediction on fixture if configured
mark ready
```

Jika salah satu file hilang/incompatible:

```text
/health may be alive
/ready = false
```

## 10. Active Version

Jangan overwrite model artifact lama.

Gunakan explicit pointer/config:

```text
RBTA_MODEL_VERSION=rbta-if-...
```

Rollback = memilih versi lama yang masih valid, bukan restore file random.

## 11. Replay/Live Restrictions

Replay/live mode dilarang:

```text
fit scaler
fit IF
change contamination
recompute calibration
recompute Tukey threshold per request
change feature order
```

Jika model version berganti saat runtime, lakukan controlled reload dengan audit event.

## 12. Decision Trace

Setiap scored meta-alert menyimpan:

```text
model_version
feature_schema_version
score_calibration_version
raw_model_score
anomaly_score
threshold_used
decision
action
```

Ini memungkinkan hasil dapat ditelusuri ke artifact yang tepat.

## 13. Artifact Integrity

Recommended manifest checksum:

```text
SHA256 setiap file artifact
```

Runtime boleh memverifikasi checksum sebelum ready.

Model files tidak boleh diterima dari untrusted upload tanpa control.

## 14. Tests

Mandatory:

```text
feature schema exact order
train artifact writes all required files
artifact load roundtrip gives same score
single meta-alert inference uses stored calibration
single meta-alert inference is not forced to 0.5
threshold not clamped
missing artifact -> ready false
wrong schema -> fail-fast
wrong sklearn compatibility -> clear error/warning policy
model version appears in scored output
```

## 15. Acceptance Gate

Artifact lifecycle lulus bila:

```text
same artifact + same feature vector -> same score
batch inference and stream inference equal
stream run performs zero fit operations
threshold is immutable during one run
artifact bundle complete and auditable
rollback to previous version possible
```