# Isolation Forest Specification

## Peran Model

Isolation Forest bukan detector serangan independen. Wazuh sudah melakukan detection dan menghasilkan alert. Isolation Forest digunakan untuk **anomaly scoring dan prioritisasi meta-alert** hasil RBTA.

Model menjawab:

```text
Seberapa tidak lazim meta-alert ini dibanding distribusi meta-alert lain?
```

Bukan:

```text
Apakah event ini pasti serangan?
```

## Input

Input model adalah matriks 7 fitur dari `03-FEATURE-ENGINEERING-SPEC.md`.

```text
shape = [n_meta_alerts, 7]
```

## Scaling

Gunakan:

```text
RobustScaler
```

Alasan metodologis: feature SIEM dapat memiliki distribusi skewed dan outlier besar.

Training:

```text
scaler.fit(X_reference)
X_scaled = scaler.transform(X_reference)
```

Replay/live:

```text
X_scaled = persisted_scaler.transform(X_new)
```

Replay/live tidak boleh melakukan `fit` atau `fit_transform`.

## Model Configuration

Konfigurasi utama dikunci:

```text
IsolationForest
n_estimators = 200
contamination = "auto"
random_state = fixed/reproducible
n_jobs = implementation choice
```

### Ground Truth Prohibition

Dilarang menghitung contamination dari:

```text
ground_truth
is_synthetic
scenario_id
label serangan apa pun
```

Hapus `compute_dynamic_contamination()` dan jalur konfigurasi sejenis dari primary pipeline.

## Raw Anomaly Score

Gunakan score dari model dan ubah orientasi secara konsisten sehingga **nilai lebih tinggi berarti lebih anomali**.

Contoh policy:

```text
raw_anomaly = -IsolationForest.score_samples(X_scaled)
```

Policy harus versioned dan digunakan sama pada batch serta stream.

## Stream-Safe Score Calibration

Legacy implementation yang menghitung min/max dari batch yang sedang diprediksi **tidak boleh digunakan** untuk replay/live.

Masalah:

```text
1 meta-alert live
-> min(raw_score) == max(raw_score)
-> normalized score degenerates
```

Calibration harus dibuat pada explicit training/reference run dan disimpan sebagai model artifact.

Policy v1 yang direkomendasikan:

```text
cal_min = min(raw_anomaly pada reference run)
cal_max = max(raw_anomaly pada reference run)

anomaly_score = (raw_anomaly - cal_min) / (cal_max - cal_min)
```

Jika `cal_max == cal_min`, artifact generation gagal; jangan fallback live ke score tetap.

Detail serialization ada pada `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`.

## Tukey IQR Threshold

Threshold dihitung terhadap score calibration/reference yang sama:

```text
Q1 = percentile 25
Q3 = percentile 75
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
```

Jangan memaksa threshold menjadi `<=1.0` hanya karena score reference dinormalisasi.

Jika threshold > maximum observed score, maka secara statistik memang tidak ada upper-outlier berdasarkan Tukey fence pada run tersebut.

Replay/live menggunakan persisted threshold milik model version aktif. Threshold tidak dihitung ulang per request atau per meta-alert.

## Decision Matrix

Kombinasikan:

```text
anomaly_high = anomaly_score >= threshold
severity_high = max_severity >= 7
```

Keputusan:

```text
anomaly high + severity high -> CRITICAL -> ESCALATE
anomaly high + severity low  -> SUSPICIOUS -> ESCALATE
anomaly low  + severity high -> NOISE_HIGH -> DAILY_DIGEST
anomaly low  + severity low  -> NOISE -> SUPPRESS
```

## False Positive Gate

Laporan seminar mencantumkan gate sebelum keputusan akhir untuk meta-alert di atas threshold:

```text
max_severity < 7
AND alert_count < 5
AND mitre_hit_count = 0
```

Dengan canonical seven-feature/context specification, implementasi memakai semantik ekuivalen:

```text
max_severity < 7
AND alert_count < 5
AND mitre_tactic_count == 0
```

Jika ketiga kondisi terpenuhi:

```text
CONTEXTUAL_ANOMALY -> SUPPRESS
```

Gate tidak boleh menggunakan IP/cross-agent feature lama.

## Output Contract

Setiap scored meta-alert minimum mempunyai:

```text
meta_id
raw_model_score
anomaly_score
threshold_used
decision
action
escalate
model_version
feature_schema_version
score_calibration_version
```

Field version/threshold penting untuk REST/SOAR audit trail.

## No Hardcoded Result

Tidak boleh menulis angka tetap seperti:

```text
5% anomaly
AUC tertentu
jumlah escalate tertentu
threshold tertentu
```

ke report atau log sebagai hasil penelitian.

Semua nilai harus dihitung dari run aktual/artifact yang aktif.

## Serialization untuk Operational Mode

Research training harus dapat menyimpan artifact:

```text
RobustScaler
IsolationForest
score calibration
Tukey threshold
feature schema version
model metadata
training timestamp
```

Replay/live hanya melakukan load dan validation.

Artifact contract lengkap ada di `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`.

## Unit/Integration Tests Minimum

```text
model config = 200 trees + contamination auto
no ground-truth parameterization
same artifact + same vector -> same raw score
same artifact + same vector -> same anomaly score
single-event inference uses stored calibration
single-event inference does not collapse to fixed 0.5
Tukey threshold not clamped to 1.0
FP gate uses mitre_tactic_count
batch and stream scoring equal with same artifact
missing/incompatible artifact fails readiness
```