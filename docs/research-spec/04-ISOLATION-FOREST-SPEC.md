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

## Anomaly Score

Gunakan score dari model dan konversikan secara konsisten sehingga **nilai lebih tinggi berarti lebih anomali**.

Jika dilakukan min-max normalization ke `[0,1]`, implementasinya harus:

- deterministik untuk input yang sama;
- terdokumentasi;
- tidak menggunakan label;
- menangani kasus seluruh raw score identik dengan aman.

## Tukey IQR Threshold

Hitung:

```text
Q1 = percentile 25
Q3 = percentile 75
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
```

Jangan memaksa threshold menjadi `<=1.0` hanya karena normalized score berada di `[0,1]`.

Jika threshold > maximum observed score, maka secara statistik memang tidak ada upper-outlier berdasarkan Tukey fence pada run tersebut.

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

Namun feature specification baru mempertahankan daftar taktik MITRE, sehingga implementasi operasional harus memakai semantik ekuivalen:

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
anomaly_score
threshold_used
decision
action
escalate
model_version
feature_schema_version
```

`threshold_used`, `model_version`, dan `feature_schema_version` penting untuk REST/SOAR audit trail.

## No Hardcoded Result

Tidak boleh menulis angka tetap seperti:

```text
5% anomaly
AUC tertentu
jumlah escalate tertentu
```

ke report atau log sebagai hasil penelitian.

Semua nilai harus dihitung dari run aktual.

## Serialization untuk Operational Mode

Research core harus dapat menyimpan artifact model yang diperlukan live scoring:

```text
RobustScaler artifact
IsolationForest artifact
feature schema version
model metadata
training timestamp
```

Threshold Tukey yang dipakai live harus memiliki policy eksplisit pada implementation plan. Minimal, operational layer harus menyimpan threshold/version yang aktif dan tidak menghitung ulang threshold secara diam-diam untuk setiap request tunggal.