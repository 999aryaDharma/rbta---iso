# Evaluation Specification

## Tujuan

Evaluasi dibagi menjadi dua kelompok: evaluasi RBTA dan evaluasi struktural Isolation Forest. Synthetic attack ground-truth evaluation tidak termasuk pipeline utama.

## Phase A — RBTA Evaluation

### A1. Sensitivity Analysis Delta-t

Uji nilai:

```text
1, 5, 10, 15, 20, 30, 45, 60 menit
```

Pada fase ini adaptive window **dimatikan** agar perbandingan delta-t statis bersifat ceteris paribus.

Output minimum:

```text
delta_t_min
n_raw
n_meta
ARR
execution_time_ms
```

Titik elbow boleh dihitung otomatis tetapi algoritma dan hasilnya harus dilaporkan, bukan hardcoded.

### A2. Fixed Window Baseline

Source of truth mengikuti laporan seminar: baseline adalah **fixed tumbling time-window** yang membagi event berdasarkan interval waktu kalender dan tidak menggunakan contextual grouping `(agent_id, rule_group_primary)` milik RBTA.

Tujuannya adalah menjadi pembanding statis terhadap RBTA context-aware + adaptive.

Output baseline harus mempertahankan schema meta-alert yang cukup kompatibel untuk perbandingan ARR, tetapi implementasi tidak boleh diam-diam menambahkan contextual key RBTA ke baseline.

### A3. Alert Reduction Rate

```text
ARR = (N_raw - N_meta) / N_raw * 100%
```

`N_raw` adalah jumlah alert valid setelah preprocessing. `N_meta` adalah jumlah output meta-alert.

ARR dihitung terpisah untuk Fixed Window dan RBTA bila keduanya dibandingkan.

### A4. Noise Robustness

Noise rate:

```text
0%, 5%, 10%, 20%, 30%
```

Noise merupakan false-positive-like alerts:

- timestamp didistribusikan pada rentang data;
- pasangan `(agent_id, agent_name)` harus valid dan di-sample sebagai pasangan, bukan independen;
- rule_group dari populasi dataset;
- severity rendah 1–4;
- tanpa sinyal MITRE.

Output minimum:

```text
noise_rate
n_noise
n_total
n_meta
ARR
ARR_degradation
noise_absorption_count
noise_absorption_rate
execution_time_ms
```

Noise injection ini berbeda dari synthetic attack injection. Ia hanya digunakan untuk menguji ketahanan agregasi.

### A5. Runtime Complexity

Jalankan RBTA pada delapan ukuran subset yang meningkat menuju 100% dataset.

Untuk setiap subset catat:

```text
n_alerts
n_meta
execution_time_ms
throughput_alerts_per_ms
```

Fit regresi linear pada `n_alerts -> execution_time_ms` dan laporkan nilai aktual:

```text
slope
intercept
R_squared
mean_throughput
throughput_variation
```

Tidak boleh menulis R² atau klaim scalability sebelum eksperimen berjalan.

## Phase B — Isolation Forest Structural Evaluation

### B1. Build Observed Partition

Gunakan:

```text
X_scaled = RobustScaler(7 features)
```

Lalu hasil decision pipeline dipetakan menjadi partisi biner evaluasi:

```text
ESCALATE = 1
non-ESCALATE = 0
```

`DAILY_DIGEST` masuk non-escalate agar partisi evaluasi tetap biner dan konsisten.

### B2. Observed Silhouette Score

Hitung:

```text
silhouette_score(X_scaled, observed_partition)
```

Jika hanya ada satu class pada partition, Silhouette tidak valid. Report harus menyatakan evaluasi tidak dapat dihitung, bukan membuat nilai fallback palsu.

### B3. Permutation Baseline

Jumlah iterasi:

```text
n_permutations = 100
```

Setiap random partition harus:

- mempunyai jumlah sample sama;
- mempertahankan jumlah label 1 dan 0 sama dengan observed partition;
- hanya mengacak assignment label;
- menggunakan seed yang reproducible.

Hitung Silhouette untuk setiap random partition yang valid.

### B4. Structural Comparison

Report minimum:

```text
observed_silhouette
random_mean
random_std
random_min
random_max
observed_percentile
z_score
empirical_p_value
n_valid_permutations
random_seed
```

Empirical p-value dapat dihitung sebagai proporsi null score yang >= observed score dengan finite-sample correction yang didokumentasikan.

### B5. Interpretation Boundary

Kesimpulan yang diperbolehkan:

```text
Partisi prioritas yang dihasilkan menunjukkan / tidak menunjukkan pemisahan struktural yang lebih kuat daripada random partition berproporsi sama.
```

Kesimpulan yang tidak boleh dibuat hanya dari evaluasi ini:

```text
Isolation Forest memiliki accuracy X%
model pasti mendeteksi attack dengan benar
false negative serangan produksi adalah Y%
```

## Removed Primary Evaluation

Hapus dari pipeline utama:

```text
synthetic attack scenario A/B/C
PR-AUC
F1/F0.5
FNR berbasis synthetic label
FPR-vs-reduction berbasis injected ground truth
contamination tuning dari label
```

Jika artefact lama dipertahankan untuk histori, harus berada di area archive/legacy dan tidak dipanggil `main` research pipeline.

## Phase Order Final

```text
1. Data preparation
2. Sensitivity analysis (adaptive OFF)
3. Select/report base delta-t
4. Final RBTA run (adaptive PER-AGENT ON)
5. Fixed Window baseline run (time-only, sesuai laporan)
6. ARR analysis
7. Noise robustness
8. Runtime proof
9. Seven-feature extraction
10. RobustScaler + Isolation Forest
11. Tukey IQR + Decision Matrix
12. Observed Silhouette
13. 100 permutation baseline
14. Final research report
```

Evaluasi tidak boleh mengubah model yang sedang dievaluasi.