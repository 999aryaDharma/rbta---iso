# RBTA + Isolation Forest — Research Source of Truth

## Status

Dokumen ini adalah aturan implementasi tertinggi untuk codebase penelitian **Rule-Based Temporal Aggregation dan Isolation Forest untuk Mitigasi Alert Fatigue pada Log Keamanan SIEM Wazuh**.

Tujuan dokumen ini adalah mencegah implementasi bergeser dari metodologi seminar proposal. Jika implementasi lama, komentar, docstring, eksperimen, atau konfigurasi bertentangan dengan dokumen penelitian ini, implementasi tersebut harus diubah atau dihapus.

## Hierarki Otoritas

Urutan otoritas yang wajib diikuti:

1. `Seminar fix.pdf` — metodologi penelitian utama.
2. Klarifikasi eksplisit peneliti setelah laporan: **Elastic Time Window/EMA dihitung lokal per agent, bukan global.**
3. Seluruh dokumen pada `docs/research-spec/`.
4. Implementasi source code terbaru.
5. Komentar, docstring, eksperimen, dan artifact lama.

Tidak boleh mengubah metodologi hanya agar cocok dengan code lama.

## Prinsip Penelitian yang Dikunci

### 1. RBTA adalah agregator utama

RBTA mengelompokkan alert Wazuh berdasarkan konteks:

```text
bucket_key = (agent_id, rule_group_primary)
```

Syarat sebuah alert dapat bergabung ke bucket aktif:

- key sama;
- gap antar-alert `<= delta_t` adaptif milik agent tersebut;
- total durasi bucket tidak melebihi 60 menit.

Output RBTA adalah `MetaAlert` yang mempertahankan informasi agregat dan traceability alert asal.

### 2. Elastic Time Window wajib lokal per agent

Tidak boleh ada satu state EMA global untuk seluruh stream.

Setiap `agent_id` wajib mempunyai state independen:

```text
AgentTemporalState
- last_timestamp
- warmup_event_count
- warmup_gaps
- baseline_gap
- ema_gap
- current_delta_t
```

Konsekuensi penting:

- traffic Agent A tidak boleh mengubah EMA Agent B;
- warm-up Agent A tidak boleh mengaktifkan adaptasi Agent B;
- baseline Agent A tidak boleh digunakan Agent B;
- `delta_t` yang dipakai bucket `(agent_id, rule_group)` berasal dari state temporal `agent_id` tersebut.

### 3. Formula Elastic Time Window

Parameter utama mengikuti laporan seminar:

```text
alpha = 0.10
warmup_event_count = 100 event pertama per agent
min_delta_t = 0.5 * base_delta_t
max_delta_t = 1.5 * base_delta_t
```

Selama 100 event pertama milik agent tersebut, inter-arrival gap yang tersedia dikumpulkan untuk membangun baseline lokal. Setelah fase warm-up selesai:

```text
EMA_gap[a,t] = alpha * gap[a,t] + (1-alpha) * EMA_gap[a,t-1]

delta_t[a,t] = base_delta_t * (EMA_gap[a,t] / baseline_gap[a])

delta_t[a,t] = clip(delta_t[a,t], 0.5*base_delta_t, 1.5*base_delta_t)
```

Implementasi step-based lama (`HIGH_FREQ`, `LOW_FREQ`, `SHRINK_RATE`, `EXPAND_RATE`, perubahan 0.8x/1.2x) tidak sesuai source of truth dan harus dihapus.

### 4. Out-of-order tidak boleh menyebabkan data loss

Penelitian tetap harus mampu menangani kedatangan alert tidak berurutan. Reorder buffer boleh digunakan, tetapi **tidak ada kebijakan `late_drop`** pada pipeline penelitian utama.

Alert valid yang berhasil diparsing harus akhirnya diproses. Tidak boleh ada alert dibuang hanya karena melewati watermark/lateness threshold yang tidak didefinisikan dalam metodologi seminar.

### 5. Hanya satu bucket RBTA

`Bucket B`, `CompoundMetaAlert`, compound window, dan behavioral sequencing lama dihapus.

Satu-satunya bucket penelitian utama adalah:

```text
(agent_id, rule_group_primary)
```

### 6. Feature vector Isolation Forest tepat tujuh fitur

Feature vector final:

1. `max_severity`
2. `mitre_tactic_count`
3. `critical_mitre_tactic_present`
4. `alert_count_log`
5. `rule_diversity_shannon`
6. `severity_dispersion`
7. `agent_criticality`

Tidak boleh menggunakan feature vector 9/11/12/13 fitur lama sebagai model penelitian utama.

### 7. Isolation Forest sepenuhnya unsupervised

Konfigurasi penelitian:

```text
n_estimators = 200
contamination = "auto"
RobustScaler = enabled
threshold = Tukey IQR
```

Ground truth tidak boleh dipakai untuk menentukan contamination, feature, threshold, atau parameter model.

### 8. Synthetic attack injection dihapus dari pipeline utama

Yang harus dihapus:

- skenario synthetic attack A/B/C;
- `is_synthetic` sebagai label model;
- `ground_truth` propagation;
- dynamic contamination dari ground truth;
- PR-AUC/F1/F0.5/FNR sebagai evaluasi utama penelitian.

**Catatan:** noise injection untuk uji robustness RBTA tetap dipertahankan karena merupakan bagian metodologi evaluasi. Noise injection bukan synthetic attack ground-truth evaluation.

### 9. Evaluasi Isolation Forest adalah validitas struktural

Evaluasi utama IF:

```text
7-feature matrix -> RobustScaler -> IF -> IQR/Decision Matrix
                  -> observed binary partition
                  -> Silhouette Score
                  -> 100 random partitions dengan proporsi sama
                  -> null distribution
                  -> percentile / z-score / empirical p-value
```

Synthetic ground truth bukan dasar kesimpulan utama.

### 10. Tidak boleh ada hasil eksperimen hardcoded

Nilai seperti ARR, R², throughput, Silhouette, noise absorption, threshold, jumlah meta-alert, atau status "lulus/production-ready" harus dihitung dari run aktual.

Report generator hanya boleh membaca hasil yang benar-benar dihasilkan eksperimen.

## Komponen yang Dihapus dari Desain Lama

```text
REMOVE FROM PRIMARY PIPELINE
- Synthetic attack scenario A/B/C
- Ground-truth propagation
- Dynamic contamination from labels
- Bucket B / CompoundMetaAlert
- rule_group_entropy lama
- tactic_progression_score lama
- cross_agent_spread lama
- alert_velocity sebagai feature IF utama
- deviation_from_baseline sebagai feature IF utama
- hour_of_day sebagai feature IF utama
- rule_group_severity_enc sebagai feature IF utama
- Watermark late_drop
- Hardcoded evaluation results
```

## Operational Extension

REST API, integrasi Wazuh live stream, Shuffle SOAR, dan Telegram adalah **lapisan operasional**. Lapisan ini boleh ditambahkan selama tidak mengubah algoritma penelitian di atas.

Aturan boundary:

```text
Research Core
ETL -> RBTA -> 7 Features -> IF -> Decision

Operational Adapters
Wazuh REST ingress -> Research Core -> Shuffle SOAR -> Telegram
```

Transport, webhook, API, atau notifikasi tidak boleh memengaruhi feature, EMA, threshold, atau keputusan metodologis.

Lihat `08-OPERATIONAL-INTEGRATION-SPEC.md` untuk kontrak integrasi.