# Feature Engineering Specification

## Tujuan

Feature engineering hanya mengubah `MetaAlert` menjadi matriks numerik 7 fitur untuk Isolation Forest. Modul ini tidak mengubah state RBTA dan tidak boleh menentukan label keputusan.

## Final Feature Vector

Urutan feature dikunci:

```text
F1 max_severity
F2 mitre_tactic_count
F3 critical_mitre_tactic_present
F4 alert_count_log
F5 rule_diversity_shannon
F6 severity_dispersion
F7 agent_criticality
```

## F1 — max_severity

```text
max(rule_level dalam bucket)
```

Range domain Wazuh yang divalidasi: 0–15.

## F2 — mitre_tactic_count

Jumlah taktik MITRE ATT&CK **unik** dalam meta-alert.

```text
len(set(mitre_tactics))
```

Bukan jumlah alert yang mempunyai MITRE flag.

## F3 — critical_mitre_tactic_present

Biner:

```text
1 jika minimal satu taktik meta-alert termasuk CRITICAL_MITRE_TACTICS
0 selainnya
```

Critical tactic set didefinisikan di domain config tunggal.

## F4 — alert_count_log

Gunakan transformasi log agar burst besar tidak mendominasi feature space:

```text
log1p(alert_count)
```

Raw `alert_count` tetap dipertahankan pada MetaAlert untuk audit.

## F5 — rule_diversity_shannon

Hitung Shannon entropy dari distribusi `rule_id` dalam bucket dan normalisasi agar dapat dibandingkan antar-bucket.

Untuk distribusi probabilitas `p_i`:

```text
H = -sum(p_i * ln(p_i))
```

Normalisasi untuk `k > 1`:

```text
H_norm = H / ln(k)
```

Jika hanya satu rule unik atau bucket singleton:

```text
H_norm = 0
```

Range target: `[0,1]`.

## F6 — severity_dispersion

Standard deviation severity seluruh alert dalam bucket.

```text
std(rule_level)
```

Untuk singleton:

```text
severity_dispersion = 0
```

Feature ini mengukur variasi severity, bukan urutan temporal eskalasi.

## F7 — agent_criticality

Ordinal domain score:

```text
1 Low
2 Medium
3 High
4 Critical
```

Mapping berasal dari config tunggal.

## Removed Features

Tidak masuk feature matrix penelitian utama:

```text
duration_sec
rule_group_severity_enc
hour_of_day
alert_velocity
mitre_hit_count
rule_concentration
severity_spread
deviation_from_baseline
unique_rules_triggered raw
rule_firedtimes
rule_group_entropy lama
tactic_progression_score
cross_agent_spread
attacker_count
```

Field tersebut boleh tetap ada sebagai metadata hanya jika benar-benar dibutuhkan untuk audit/operasional, tetapi tidak boleh diam-diam masuk ke `FEATURE_COLUMNS`.

## Feature Contract

Harus ada satu konstanta tunggal:

```text
FEATURE_COLUMNS = [7 field di atas]
```

Dilarang menduplikasi feature list di RBTA core, model, evaluation, dan plotting.

## Validation

Sebelum fitting:

- seluruh 7 field harus tersedia;
- seluruh field numerik dan finite;
- tidak boleh silently membuat feature yang hilang menjadi 0 pada production/research run;
- missing required feature harus menghasilkan error yang jelas.

Fallback silent hanya boleh digunakan dalam test fixture yang eksplisit.

## Scaling

Gunakan `RobustScaler` pada matriks 7 fitur sebelum Isolation Forest.

Scaler harus fit pada dataset yang sama yang digunakan pada eksperimen IF sesuai rancangan penelitian, dan hasil `X_scaled` dipakai juga untuk evaluasi Silhouette sehingga ruang evaluasi sama dengan ruang input model.

## Unit Tests Minimum

```text
singleton -> entropy=0, dispersion=0
repeated same rule -> entropy=0
balanced two rules -> entropy mendekati 1
unique MITRE tactics counted once
critical tactic flag correct
alert_count_log == log1p(count)
feature output exactly 7 columns in fixed order
```