# RBTA Algorithm Specification

## Tujuan

Mendefinisikan implementasi RBTA yang identik dengan metodologi seminar dan menghilangkan perilaku lama yang tidak lagi termasuk penelitian.

## Bucket Identity

Primary key:

```text
(agent_id, rule_group_primary)
```

Tidak ada Bucket B, compound bucket, cross-agent bucket, atau fixed behavioral bucket pada pipeline utama.

## MetaAlert State

Meta-alert minimum menyimpan:

```text
meta_id
agent_id
agent_name
rule_group_primary
start_time
end_time
alert_count
max_severity
rule_id_distribution
severity_distribution
mitre_tactics_unique
critical_mitre_present
agent_criticality
wazuh_alert_ids
```

`wazuh_alert_ids` wajib dipertahankan untuk audit/traceability.

## Agent-Local Temporal State

### Hard Requirement

**EMA dan Elastic Time Window tidak global.**

Gunakan state map secara konseptual:

```text
temporal_states[agent_id] -> AgentTemporalState
```

Setiap agent mempunyai:

```text
last_timestamp
warmup_event_count
warmup_gaps
baseline_gap
ema_gap
current_delta_t
is_warmed_up
```

Rule isolasi state:

```text
update(event(agent=A)) hanya boleh memodifikasi temporal_states[A]
```

Tidak boleh membaca EMA/baseline agent lain untuk menghitung `delta_t` Agent A.

## Gap Calculation

Untuk alert valid milik Agent A:

```text
gap_A = current_event_time_A - previous_event_time_A
```

Gap tidak dihitung dari event global sebelumnya.

Contoh:

```text
10:00:00 Agent A
10:00:01 Agent B
10:05:00 Agent A
```

Maka:

```text
gap_A = 300 detik
```

bukan 299 detik dan bukan 1 detik.

## Warm-up

Laporan seminar menetapkan fase pemanasan **100 event pertama**. Klarifikasi implementasinya: fase tersebut dihitung **per agent**.

Untuk setiap agent:

- event pertama membentuk `last_timestamp`;
- event-event berikutnya selama fase 100 event pertama menghasilkan inter-arrival gap lokal yang dikumpulkan ke `warmup_gaps`;
- sebelum warm-up selesai, `current_delta_t = base_delta_t`;
- setelah 100 event pertama agent selesai diproses, `baseline_gap` dihitung dari gap lokal yang tersedia pada fase tersebut dan `ema_gap` diinisialisasi dari baseline itu.

Agent lain tetap memiliki hitungan warm-up independen.

## EMA Formula

Setelah warm-up:

```text
ema_gap = alpha * current_gap + (1-alpha) * previous_ema_gap
alpha = 0.10
```

## Elastic Delta-t Formula

```text
ratio = ema_gap / baseline_gap
candidate = base_delta_t * ratio
current_delta_t = clip(candidate,
                       0.5 * base_delta_t,
                       1.5 * base_delta_t)
```

Interpretasi:

- traffic lebih rapat -> `ema_gap < baseline_gap` -> window menyusut;
- traffic lebih renggang -> `ema_gap > baseline_gap` -> window melebar.

### Yang Dilarang

Hapus model step-based lama:

```text
HIGH_FREQ
LOW_FREQ
SHRINK_RATE
EXPAND_RATE
current_dt *= 0.8
current_dt *= 1.2
```

## Bucket Merge Rule

Untuk incoming alert dengan key `K`:

### Jika tidak ada active bucket

Buat bucket baru.

### Jika bucket aktif ada

Hitung:

```text
gap = event_time - bucket.end_time
prospective_duration = event_time - bucket.start_time
```

Alert bergabung jika:

```text
gap <= temporal_states[agent_id].current_delta_t
AND
prospective_duration <= 60 minutes
```

Jika gagal, finalize bucket lama lalu buat bucket baru.

## Handling Earlier Out-of-Order Event

Jika setelah reorder masih terdapat event yang lebih awal daripada `bucket.start_time`, update boundary dengan aman:

```text
bucket.start_time = min(bucket.start_time, event_time)
bucket.end_time   = max(bucket.end_time, event_time)
```

Durasi tidak boleh negatif.

## Out-of-Order Buffer

Penelitian boleh menggunakan bounded min-heap/reorder buffer agar stream dengan arrival order tidak rapi dapat dikonsumsi secara temporal.

Persyaratan:

- setiap alert valid akhirnya diproses;
- buffer didrain saat stream selesai;
- tidak ada `late_drop`;
- tidak ada event loss berbasis watermark;
- jumlah raw alert valid harus dapat direkonsiliasi dengan total membership meta-alert.

## Mapping Integrity

Invariant wajib:

```text
sum(meta.alert_count) == number_of_processed_raw_alerts
```

serta setiap `wazuh_alert_id` valid muncul tepat satu kali pada seluruh meta-alert.

Acceptance:

```text
no duplicates
no missing membership
no negative duration
no cross-agent temporal state mutation
```

## Complexity Target

Dengan bounded reorder buffer berukuran `k`, biaya reorder dapat dipandang `O(n log k)` dan mendekati linear untuk `k` tetap.

Klaim runtime akhir harus berasal dari pengukuran empiris, bukan hardcoded.