# Implementation Checklist

Dokumen ini adalah checklist eksekusi refactor. Agent coding wajib menyelesaikan fase secara berurutan dan tidak menandai selesai tanpa test/evidence.

## Phase 0 — Protect Current State

- [ ] buat branch implementasi terpisah
- [ ] simpan baseline output penting untuk perbandingan
- [ ] identifikasi entry point aktif dan dependency antar module
- [ ] pastikan docs `00-SOURCE-OF-TRUTH.md` dibaca sebelum perubahan

## Phase 1 — Remove Obsolete Research Paths

- [ ] synthetic attack scenario A/B/C tidak dipanggil primary pipeline
- [ ] `ground_truth` propagation dihapus dari primary pipeline
- [ ] dynamic contamination berbasis label dihapus
- [ ] Bucket B / `CompoundMetaAlert` dihapus
- [ ] output compound bucket dihapus
- [ ] watermark `late_drop` dihapus
- [ ] feature vector 9/11/12/13 lama dihapus
- [ ] report hardcoded dihapus
- [ ] stale fields pada SOAR payload dihapus

## Phase 2 — Canonical Data Contract

- [ ] satu `RawAlert` schema untuk batch dan REST
- [ ] `wazuh_alert_id` wajib dan idempotent
- [ ] MITRE tactics dipertahankan sebagai daftar/set
- [ ] `rule_group_primary` berasal dari config tunggal
- [ ] agent criticality berasal dari config tunggal
- [ ] parser file dan parser API menghasilkan canonical representation setara

## Phase 3 — Agent-Local EMA / ETW

- [ ] state disimpan per `agent_id`
- [ ] `last_timestamp` per agent
- [ ] `warmup_event_count` per agent
- [ ] inter-arrival `warmup_gaps` dikumpulkan dari 100 event pertama agent
- [ ] warm-up selesai setelah 100 event pertama **per agent**
- [ ] `baseline_gap` per agent
- [ ] `ema_gap` per agent
- [ ] alpha = 0.10
- [ ] formula proportional `base_dt * EMA/baseline`
- [ ] clamp 0.5x–1.5x base delta-t
- [ ] tidak ada `HIGH_FREQ/LOW_FREQ/SHRINK_RATE/EXPAND_RATE`
- [ ] event Agent A tidak memodifikasi state Agent B

### Mandatory EMA Isolation Test

Input minimal harus membuktikan dua agent dengan traffic sangat berbeda:

```text
Agent A: frekuensi sangat tinggi
Agent B: frekuensi rendah
```

Acceptance:

```text
Delta-t A berubah hanya dari gap A
Delta-t B berubah hanya dari gap B
```

Jika event Agent A mengubah EMA/baseline/delta-t B, test gagal.

## Phase 4 — RBTA Core

- [ ] bucket key tepat `(agent_id, rule_group_primary)`
- [ ] satu jenis bucket
- [ ] merge memakai local agent delta-t
- [ ] max bucket duration 60 menit
- [ ] start/end time aman untuk out-of-order event
- [ ] reorder buffer drain seluruh event
- [ ] tidak ada event valid hilang karena lateness threshold
- [ ] `wazuh_alert_ids` terakumulasi
- [ ] mapping integrity lulus
- [ ] `sum(alert_count) == processed_raw_alert_count`

## Phase 5 — Seven Features

- [ ] F1 `max_severity`
- [ ] F2 `mitre_tactic_count`
- [ ] F3 `critical_mitre_tactic_present`
- [ ] F4 `alert_count_log`
- [ ] F5 `rule_diversity_shannon`
- [ ] F6 `severity_dispersion`
- [ ] F7 `agent_criticality`
- [ ] exactly 7 feature columns
- [ ] satu source `FEATURE_COLUMNS`
- [ ] singleton entropy = 0
- [ ] singleton dispersion = 0
- [ ] duplicate MITRE tactic tidak dihitung dua kali

## Phase 6 — Isolation Forest

- [ ] RobustScaler
- [ ] IsolationForest `n_estimators=200`
- [ ] `contamination="auto"`
- [ ] fixed random state
- [ ] tidak ada penggunaan ground truth
- [ ] Tukey IQR tanpa clamp threshold ke 1.0
- [ ] Decision Matrix 4 kuadran
- [ ] False Positive Gate memakai `mitre_tactic_count == 0`
- [ ] output menyimpan model/schema version

## Phase 7 — RBTA Evaluation

- [ ] sensitivity delta-t 1,5,10,15,20,30,45,60
- [ ] adaptive OFF saat sensitivity
- [ ] final RBTA adaptive PER-AGENT ON
- [ ] Fixed Window baseline mengikuti definisi laporan: fixed tumbling time-window tanpa contextual key RBTA
- [ ] ARR dihitung dari raw valid vs meta-alert
- [ ] noise rate 0,5,10,20,30%
- [ ] noise sampling mempertahankan valid `(agent_id, agent_name)` pair
- [ ] runtime 8 subset
- [ ] R² dihitung dari run aktual
- [ ] throughput dihitung dari run aktual
- [ ] tidak ada claim hardcoded

## Phase 8 — IF Structural Evaluation

- [ ] observed binary partition dibuat
- [ ] Silhouette menggunakan `X_scaled`
- [ ] one-class result ditangani sebagai invalid evaluation
- [ ] 100 permutation labels
- [ ] proporsi class dipertahankan
- [ ] random seed dicatat
- [ ] random mean/std/min/max dihitung
- [ ] percentile dihitung
- [ ] z-score dihitung
- [ ] empirical p-value dihitung

## Phase 9 — Operational API Readiness

- [ ] batch core dan live core memakai service yang sama
- [ ] endpoint menerima Wazuh alert payload
- [ ] duplicate `wazuh_alert_id` tidak diproses dua kali
- [ ] temporal state per agent persisten/recoverable
- [ ] active RBTA buckets persisten/recoverable atau lifecycle restart terdokumentasi
- [ ] model/scaler/version dapat dimuat saat service start
- [ ] scored meta-alert dapat dikirim ke Shuffle
- [ ] delivery retry tidak menggandakan alert pada downstream
- [ ] health/readiness endpoint tersedia
- [ ] structured logging dan correlation ID tersedia

## Phase 10 — Shuffle + Telegram Integration

- [ ] Shuffle menerima normalized scored meta-alert
- [ ] action routing mengikuti `ESCALATE`, `DAILY_DIGEST`, `SUPPRESS`
- [ ] CRITICAL/SUSPICIOUS dapat diteruskan ke Telegram
- [ ] Telegram formatting berada di workflow/adaptor, bukan research model
- [ ] suppressed alert tidak dikirim sebagai immediate notification
- [ ] retry/idempotency downstream teruji

## Phase 11 — Final Verification

- [ ] seluruh unit tests lulus
- [ ] integration tests batch lulus
- [ ] integration tests REST ingestion lulus
- [ ] EMA isolation test lulus
- [ ] no-event-loss test lulus
- [ ] duplicate ingress test lulus
- [ ] research pipeline end-to-end lulus
- [ ] operational pipeline Wazuh-like payload -> RBTA -> IF -> Shuffle stub lulus
- [ ] grep/search tidak menemukan import primary ke synthetic/compound legacy
- [ ] dokumentasi dan code menyebut 7 fitur secara konsisten
- [ ] angka report berasal dari artifact eksperimen aktual