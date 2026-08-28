# Implementation Checklist

Dokumen ini adalah high-level checklist eksekusi refactor. Agent coding wajib menyelesaikan fase secara berurutan dan tidak menandai selesai tanpa test/evidence. Untuk task-level implementation steps gunakan `13-ANTIGRAVITY-SPRINT-PLAN.md`; completion rules ada di `14-AGENT-DEFINITION-OF-DONE.md`.

## Phase 0 — Protect Current State

- [ ] buat branch implementasi terpisah
- [ ] simpan baseline output penting untuk perbandingan
- [ ] identifikasi entry point aktif dan dependency antar module
- [ ] pastikan docs `00-SOURCE-OF-TRUTH.md` dibaca sebelum perubahan
- [ ] project manifest + pytest harness tersedia
- [ ] tidak ada credential/secrets di repo/fixtures

## Phase 1 — Remove Obsolete Research Paths

- [ ] synthetic attack scenario A/B/C tidak dipanggil primary pipeline
- [ ] `ground_truth` propagation dihapus dari primary pipeline
- [ ] dynamic contamination berbasis label dihapus
- [ ] Bucket B / `CompoundMetaAlert` dihapus
- [ ] output compound bucket dihapus
- [ ] watermark `late_drop` dihapus
- [ ] step ETW `HIGH_FREQ/LOW_FREQ/SHRINK_RATE/EXPAND_RATE` dihapus
- [ ] feature vector 9/11/12/13 lama dihapus
- [ ] report hardcoded dihapus
- [ ] stale fields pada SOAR payload dihapus

## Phase 2 — Canonical Data Contract

- [ ] satu `CanonicalRawAlert` schema untuk batch/API/archive/live
- [ ] `wazuh_alert_id` wajib dan idempotent
- [ ] OpenSearch `_id` disimpan sebagai source metadata, bukan idempotency key utama
- [ ] MITRE tactics dipertahankan sebagai daftar/set
- [ ] parser support `rule.mitre.*` dan `rule.mitre_tactics/techniques`
- [ ] `rule_group_primary` berasal dari config tunggal
- [ ] agent criticality berasal dari config tunggal
- [ ] parser file dan parser API menghasilkan canonical representation setara
- [ ] tidak ada hardcoded workstation path pada production path

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
- [ ] `sum(alert_count) == processed_unique_raw_alert_count`

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
- [ ] missing required feature fail-fast; no silent production zero-fill

## Phase 6 — Isolation Forest + Artifact Lifecycle

- [ ] RobustScaler
- [ ] IsolationForest `n_estimators=200`
- [ ] `contamination="auto"`
- [ ] fixed random state
- [ ] tidak ada penggunaan ground truth
- [ ] persistent score calibration dibuat saat training/reference run
- [ ] single-event inference memakai stored calibration
- [ ] single-event inference tidak fallback menjadi 0.5 karena request min=max
- [ ] Tukey IQR tanpa clamp threshold ke 1.0
- [ ] Decision Matrix 4 kuadran
- [ ] False Positive Gate memakai `mitre_tactic_count == 0`
- [ ] output menyimpan model/schema/calibration version
- [ ] artifact bundle atomic + validated
- [ ] replay/live melakukan zero fit operations

## Phase 7 — Historical Wazuh Ingestion

- [ ] discover daily indices; missing date valid
- [ ] PIT dibuat per daily index, bukan wildcard multi-month
- [ ] partial PIT ditolak
- [ ] stable sort `@timestamp ASC, id ASC`
- [ ] page fetch default 500; yield event satu-per-satu
- [ ] `search_after` memakai exact previous sort
- [ ] PIT ditutup pada `finally`
- [ ] checkpoint simpan index + last_sort + count
- [ ] restart membuat PIT baru dan resume checkpoint
- [ ] dedup mencegah duplicate state mutation
- [ ] credential tidak hardcoded/logged

## Phase 8 — Dual Mode Runtime

- [ ] `research-batch` dan `replay-stream` memakai shared `RBTAEngine`
- [ ] archived source dan Wazuh source menghasilkan canonical contract sama
- [ ] replay speed 1x/10x/100x/MAX
- [ ] replay sleep tidak mengubah event timestamp research
- [ ] finite replay `drain()` di EOF
- [ ] same canonical fixture: BatchRunner == MAX StreamRunner research output
- [ ] `run_id` dicatat

## Phase 9 — RBTA Evaluation

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

## Phase 10 — IF Structural Evaluation

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

## Phase 11 — Live Wazuh + Runtime Persistence

- [ ] live source dapat berupa campus collector push atau approved private route
- [ ] Wazuh Indexer port 9200 tidak perlu dipublish ke Internet
- [ ] live Indexer polling tidak memakai long-lived PIT snapshot
- [ ] polling overlap configurable; baseline recommendation 5 menit
- [ ] duplicate overlap dibuang sebelum RBTA mutation
- [ ] late-indexed event dalam overlap tetap diproses
- [ ] daily index rollover teruji
- [ ] temporal state per agent persisten/recoverable
- [ ] active RBTA buckets persisten/recoverable
- [ ] source checkpoint persisten
- [ ] restart mempertahankan continuity state
- [ ] idle flush tersedia

## Phase 12 — Operational API Readiness

- [ ] endpoint menerima Wazuh alert payload dari authorized collector
- [ ] duplicate `wazuh_alert_id` tidak diproses dua kali
- [ ] model/scaler/calibration/threshold/version dimuat saat service start
- [ ] scored meta-alert dapat disimpan/dikirim ke outbox
- [ ] delivery retry tidak menggandakan logical downstream event
- [ ] `/health`, `/ready`, `/runtime/stats` tersedia
- [ ] structured logging dan correlation/run ID tersedia

## Phase 13 — Shuffle + Telegram Integration

- [ ] Shuffle menerima normalized scored meta-alert
- [ ] custom Shuffle app memanggil REST contract; tidak menghitung research logic
- [ ] action routing mengikuti `ESCALATE`, `DAILY_DIGEST`, `SUPPRESS`
- [ ] CRITICAL/SUSPICIOUS dapat diteruskan ke Telegram
- [ ] Telegram formatting berada di workflow/adaptor, bukan research model
- [ ] suppressed alert tidak dikirim sebagai immediate notification
- [ ] retry/idempotency downstream teruji

## Phase 14 — ASUS Deployment

- [ ] Dockerfile production non-root
- [ ] config/env external
- [ ] persistent state/artifact volumes
- [ ] healthcheck + readiness
- [ ] graceful shutdown/restart recovery
- [ ] no public Wazuh 9200 dependency
- [ ] secrets absent from image/repo
- [ ] CI runs relevant tests before deploy

## Phase 15 — Final Verification

- [ ] seluruh unit tests lulus
- [ ] integration tests batch lulus
- [ ] integration tests historical ingestion lulus
- [ ] integration tests live overlap/dedup lulus
- [ ] EMA isolation test lulus
- [ ] no-event-loss test lulus
- [ ] duplicate ingress test lulus
- [ ] artifact roundtrip test lulus
- [ ] batch-replay equivalence test lulus
- [ ] restart recovery test lulus
- [ ] research pipeline end-to-end lulus
- [ ] operational Wazuh-like payload -> RBTA -> IF -> outbox/Shuffle stub lulus
- [ ] grep/search tidak menemukan import primary ke synthetic/compound legacy
- [ ] dokumentasi dan code menyebut 7 fitur secara konsisten
- [ ] angka report berasal dari artifact eksperimen aktual
- [ ] final status tidak claim real-live verified jika agent/jalur kampus belum tersedia