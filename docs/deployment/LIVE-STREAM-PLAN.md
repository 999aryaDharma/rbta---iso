# Live stream plan — L2–L6 (menuju live mode lengkap)

Status: **RENCANA — belum diimplementasi**. Fondasi L1 selesai
(`16ab574` + follow-up fixes, lihat `LIVE-FOUNDATION-HANDOFF.md`).
Deployment kampus tetap **BLOCKED_EXTERNAL** sampai L6 tuntas.

## Keputusan desain yang terkunci (2026-09-29, pilihan peneliti)

1. **Worker = thread di dalam proses FastAPI** (lifespan startup/shutdown).
   Tanpa proses terpisah, tanpa IPC. Eksklusi cukup dengan RLock yang ada;
   single-writer antar-proses menjadi non-goal dan diganti batasan
   "satu proses backend".
2. **Buffer sebagai ruang tunggu pengurut.** Poller mengambil per halaman
   (urutan kedatangan tak menjamin urutan event-time lintas index/shard);
   buffer menahan sebentar lalu melepas terurut `@timestamp` + `id`.
   Alert yang telat **tetap diproses (ditandai late), tidak dibuang** —
   sesuai locked rule "no valid alert may be dropped merely for arriving
   out of order".
3. **Notifikasi hanya Telegram.** Memakai pola `escalation_sink.emit`
   yang sudah ada; tidak ada kanal lain.

## Batasan locked yang tidak boleh dilanggar tiap fase

RBTA bucket key `(agent_id, rule_group_primary)`; elastic window lokal per
agent; 7 fitur terkunci; Isolation Forest unsupervised + frozen untuk
replay/eval; Tukey + decision matrix di luar model; ARR = reduksi unit
triase (bukan akurasi); tanpa synthetic attack ground truth; tanpa label
palsy; `contamination="auto"` tidak diklaim sebagai estimasi prevalensi
serangan. Replay dan artefak model tidak berubah.

Sistem ini mengurangi dan memprioritaskan unit triase. Alert bukan otomatis
serangan, anomaly score bukan label serangan, dan setiap MetaAlert harus
tetap dapat ditelusuri ke bukti mentahnya.

---

## L2 — Worker thread + lifecycle — SELESAI (working tree, belum komit)

**Tujuan:** siklus `LiveIngestionCoordinator.run_cycle` berjalan kontinu di
dalam proses backend.

- Hook ke lifespan `src/api/server.py` (startup: start sekali dengan guard
  anti double-start untuk reload/dev; shutdown: stop graceful, opsi drain
  buffer → RBTA → persist).
- Start/stop idempotent; interval fast poll / reconciliation konfigurabel
  via env; cycle gagal = fail-closed (cursor tidak maju, error terakhir
  tercatat tanpa body respons).
- Model dipin saat worker start dan tercatat di provenance sesi live;
  ganti model hanya via restart eksplisit.
- **Test (TDD):** start ganda tetap satu thread; shutdown drain
  memfinalisasi bucket aktif; crash sebelum commit tidak duplikasi
  (ditopang dedup SQLite + seen IDs).
- **Done:** worker jalan kontinu di localhost; restart tanpa loss/duplikasi
  pada skenario uji.
- **Implementasi:** `src/runtime/live_worker.py` (`LiveWorker`: start idempotent
  sekali-thread, stop idempotent + drain, fail-cycle tercatat dan lanjut,
  pin `live_model_version` di source state, `status()` untuk API L3);
  wiring lifespan + `app.state.live_worker` di `src/api/server.py`;
  opt-in via `RBTA_LIVE_WORKER_ENABLED=true` (default off — replay/demo tak
  terpengaruh); interval `RBTA_LIVE_POLL_INTERVAL_SEC` (default 5.0), drain
  `RBTA_LIVE_WORKER_DRAIN_ON_STOP` (default true). Batasan satu proses
  didokumentasikan di modul. Test: `tests/unit/runtime/test_live_worker.py`
  (6 test). Gate: 141 passed (135 + 6 baru); bootstrap/observability 13 passed.

## L3 — Status API read-only + isolasi live vs replay — SELESAI (paralel subagent, terverifikasi independen)

**Tujuan:** operator memantau live tanpa akses tulis ke pipeline.

- `GET /api/v1/live/status` (auth sama): mode, worker hidup/mati, cursor
  terakhir, lag vs waktu Indexer, hasil poll terakhir, error terakhir
  (redacted), ukuran buffer, antrian outbox.
- Isolasi state: service/buffer/cursor live terpisah dari path replay dan
  evaluasi, agar replay tetap frozen dan metrik skripsi tidak terkontaminasi
  data live.
- **Test (TDD):** kontrak respons status; output replay identik sebelum dan
  sesudah worker hidup.
- **Done:** status terpantau; replay deterministic tidak berubah.
- **Implementasi:** `GET /api/v1/live/status` di `src/api/app.py` (auth sama,
  read-only murni — hanya `status()`, `get_live_source_state()`,
  `get_outbox()`; null-safe bila worker disabled). Test:
  `tests/unit/api/test_live_status.py` (5 test). Catatan: `lag_sec` tidak
  di-clamp terhadap skew jam — dokumentasikan saat dipakai operator.

## L4 — Buffer urut + streaming bounded + recovery paruh-cycle — SELESAI kecuali persistensi checkpoint (paralel subagent, terverifikasi independen)

**Tujuan:** input ke RBTA selalu terurut event-time dengan memori bounded
dan tahan restart. (Fase tersulit.)

- **Ruang tunggu:** rilis terurut saat watermark
  (`max observed event-time − hold_window`) terlampaui atau `max_hold`
  tercapai; alert lebih tua dari watermark = rilis langsung + cap `late`.
  Kapasitas `max_items`; saat penuh = backpressure (perlambat poll, catat
  metrik) — **jangan drop**, jangan silent.
- **Streaming:** konsumsi per halaman; `poll_full_reconciliation` hanya
  untuk bootstrap/shadow, bukan loop rutin (menggantikan materialisasi
  full-retention saat ini).
- **Recovery:** checkpoint = cursor Indexer + isi buffer belum rilis +
  snapshot RBTA; restart lanjut tepat dari sana (uji injeksi crash di tengah
  cycle; exactly-once effect via dedup yang ada).
- Keterkaitan kode: `run_cycle` saat ini mengurutkan hanya di dalam satu
  cycle (`live_coordinator.py:204-214`) — buffer menutup celah urutan
  **antar-cycle** (telat dari reconciliation vs fast poll).
- **Test (TDD):** urutan rilis benar walau kedatangan acak; telat tidak
  hilang; buffer penuh tidak drop; restart tengah-cycle tepat-sekali.
- **Done:** RBTA selalu menerima event terurut; memori bounded; tahan restart.
- **Implementasi:** `src/runtime/order_buffer.py` (watermark `hold_window` /
  `max_hold`, cap `late`, `max_items` + backpressure tanpa drop, dedup
  by-id, `to_checkpoint()`/`from_checkpoint()`); integrasi opsional
  default-off di `LiveIngestionCoordinator` (perilaku lama identik).
  Test: `tests/unit/runtime/test_order_buffer.py` (7 test).
  **Sisa jujur:** persistensi durable checkpoint buffer (gabung cursor +
  snapshot RBTA) belum ada — crash dengan buffer aktif mengandalkan re-poll
  + dedup; refused-backpressure di-ingest langsung (tradeoff no-drop vs
  ordering, metric-visible).

## L5 — Dispatcher Telegram + switcher dashboard — SELESAI + terintegrasi ke lifespan

**Tujuan:** analis menerima ESCALATE di Telegram; MetaAlert live tertelusur.

- Dispatcher (bagian worker loop atau thread sendiri): ambil outbox
  `action == "ESCALATE"`, kirim Telegram dengan retry/backoff +
  idempotensi (terkirim → tandai di SQLite; gagal → tetap di outbox,
  bukan hilang). Format pesan konsisten dengan format Telegram replay yang
  ada; decision vs action ditampilkan terpisah (tanpa label ganda).
- Dashboard: switcher konteks live/replay; drill-down MetaAlert live ke
  raw evidence memakai kontrak provenance yang sama; state loading, error,
  empty, partial-evidence ditangani.
- **Test (TDD):** gagal-lalu-sukses = satu pesan terkirim, outbox kosong;
  kontrak provenance live identik replay; frontend unit test untuk switcher.
- **Done:** ESCALATE sampai ke Telegram tepat-sekali; tiap MetaAlert live
  tertelusur ke bukti mentah.
- **Implementasi:** `src/runtime/telegram_dispatcher.py` (thread sendiri,
  retry/backoff, idempotensi `run:meta`, sukses → `commit_outbox`, gagal →
  tetap antre, tanpa kredensial = dry-run aman); `make_telegram_sender()`
  real Bot API `sendMessage` (raise saat `ok:false`/transport gagal, tanpa
  log kredensial); wiring lifespan di `src/api/server.py`
  (`app.state.telegram_dispatcher`, start hanya bila kredensial ada, stop
  sebelum worker); `dashboard/src/features/live/LiveReplaySwitcher.tsx` +
  test. Test: 9 backend + 4 Vitest. Kredensial via
  `RBTA_TELEGRAM_BOT_TOKEN`/`RBTA_TELEGRAM_CHAT_ID` — tidak pernah di-commit.

## L6 — Hardening kampus + shadow run — SEBAGIAN (sisi-kode); eksternal masih BLOCKED_EXTERNAL

**Tujuan:** bukti live-ready terbatas dari lingkungan nyata, lalu buka
`BLOCKED_EXTERNAL`.

- Verifikasi VPS: OS, reachability Indexer, user/RBAC, mapping index,
  CA TLS, kapasitas CPU/memori vs ukuran buffer + throughput.
- Shadow run: worker baca data real, state terpisah, **tanpa kirim Telegram
  real** (dry-run/file sink) — ukur lag, keterurutan, memori, conflict rate.
- Klaim dibatasi pada yang terukur; tanpa klaim throughput/power-failure
  dari unit test.
- **Done:** `BLOCKED_EXTERNAL` dibuka hanya setelah akses terkonfirmasi +
  shadow run terotorisasi tercatat di evidence.
- **Sisi-kode selesai:** HTTP sender real + wiring kredensial eksplisit
  (tanpa kredensial = dry-run, tidak ada request jaringan). Gate terintegrasi
  L2–L5: **205 passed** (`tests/unit/runtime`, `tests/unit/ingestion`,
  `tests/unit/api`, `tests/integration/runtime`, `tests/integration/runners`).
- **Tetap membutuhkan peneliti (tak bisa dikerjakan agen):** OS VPS,
  reachability Indexer, user/RBAC, mapping index, CA TLS, kapasitas
  CPU/memori, kredensial bot Telegram, dan shadow run data-real terotorisasi
  (dry-run dulu). Sampai itu ada: **BLOCKED_EXTERNAL tetap berlaku** -
  live mode lengkap secara kode, belum live-ready secara deployment.

---

## Review F1–F13 — resolusi (2026-09-30, paralel 4 agen + integrasi)

Kebijakan quarantine-vs-halt (F2, diputuskan eksplisit): **QUARANTINE, bukan
halt**. Halt membuat SOC buta berhari-hari oleh satu dokumen cacat;
quarantine durable (`quarantined_alerts`: id, index, doc, tipe error, count)
+ terlihat di status (`quarantine_total`) — tidak silent. Error transien
tetap menggagalkan cycle (cursor tidak maju + backoff eksponensial+jitter
5 dtk–5 mnt). Error canonicalization level-halaman tetap fail-closed
(batas jujur: poller di luar quarantine per-hit).

| Temuan | Resolusi |
| --- | --- |
| F1 token di log | Pesan error hanya `TypeName [+ HTTP status]`, tanpa URL/body; rantai `from exc` terjaga. Rotasi tak perlu (hanya token palsu di test). |
| F3 drain default | Default false (konstruktor + env); drain hanya decommission eksplisit; pesan lifespan kondisional; test ekuivalensi restart. |
| F4 banjir bootstrap | `notify_max_age` (default 1 jam, env `RBTA_TELEGRAM_NOTIFY_MAX_AGE_SEC`): item basi di-suppress + counter + commit (drain backlog, tercatat). |
| F2 quarantine/backoff | Di atas. |
| F6 fingerprint vs config | Opsi murah: `compute_derivation_hash()` + guard start (override `RBTA_LIVE_DERIVATION_OVERRIDE`); fingerprint v3 ditunda. Integrator: guard sempat mati (probing di objek salah + hash tak disimpan) — diperbaiki TDD, kini refusal + baseline-pin teruji. |
| F5 klaim L4 | `buffer_size: null` bila disabled; aktivasi via `RBTA_ORDER_BUFFER_ENABLED`; done-criterion jujur: **terurut dalam hold_window, sisanya ditandai late** (bukan "selalu terurut"). |
| F7 semantik buffer | Clamp masa-depan (`future_tolerance`, counter `future_anomalies`); helper `partition_unseen`; hold_window = jitter < window, recon = expected-late (docstring). |
| F8 Telegram | Klaim jujur **at-least-once across restarts**; cabang duplikat tetap commit; baris Meta-ID; throttle ~1/detik; hormat Retry-After (cap 60 dtk). |
| F9 pin model | Tolak start bila pin beda (override `RBTA_LIVE_MODEL_OVERRIDE`/run baru); metadata tanpa versi → raise. |
| F10 poller | `allow_unavailable` param (default True); full-recon strict; helper `validate_index_id_uniqueness` untuk smoke test; asumsi id-unik didokumentasikan. |
| F11 telemetri | `dispatcher`, `quarantine_total`, `newest_scored_event_time`/`event_lag_sec`, redaksi host; `lag_sec` = umur cycle (docstring). |
| F12 satu proses | OS file lock `<state>.lock` (msvcrt/flock), gagal cepat bila terkunci. |
| F13 kecil | `RBTA_TELEGRAM_DRY_RUN`; source_mode LIVE saat worker (replay tak tersentuh); warning TLS-off + `tls_verify` di state; stop skip-drain bila thread hidup; flush-sebelum-clear cache; fingerprint `''` eksplisit; backup `.pre-1.2.bak`; restore fail-closed bila seen tak kosong. |
| M1–M6 re-review | Bootstrap-suppress aktif di produksi (`live_first_started_at` set-once + lazy read + fail-closed + wiring-test app); breaker karantina-baru + skip-ID-lama; thread selalu jalan + `notification_log` durable; re-pin derivasi + audit; telemetri (`COUNT(*)`, blok buffer, tls selalu, lag dari ingest); lock-test regresi diperbaiki via release. |

Sisa jujur: persistensi durable checkpoint buffer (re-poll + dedup sementara);
exactly-once Telegram ujung-ke-ujung belum diuji; `lag_sec` tak di-clamp.
Gate terintegrasi pasca-fix: **259 passed** (`unit/runtime+ingestion+api`,
`integration/runtime+runners`), nol gagal. Integrator: +`partition_unseen`
di-wire ke coordinator (F7b selesai penuh), guard derivasi dihidupkan,
wiring buffer-env + strict-discovery, minor telemetri
(`quarantine_count`, blok `buffer_stats`, `tls_verify` selalu tercatat,
`event_lag` dari ingest mentah), M1–M6 (bootstrap-suppress aktif produksi,
breaker karantina-baru, thread selalu jalan + notification_log, re-pin),
P1/P2 (latch breaker + ack, hold-vs-dryrun, endpoint release,
`notification_verdicts`, wiring `breaker_ack`),
gate akhir **357 passed, 0 failed**.
Full backend, frontend build, E2E tidak diulang (di luar area sentuh;
failure lama tetap berlaku).

---

## Urutan eksekusi dan verifikasi tiap fase

L2 → L3 → L4 → L5 → L6. Tiap fase: RED test → implementasi minimal →
gate suite (`tests/unit/runtime`, `tests/unit/ingestion`,
`test_direct_ingress_durability`, `tests/integration/runtime`,
`tests/integration/runners`, `test_app_endpoints`) → update evidence
(`docs/research-spec/evidence/`) + handoff ini. Full backend suite,
frontend checks, dan E2E dijalankan saat fase menyentuh areanya; browser
yang tak tersedia dilaporkan sebagai keterbatasan lingkungan, bukan
kegagalan aplikasi.
