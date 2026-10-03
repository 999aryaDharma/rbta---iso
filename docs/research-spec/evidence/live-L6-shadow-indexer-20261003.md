# Live L6 shadow run — indexer source, real campus data — 2026-10-03

Status: **SHADOW OK (dry-run). Deployment tetap non-prod sampai Telegram asli
+ rotasi password reader tercatat.**

Branch: `prod/final-dashboard-demo`. Sumber live: `RBTA_LIVE_SOURCE=indexer`
terhadap `https://wazuh.indexer:9200` (hosts: `172.16.83.207 wazuh.indexer`).
State terisolasi (`shadow-state.json`, `shadow-evidence.sqlite3`); data demo
tidak tersentuh. Telegram `dry_run` (tidak ada pesan real terkirim).

## Transport yang terbukti

- TCP 9200 reachable dari dev; TLS dengan pin leaf `wazuh-indexer-leaf.pem`
  (CN=wazuh.indexer, SAN hanya DNS — wajib akses via nama, bukan IP).
- CA Wazuh (`api-ca.pem`) DITOLAK OpenSSL 3.x sebagai trust anchor
  (`CA cert does not include key usage extension`) — cacat penerbitan lama,
  bukan salah config. Perbaikan permanen di sisi server (terbitkan ulang
  dengan keyUsage). Utang dicatat, bukan disembunyikan.
- Auth reader (`arya`): search OK (1899 alert/2 hari terkanonikalisasi,
  id asli mis. `1790901144.0`), `_cat/indices` 403 → role
  `wazuh_alerts_reader` ditambah `cluster_monitor` (+ `indices_monitor`
  bila sapuan pertama masih 403) via security API oleh admin. Tanpa ini
  worker fail-closed sebelum memproses apa pun (benar by design).
- `.env` kampus memakai alias `WAZUH_INDEXER_CA_BUNDLE`; client kini
  menerimanya (fallback, `CA_PATH` menang) — 2 test alias.

## Observasi shadow run (`GET /api/v1/live/status`)

```json
{
  "worker_alive": true, "cycles_completed": 5, "consecutive_failures": 0,
  "last_error": null, "live_model_version": "rbta-if-v1",
  "lag_sec": 0.69, "buffer_size": null, "outbox_pending": 0,
  "quarantine_total": 0,
  "dispatcher": {"sent_total": 0, "failed_total": 0,
    "suppressed_historical_total": 67, "dry_run_total": 5, "mode": "dry_run"},
  "event_lag_sec": 4224.0, "tls_verify": true
}
```

Artinya: worker stabil, model beku terpin, 67 backlog historis
tersuppress (anti banjir, M1), 5 item baru dry-run, 0 korup, 0 antrean.
`event_lag` 70 menit = agen sedang sepi (19:07 vs 20:18), bukan macet.

## Gate regresi saat evidence ditulis

`tests/unit/runtime + tests/unit/ingestion +
test_direct_ingress_durability + tests/integration/runtime +
tests/integration/runners + test_app_endpoints`: **291 passed**.
Frontend: lint + typecheck + **62 passed/20 files** + build OK.

## Batas klaim

- Ini bukti pipeline hidup + reduksi triase berjalan, BUKAN bukti deteksi
  serangan dan bukan izin produksi.
- Belum dilakukan: kirim Telegram asli, rotasi password reader yang sempat
  tersingkap di log sesi, CA permanen sisi server, Runbook VPS.
