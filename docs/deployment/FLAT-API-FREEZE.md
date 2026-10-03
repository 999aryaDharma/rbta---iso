# FLAT API FREEZE — custom intermediary `{events,total,limit,offset}`

Status: **BEKU 2026-10-03** — pilihan peneliti. Live production hit API
perantara custom ini, BUKAN Indexer `:9200/_search` langsung dan BUKAN
collector push. Bentuk flat di bawah tidak boleh berubah tanpa revisi
dokumen ini + fixture + adaptor.

Contoh nyata: `tests/fixtures/wazuh/flat_api_page.json` (2 dari 306 events).

## Batasan locked (tak tersentuh)

RBTA bucket key `(agent_id, rule_group_primary)`; elastic window lokal
per agent; 7 fitur terkunci; IF unsupervised + frozen untuk replay/eval;
Tukey + decision matrix di luar model; ARR = reduksi unit triase;
tanpa synthetic attack ground truth; `contamination="auto"` bukan
estimasi prevalensi serangan.

Sistem ini mengurangi dan memprioritaskan unit triase. Alert bukan
otomatis serangan, anomaly score bukan label serangan, dan setiap
MetaAlert harus tetap dapat ditelusuri ke bukti mentahnya.

## Skema flat yang dibekukan

| Flat key | Wajib | Aturan |
|---|---|---|
| `id` | Ya | string non-kosong. Catatan: contoh `OmBo_6AB...` menyerupai OpenSearch `_id`; dipakai sebagai `wazuh_alert_id` apa adanya. Bila `_source.id` asli tersedia di perantara, utamakan itu dan simpan `_id` sebagai `source_document_id`. |
| `timestamp` | Ya | ISO8601 TZ-aware (`Z`/offset) atau epoch. Naive ditolak canonicalizer. |
| `agent_id` | Ya | string, mis. `005`. Hilang → `000`. |
| `agent_name` | Ya | string, mis. `rbta-arya`. Hilang → `manager.name` → `unknown`. Unknown → criticality default 1 (`src/config/domain.py`). |
| `rule_id` | Ya | string/int, mis. `510`, `19007`. |
| `rule_level` | Ya | int 0–15. |
| `rule_groups` | Ya | list string, mis. `["ossec","rootcheck"]`, `["sca"]`. Primary via `resolve_primary_rule_group` (`rootcheck`> `ossec`; `sca` tetap `sca`). |
| `rule_description` | Anjuran | → `metadata.rule_description` (tampil sebagai signature). |
| `location` | Anjuran | → `metadata.location`. |
| `message` | Anjuran | → `full_log` (evidence drill-down). |
| `decoder` | Opsional | `null` → diabaikan. |
| `details.manager.name` | Anjuran | → `manager.name` (fallback agent_name). |
| `details.data` / `srcip` | Opsional | Bila ada → `data` + `srcip`. Contoh beku tidak membawa IP → `srcip=None` (sah). |
| MITRE | Opsional | Contoh beku tidak membawa taktik → `mitre_tactics=()` (sah; fitur terkait 0). |

Pagination response: `{events[], total, limit, offset}`. Client
`offset+=len(events)` sampai `offset>=total`. Dedup otoritatif tetap
`wazuh_alert_id` di service; tidak ada drop berbasis timestamp.

## Peta translate (satu-satunya tempat yang tahu bentuk flat)

`src/ingestion/flat_api_adapter.py::flat_api_event_to_raw`:

```text
id → id | timestamp → timestamp
agent_id/agent_name → agent{id,name}
rule_id/rule_level/rule_groups/rule_description → rule{id,level,groups,description}
location → location | message → full_log
details.manager.name → manager{name}
details.data/srcip → data/srcip
```

`canonicalize_wazuh_alert` nested tidak diubah. Client:
`src/ingestion/wazuh_api_client.py::WazuhAPIClient`
(`WAZUH_API_URL`, `WAZUH_API_KEY` Bearer dan/atau
`WAZUH_API_SESSION_COOKIE` via header `Cookie`, retry 429/502/503/504,
fail-fast 401/403 tanpa body, TLS via `WAZUH_API_VERIFY_TLS`/
`WAZUH_API_CA_PATH`). Verifikasi live 2026-10-03 terhadap
`https://172.16.83.207/api/events`: `200`, `{events,total,limit,offset}`
dengan `total=306`, key event identik dengan tabel beku, paginasi
`offset=0/2` tanpa overlap, event pertama terkanonikalisasi menjadi
`(005, rbta-arya, 510, level 7, rootcheck)`. Login asli terbukti:
`POST /auth/login {email,password}` → `LOGIN OK` + `fetch_all_canonical`
306 events (field `email` dikoreksi dari tebakan awal `login` setelah
respons 422 menunjukkan `missing email`). Poller drop-in:
`src/runtime/api_live_source.py::WazuhAPILivePoller`
(poller interface sama → `LiveIngestionCoordinator(service, poller=...)`).
Aktif via `RBTA_LIVE_SOURCE=api` (default `indexer` agar replay tak tersentuh).

## Opsi C — kredensial kadaluwarsa (beku 2026-10-03)

Kontrak login perantara: `POST {WAZUH_API_URL}/auth/login` dengan
`{email, password}` (nama field via `WAZUH_API_LOGIN_FIELD` /
`WAZUH_API_PASSWORD_FIELD`), cookie baru kembali via `Set-Cookie:
guardins_session=...`. Kredensial layanan hanya dari env
(`WAZUH_API_USERNAME` / `WAZUH_API_PASSWORD`) — tidak pernah di-commit,
di-log, atau ditulis ke session file (file hanya berisi cookie).

Perilaku `WazuhAPIClient`:

```text
GET /events → 401 pertama + kredensial login ada
  → POST /auth/login sekali → simpan cookie ke WAZUH_API_SESSION_FILE
  → retry GET sekali → lanjut cycle
GET /events → 401 lagi (login ditolak / cookie langsung mati)
  → WazuhAPIAuthError fail-fast → worker backoff, cursor beku,
    terlihat di live/status.last_error (tanpa URL/body)
Tanpa kredensial login → 401 langsung fail-fast seperti sebelumnya.
```

Watcher file: sebelum setiap request client memuat ulang
`WAZUH_API_SESSION_FILE` bila isinya berubah, sehingga refresh eksternal
(operator/cron) terpakai tanpa restart. Login yang gagal (401/403 di
`/auth/login`) tidak di-retry berulang — anti lockout.

## Contoh agregasi yang diharapkan

49+ event `510/rootcheck` × `(005, rootcheck)` dalam milidetik → RBTA
menggabung ke sedikit MetaAlert (uji ARR). `/dev/.lxc/proc/*` adalah pola
false-positive container khas — reduksi triase, bukan bukti serangan.
