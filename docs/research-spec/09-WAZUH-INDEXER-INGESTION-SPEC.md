# Wazuh Historical + Live Ingestion Specification

## Status

Dokumen ini mendefinisikan **cara resmi mengambil raw alert Wazuh** untuk dua kebutuhan:

1. historical backfill/replay;
2. live alert ingestion ketika agent Wazuh kembali aktif.

Dokumen ini adalah operational specification. Ia **tidak mengubah** formula RBTA, EMA per-agent, tujuh feature, Isolation Forest, threshold, atau Decision Matrix pada dokumen `00`–`05`.

## Observasi Infrastruktur yang Sudah Terbukti

Pada environment Wazuh kampus yang diuji:

```text
Wazuh Dashboard HTTPS : 172.16.83.180:443   reachable dari jaringan kampus
Wazuh Server API      : 172.16.83.180:55000 reachable dari jaringan kampus
SSH host              : 172.16.83.180:22    reachable dari jaringan kampus
Wazuh Indexer API     : 172.16.83.180:9200  tidak diekspos langsung ke client LAN
```

Indexer tetap dapat dikueri melalui Dashboard Dev Tools. Pengujian berhasil mengambil raw alerts dari `wazuh-alerts-*`, membuat Point-In-Time untuk index harian, dan melakukan pagination dengan `search_after`.

Konsekuensi desain:

- **JANGAN membuka port 9200 ke Internet.**
- Script ASUS hanya boleh mengakses Indexer melalui jalur privat/terotorisasi atau menerima push dari collector di sisi kampus.
- Dashboard Dev Tools hanya alat diagnosis/manual, bukan transport production.

---

# 1. Canonical Ingestion Boundary

Semua sumber data wajib menghasilkan object canonical yang sama sebelum masuk Research Core.

```text
Historical Wazuh Indexer
        |
        v
WazuhIndexerHistoricalSource
        |
        +------------------+
                           |
Live Wazuh source          |
        |                  |
        v                  v
WazuhLiveSource ------> WazuhCanonicalizer
                           |
                           v
                    CanonicalRawAlert
                           |
                           v
                       RBTAEngine
```

Research Core tidak boleh mengetahui apakah alert berasal dari CSV, Indexer API, replay archive, atau live collector.

## Minimum Source Envelope

Adapter API mempertahankan metadata transport berikut untuk audit:

```text
source_index
source_document_id
source_sort
fetched_at
source_mode
raw_source
```

`raw_source` adalah `_source` Wazuh asli atau object alert JSON asli.

Canonical `wazuh_alert_id` berasal dari top-level `_source.id`/`id` Wazuh, bukan dari OpenSearch `_id`.

OpenSearch `_id` tetap disimpan sebagai `source_document_id` karena berguna untuk audit dan troubleshooting.

---

# 2. Historical Backfill / Replay Source

Historical mode digunakan untuk mengambil data masa lalu yang sudah tersimpan di Wazuh Indexer.

## 2.1 Daily Index Discovery

Jangan mengasumsikan setiap tanggal mempunyai index.

Contoh yang sudah ditemukan:

```text
2026-04-01 -> index tidak ada
2026-04-02 -> index ada
...
2026-04-30 -> index ada
```

Source harus discover index terlebih dahulu menggunakan pola tanggal, lalu mengurutkannya secara ascending.

Contoh target:

```text
wazuh-alerts-4.x-2026.04.02
wazuh-alerts-4.x-2026.04.03
...
```

Missing date adalah kondisi valid dan harus dicatat, bukan dianggap fatal.

## 2.2 PIT Scope

Point-In-Time **wajib dibuat per daily index**, bukan satu PIT wildcard untuk seluruh periode.

Alasan praktis yang sudah terbukti pada cluster:

```text
POST /wazuh-alerts*/_search/point_in_time
```

mencoba membuka context terlalu banyak dan menyentuh limit `search.max_open_pit_context`.

Gunakan:

```text
POST /wazuh-alerts-4.x-YYYY.MM.DD/_search/point_in_time
```

Requirement:

```text
allow_partial_pit_creation = false
```

Historical export tidak boleh menerima snapshot yang kehilangan shard secara diam-diam.

Jika PIT gagal penuh, retry sesuai policy. Jangan lanjut ke export hari tersebut dengan partial data.

## 2.3 Stable Sort

Urutan canonical historical replay:

```text
@timestamp ASC
id ASC
```

Mapping yang diuji menunjukkan top-level field `id` bertipe `keyword`, sehingga dapat digunakan sebagai tie-breaker.

Cursor page berbentuk:

```json
[
  1775118765670,
  "1775118765.0"
]
```

`search_after` berikutnya wajib menggunakan seluruh nilai `sort` persis dalam urutan yang sama.

## 2.4 Page Fetch vs Event Delivery

Transport API **tidak** boleh melakukan satu HTTP request untuk setiap alert.

Default production recommendation:

```text
page_size = 500
```

Boleh dinaikkan ke `1000` setelah benchmark memory/latency.

Semantik:

```text
Indexer request -> 500 hits
                   |
                   +-> yield alert #1
                   +-> yield alert #2
                   +-> ...
                   `-> yield alert #500

next search_after -> next 500 hits
```

Research Core tetap menerima alert satu-per-satu walaupun transport mengambil page secara efisien.

## 2.5 Historical Checkpoint

Checkpoint tidak boleh menyimpan PIT sebagai satu-satunya state karena PIT dapat expire.

Persist minimum:

```json
{
  "mode": "historical",
  "index_name": "wazuh-alerts-4.x-2026.04.02",
  "last_sort": [1775118766650, "1775118766.393"],
  "processed_count": 2,
  "last_wazuh_alert_id": "1775118766.393",
  "updated_at": "..."
}
```

PIT adalah ephemeral session state.

Jika process restart:

1. buka PIT baru pada daily index yang sama;
2. gunakan checkpoint cursor;
3. dedup tetap wajib melindungi dari replay overlap;
4. lanjutkan sampai index habis.

## 2.6 Daily Lifecycle

Pseudo lifecycle:

```text
for index in sorted(discovered_daily_indices):
    if index already completed:
        continue

    pit = open_pit(index)
    try:
        cursor = checkpoint_for(index)
        while True:
            page = search(pit, sort, search_after=cursor)
            if page empty:
                mark index completed
                break

            for hit in page:
                yield canonicalize(hit)

            cursor = page[-1].sort
            persist checkpoint
    finally:
        close_pit(pit)
```

`close_pit()` wajib berada pada `finally`.

---

# 3. Historical Archive on ASUS

Setelah historical alerts berhasil ditarik dari kampus, ASUS boleh menyimpan local archive agar demo/replay tidak membutuhkan koneksi kampus.

Recommended format:

```text
JSONL compressed (.jsonl.gz) untuk raw/audit archive
Parquet untuk analytical batch jika dibutuhkan
```

Jangan menyalin shard OpenSearch secara langsung.

Archive raw minimum mempertahankan alert JSON Wazuh dan metadata:

```text
source_index
source_document_id
wazuh_alert_id
timestamp
```

Retention local raw archive adalah operational policy; research output/meta-alert/artifact tidak boleh ikut terhapus oleh cleanup cache.

---

# 4. Replay Clock

Historical replay menggunakan dua clock berbeda.

```text
event_time  = timestamp asli Wazuh
wall_time   = waktu process ASUS menerima/memutar event
```

**RBTA/EMA wajib menggunakan `event_time`.**

Contoh:

```text
Wazuh event A = 10:00:00
Wazuh event B = 10:15:00

Replay 100x:
wall gap sekitar 9 detik
research gap tetap 15 menit
```

Replay acceleration tidak boleh mengubah hasil RBTA.

Supported replay speed:

```text
1x
10x
100x
MAX
```

`MAX` berarti tidak melakukan artificial sleep, tetapi tetap mempertahankan event timestamp asli.

---

# 5. Live Ingestion When Agents Return

Ketika agent yang sekarang disconnected kembali aktif, Wazuh Manager akan kembali menerima event dan menghasilkan alert baru. Live architecture harus dapat berpindah dari historical replay ke event baru tanpa mengganti Research Core.

## Important Constraint

ASUS berada di rumah sedangkan Wazuh berada pada jaringan kampus private.

Karena itu ASUS **tidak otomatis dapat** memanggil `172.16.83.180:9200` hanya karena code live sudah dibuat.

Live ingestion memerlukan salah satu jalur network yang disetujui kampus.

## 5.1 Preferred Production Pattern — Campus Collector Push

Pattern yang direkomendasikan:

```text
Wazuh Manager / campus-side host
       |
       | read new alerts
       v
Lightweight Collector
       |
       | outbound authenticated HTTPS / private overlay
       v
ASUS RBTA Ingress
       |
       v
CanonicalRawAlert -> RBTA -> IF -> Shuffle
```

Collector dapat memperoleh alert melalui dua sumber:

### Source A — `alerts.json` tail

Wazuh Manager secara default menghasilkan JSON alerts pada:

```text
/var/ossec/logs/alerts/alerts.json
```

Collector dapat mengikuti file tersebut dan push alert baru ke ASUS.

Keunggulan:

- near-real-time;
- tidak perlu polling Indexer untuk setiap alert;
- tidak perlu expose 9200;
- alert diterima sebelum/bersamaan dengan proses indexing downstream.

### Source B — Indexer polling dari sisi kampus

Collector yang berjalan di dalam jaringan kampus dapat memanggil Indexer secara lokal/private lalu push alert ke ASUS.

Pilih Source A bila akses file manager diizinkan. Pilih Source B bila hanya Indexer API yang diizinkan.

## 5.2 Alternative — Private Route from ASUS to Campus

Jika kampus mengizinkan VPN/private overlay:

```text
ASUS
  -> VPN / Tailscale route / approved tunnel
  -> Wazuh network
  -> Indexer API
```

Bentuk yang dapat digunakan:

```text
Tailscale subnet router di host kampus
VPN kampus resmi
SSH jump/tunnel melalui host yang memang dapat dijangkau ASUS
```

Route sebaiknya dibatasi ke host/port yang dibutuhkan, bukan seluruh subnet kampus.

Jangan port-forward `9200` langsung ke Internet.

---

# 6. Live Indexer Polling Semantics

Historical mode menggunakan PIT karena dataset harus snapshot-consistent.

**Live mode tidak menggunakan long-lived PIT untuk tailing event baru**, karena PIT adalah snapshot dan tidak akan melihat dokumen yang masuk setelah PIT dibuat.

Jika live source memakai Indexer API, gunakan incremental polling.

## 6.1 Poll Window

Recommended starting configuration:

```text
poll_interval = 5 seconds
lookback_overlap = 5 minutes
page_size = 500
```

Query current/recent daily indices:

```text
@timestamp >= high_watermark - lookback_overlap
sort = [@timestamp ASC, id ASC]
```

Overlap disengaja untuk menangkap indexing delay/out-of-order ingestion.

Semua duplicate dibuang berdasarkan `wazuh_alert_id` sebelum mutasi RBTA.

## 6.2 Why Overlap + Dedup

Tanpa overlap:

```text
checkpoint timestamp -> query strictly greater
```

alert yang terlambat ter-index dengan timestamp lebih lama dapat terlewat.

Dengan overlap:

```text
query sedikit mundur
-> baca ulang sebagian alert
-> dedup registry
-> process hanya alert baru
```

Ini adalah reliability mechanism transport, bukan perubahan algoritma RBTA.

## 6.3 Daily Rollover

Live source harus menangani pergantian index UTC/date tanpa restart manual.

Pada rollover:

```text
poll index hari sebelumnya + hari sekarang selama overlap period
```

Setelah overlap aman berlalu, index lama tidak perlu dipoll lagi.

---

# 7. Seamless Backfill -> Live Handoff

Goal:

```text
historical end
2026-08-xx alert terakhir
       |
       | no gap/duplicate
       v
live alert pertama ketika agent aktif kembali
```

Persist `last_processed_event_time` dan dedup registry.

Handoff procedure:

1. historical export/replay selesai pada cutoff yang diketahui;
2. simpan `wazuh_alert_id` terakhir dan event time;
3. live collector mulai dengan lookback sebelum cutoff;
4. duplicate historical events dibuang oleh idempotency layer;
5. event yang belum pernah diproses masuk ke RBTA.

Tidak boleh melakukan reset EMA hanya karena source berpindah historical -> live jika target experiment/runtime memang membutuhkan continuity state yang sama.

Untuk demo yang dimulai dari clean state, reset harus eksplisit dan diberi `run_id` baru.

---

# 8. Agent Reconnect / Agent ID Change

Server yang dihidupkan kembali dapat:

```text
reconnect menggunakan agent_id lama
atau
terdaftar sebagai agent_id baru
```

Pipeline tidak boleh hardcode daftar ID `001..007`.

Canonical identity mempertahankan:

```text
agent_id
agent_name
agent_ip optional
```

EMA state dipartisi berdasarkan `agent_id` sesuai source-of-truth penelitian.

Jika server yang secara logical sama diregistrasikan ulang dengan agent ID baru, runtime memperlakukannya sebagai temporal state baru kecuali peneliti secara eksplisit mendefinisikan migration mapping sebelum run. Jangan menggabungkan dua ID secara otomatis.

Agent criticality berasal dari domain config berdasarkan asset identity yang didefinisikan penelitian. Unknown asset harus menghasilkan warning/audit metadata dan mengikuti explicit default policy, bukan silently mengubah mapping lama.

---

# 9. Idempotency

Ingress invariant:

```text
same wazuh_alert_id -> at most one Research Core mutation
```

Dedup diperlukan untuk:

- retry HTTP;
- live polling overlap;
- historical resume;
- source failover;
- replay accidental duplication.

Dedup check dilakukan **sebelum** `RBTAEngine.process()`.

OpenSearch `_id` tidak menggantikan Wazuh alert ID sebagai primary idempotency key.

---

# 10. Retry / Failure Policy

Retry transient failure:

```text
connect timeout
read timeout
HTTP 429
HTTP 502
HTTP 503
HTTP 504
connection reset
```

Gunakan exponential backoff + bounded jitter.

Fail-fast / operator intervention:

```text
401 authentication failure
403 authorization failure
invalid TLS trust configuration
canonical schema incompatibility
feature/model schema mismatch
repeated PIT shard failure
checkpoint corruption
```

Retry tidak boleh menggandakan processing karena idempotency tetap aktif.

---

# 11. Credentials and Security

Credential Wazuh Indexer, SSH, tunnel, ASUS ingress token, dan Shuffle token:

```text
MUST NOT be committed to Git
MUST NOT be hardcoded
MUST NOT be printed in normal logs
```

Gunakan environment/secrets file di deployment host dengan permission terbatas.

Minimum transport requirements:

```text
TLS/private overlay
authentication
least-privilege Indexer account (read only alerts)
request timeout
audit log
credential rotation support
```

Indexer account untuk collector hanya membutuhkan read/search pada alert indices dan API yang dibutuhkan; jangan gunakan admin credential jika read-only role dapat dibuat.

---

# 12. Required Interfaces

Target interface konseptual:

```python
class AlertSource(Protocol):
    def iter_alerts(self) -> Iterator[RawSourceEvent]: ...

class HistoricalWazuhIndexerSource(AlertSource):
    ...

class ArchivedJsonlSource(AlertSource):
    ...

class LiveWazuhSource(AlertSource):
    ...
```

Canonicalizer:

```python
def canonicalize_wazuh_event(event: RawSourceEvent) -> CanonicalRawAlert:
    ...
```

Source tidak melakukan feature engineering atau RBTA aggregation.

---

# 13. Test Requirements

Minimum tests:

```text
index discovery skips missing dates
PIT opened per daily index only
partial PIT rejected
stable sort contains timestamp + id
search_after uses exact previous sort
500-hit page yields 500 individual events
PIT always closed on success
PIT always closed on exception
checkpoint resume does not duplicate events
live overlap duplicate is ignored
late indexed event inside overlap is processed
index rollover does not lose events
event_time is Wazuh timestamp, not fetched_at
agent ID is not hardcoded
401/403 fail fast
429/503 retry boundedly
credentials never appear in logs
```

Integration fixture harus menggunakan saved Wazuh hit yang menyerupai alert nyata yang sudah diuji pada cluster.

---

# 14. Acceptance Gate

Ingestion layer boleh dinyatakan siap jika evidence membuktikan:

```text
historical daily-index export dapat resume
0 duplicate state mutation
0 valid event loss pada fixture
PIT resource selalu ditutup
stream replay menggunakan event_time
live overlap + dedup berjalan
historical dan live source menghasilkan CanonicalRawAlert schema identik
Research Core tidak mengetahui transport source
```

Tidak boleh menyebut production-ready hanya karena satu query Dev Tools berhasil.