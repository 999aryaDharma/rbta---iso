# Operational Integration Specification

## Status dan Boundary

Dokumen ini mendefinisikan extension operasional agar research core dapat digunakan langsung pada alur live tanpa mengubah metodologi penelitian.

Target operasional utama:

```text
Wazuh
  -> RBTA REST Service
  -> agent-local RBTA runtime
  -> MetaAlert finalization
  -> 7-feature extraction
  -> Isolation Forest scoring
  -> Decision Matrix
  -> Shuffle SOAR Workflow
       -> custom RBTA Shuffle App node(s)
       -> Telegram Bot
```

Research core pada dokumen `00`–`05` tetap menjadi source of truth algoritmik. REST API, Shuffle, dan Telegram adalah adapter operasional.

## Design Goal

Code hasil refactor tidak boleh menjadi script batch-only. Core yang sama harus dapat dipanggil dari:

```text
BatchResearchRunner
LiveRBTAService
```

Tidak boleh ada implementasi RBTA/EMA/feature/model kedua di dalam Shuffle.

## Keputusan Integrasi Shuffle

**Custom Shuffle App adalah requirement utama, bukan opsi tambahan.**

Dalam istilah Shuffle, app/action adalah node reusable pada workflow. Custom app dapat dibuat dari OpenAPI untuk HTTP API atau dengan Python App SDK untuk action yang membutuhkan custom logic.

Karena RBTA akan menyediakan REST API, integrasi default adalah:

```text
RBTA REST API
    -> OpenAPI specification
    -> generate/import custom Shuffle App
    -> reusable RBTA actions visible as nodes in Shuffle workflow
```

Gunakan Python App SDK hanya bila action membutuhkan logic yang tidak tepat diletakkan pada REST API wrapper.

### Prinsip Penting

Custom Shuffle node **tidak mengimplementasikan ulang**:

```text
EMA
RBTA bucket logic
7 features
Isolation Forest
IQR threshold
Decision Matrix
```

Node hanya menjadi adapter/orchestrator yang memanggil kontrak REST service dan mengembalikan hasil terstruktur kepada workflow.

## Arsitektur Target

```text
                         +----------------------+
Wazuh -----------------> | RBTA REST Ingress    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Canonical Parser     |
                         | + Idempotency        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | RBTA Runtime         |
                         | per-agent EMA state  |
                         | active buckets       |
                         +----------+-----------+
                                    |
                          bucket finalized
                                    |
                                    v
                         +----------------------+
                         | 7 Feature Extractor  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | IF Scoring Service   |
                         | scaler/model/theta   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Decision + Outbox    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Shuffle Webhook      |  <- workflow trigger
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Custom RBTA App Node |
                         | fetch/validate/ack   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Switch / Enrichment  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Telegram Node        |
                         +----------------------+
```

## Mengapa REST Service Tetap Menangkap Alert Langsung dari Wazuh

Wazuh tidak perlu bergantung pada availability Shuffle untuk menjalankan agregasi penelitian.

Jalur ingest utama:

```text
Wazuh -> RBTA REST Service
```

bukan:

```text
Wazuh -> Shuffle -> RBTA
```

Alasannya:

- state EMA per-agent harus konsisten dan durable;
- active bucket harus tetap berjalan walaupun Shuffle sedang down;
- SOAR adalah downstream orchestration, bukan bagian algoritma agregasi;
- retry SOAR tidak boleh menggandakan raw alert di RBTA.

## Workflow Shuffle yang Direkomendasikan

Workflow dipicu ketika RBTA menghasilkan **scored/finalized MetaAlert**, bukan untuk setiap raw alert.

```text
Webhook Trigger: rbta.meta_alert.scored
        |
        v
RBTA App - Validate Event
        |
        v
RBTA App - Get Meta Alert Details   (opsional bila payload trigger sudah lengkap)
        |
        v
Switch(action)
  |-- ESCALATE
  |      -> optional enrichment
  |      -> Telegram
  |
  |-- DAILY_DIGEST
  |      -> aggregate/store digest
  |
  `-- SUPPRESS
         -> audit/end
```

Webhook adalah trigger workflow; custom RBTA app adalah node/action yang reusable di dalam workflow.

## Custom Shuffle App Specification

Nama app yang disarankan:

```text
RBTA Security Analytics
```

Versioning:

```text
1.0.0
```

App harus berasal dari satu OpenAPI contract yang juga digunakan REST API agar tidak terjadi schema drift.

### Authentication

Konfigurasi app minimum:

```text
base_url
api_key / bearer_token
request_timeout
```

Credential tidak boleh hardcoded dalam workflow atau source code.

### Action 1 — Validate Scored Event

Tujuan:

- validasi `schema_version`;
- validasi event type;
- memastikan field minimum tersedia;
- mengembalikan normalized object untuk node berikutnya.

Input:

```text
event JSON
```

Output:

```json
{
  "valid": true,
  "event_id": "meta-1234",
  "meta_id": 1234,
  "action": "ESCALATE",
  "decision": "CRITICAL"
}
```

Validasi ini tidak menghitung ulang model.

### Action 2 — Get Meta Alert

REST target konseptual:

```text
GET /v1/meta-alerts/{meta_id}
```

Output mencakup:

```text
meta-alert summary
7 feature values
score/threshold/decision
source alert IDs
model/schema version
```

### Action 3 — Get Alert Trace

REST target konseptual:

```text
GET /v1/meta-alerts/{meta_id}/trace
```

Tujuan: analyst dapat memperoleh source `wazuh_alert_id` tanpa memasukkan full raw log ke notification default.

### Action 4 — Get Runtime Stats

REST target:

```text
GET /v1/runtime/stats
```

Untuk observability workflow/admin, bukan metrik penelitian utama.

### Action 5 — Get Runtime Health

REST target:

```text
GET /v1/health
GET /v1/ready
```

Dapat digunakan dalam workflow monitoring terpisah.

### Action 6 — Acknowledge Delivery

REST target konseptual:

```text
POST /v1/events/{event_id}/ack
```

Digunakan untuk mencatat bahwa event sudah berhasil diterima/ditangani oleh Shuffle.

ACK adalah state operasional dan tidak mengubah keputusan model.

### Administrative Action — Flush Eligible Buckets

Bila dibutuhkan:

```text
POST /v1/runtime/flush
```

Harus admin-only dan **bukan** node normal pada workflow alert.

Tujuannya hanya controlled operational maintenance/testing.

## REST API Contract

### POST `/v1/alerts/wazuh`

Menerima satu alert langsung dari Wazuh/bridge.

Responsibilities:

```text
validate
canonicalize
deduplicate
update per-agent temporal state
update/finalize RBTA bucket
queue finalized meta-alert for scoring
```

Possible status:

```text
202 accepted
200 duplicate/idempotent replay
400 invalid payload
401/403 unauthorized
503 runtime/model unavailable
```

Raw ingest endpoint bukan action utama Shuffle. Ia terutama digunakan Wazuh -> RBTA.

### POST `/v1/alerts/wazuh/batch`

Opsional untuk replay/bulk ingestion. Tetap melalui canonical service yang sama.

### GET `/v1/meta-alerts/{meta_id}`

Mengembalikan meta-alert dan scoring trace.

### GET `/v1/meta-alerts/{meta_id}/trace`

Mengembalikan trace/source alert identifiers yang diperlukan analyst.

### GET `/v1/health`

Liveness process.

### GET `/v1/ready`

Readiness memverifikasi minimum:

```text
state store accessible
model loaded
scaler loaded
feature schema compatible
outbox available
```

### GET `/v1/runtime/stats`

Metric operasional:

```text
accepted alerts
duplicates
invalid alerts
active agents
active buckets
finalized meta-alerts
outbox pending
Shuffle delivery failures
```

### POST `/v1/events/{event_id}/ack`

Idempotent acknowledgement dari Shuffle.

## Important Streaming Constraint

Incoming raw alert tidak selalu langsung menghasilkan scored meta-alert.

RBTA melakukan aggregation. Scoring dilakukan saat meta-alert final.

Contoh:

```text
POST raw alert #1
-> accepted / aggregating

POST raw alert #2
-> accepted / aggregating

... local delta-t terlewati ...
-> previous bucket finalized
-> 7 features
-> IF score
-> Decision
-> outbox event
-> Shuffle workflow triggered
```

Jangan memaksa satu HTTP request raw alert = satu model prediction.

## MetaAlert Finalization Policy

Bucket dapat difinalisasi ketika:

1. alert berikut untuk key sama memiliki gap lebih besar dari local agent `delta_t`;
2. bucket mencapai maksimum 60 menit;
3. idle flush mendeteksi bucket sudah tidak menerima event melebihi window aktif;
4. controlled shutdown melakukan drain sesuai lifecycle policy.

### Idle Flush

Live service tidak memiliki akhir file. Scheduler internal perlu memeriksa bucket idle agar meta-alert terakhir suatu stream tetap dapat selesai.

Idle flush adalah extension operasional dan tidak mengubah formula EMA/RBTA.

## Idempotency

`wazuh_alert_id` adalah idempotency key ingress.

Invariant:

```text
same wazuh_alert_id -> at most one RBTA state mutation
```

Untuk downstream:

```text
event_id/meta_id -> at most one logical Shuffle delivery
```

Retry transport boleh terjadi, tetapi tidak boleh menyebabkan Telegram ganda.

## Runtime State Persistence

Persist minimum:

```text
per-agent EMA state
active RBTA buckets
processed wazuh_alert_id registry / dedup window
finalized meta-alert records
outbox delivery state
```

### Per-Agent EMA Persistence

State harus terpisah per agent:

```text
temporal_state:{agent_id}
```

Minimum persisted fields:

```text
last_timestamp
warmup_event_count
warmup gaps/baseline accumulator yang diperlukan
baseline_gap
ema_gap
current_delta_t
state_version
```

Restart tidak boleh mencampur state agent.

## Concurrency Requirement

Mutasi state untuk agent yang sama harus ordered/serialized.

```text
partition by agent_id
-> one ordered mutation stream per agent
```

Agent berbeda boleh diproses paralel.

## Model Artifact Lifecycle

REST runtime tidak melakukan training ulang per request.

Artifact aktif:

```text
IsolationForest model
RobustScaler
Tukey threshold/value dari research calibration run
feature_schema_version
model_version
metadata
```

Operational runtime memuat artifact versioned yang sudah dihasilkan research pipeline.

## Scored MetaAlert Event Contract

Contoh kontrak ke Shuffle:

```json
{
  "event_type": "rbta.meta_alert.scored",
  "schema_version": "1.0",
  "event_id": "meta-1234",
  "meta_id": 1234,
  "agent": {
    "id": "002",
    "name": "pusatkarir",
    "criticality": 3
  },
  "rule_group": "authentication_failed",
  "window": {
    "start": "2026-08-11T12:00:00Z",
    "end": "2026-08-11T12:04:00Z",
    "alert_count": 87
  },
  "security": {
    "max_severity": 10,
    "mitre_tactics": ["Credential Access"]
  },
  "scoring": {
    "anomaly_score": 0.91,
    "threshold": 0.82,
    "decision": "CRITICAL",
    "action": "ESCALATE",
    "model_version": "if-example",
    "feature_schema_version": "7f-v1"
  },
  "trace": {
    "source_alert_ids": ["...", "..."]
  }
}
```

Nilai hanya contoh schema, bukan hasil eksperimen.

## Outbox Pattern untuk Trigger Shuffle

Setelah scoring:

```text
save finalized/scored meta-alert
save outbox event
commit state
-> delivery worker POST ke Shuffle webhook trigger
-> Shuffle execution starts
-> custom RBTA node validates/fetches required details
-> successful handling -> optional ACK
-> mark delivered/handled according to delivery policy
```

Jika Shuffle down:

```text
retry with bounded exponential backoff
```

`event_id` wajib tetap sama pada retry.

## Telegram Responsibility

Telegram adalah downstream workflow action, bukan bagian model atau REST research core.

Routing default:

```text
CRITICAL   / ESCALATE -> immediate Telegram
SUSPICIOUS / ESCALATE -> immediate Telegram
NOISE_HIGH / DAILY_DIGEST -> digest path
NOISE / SUPPRESS -> no immediate notification
CONTEXTUAL_ANOMALY / SUPPRESS -> no immediate notification
```

Telegram message mengambil data dari normalized event / custom RBTA node output.

Jangan mengirim `full_log` sensitif secara default.

## Custom Shuffle App Source Layout

Jika menggunakan Python App SDK, target layout mengikuti struktur custom app Shuffle secara konseptual:

```text
shuffle-apps/
└── rbta_security_analytics/
    └── 1.0.0/
        ├── api.yaml
        ├── Dockerfile
        ├── docs.md
        ├── requirements.txt
        └── src/
            └── app.py
```

Namun karena service kita adalah HTTP REST API, **OpenAPI-generated app adalah default yang direkomendasikan**. Python SDK digunakan bila ada kebutuhan custom behavior di node yang tidak cukup diekspresikan oleh REST action biasa.

## OpenAPI as Integration Contract

REST service wajib menyediakan versioned OpenAPI schema.

Target:

```text
openapi/rbta-api-v1.yaml
```

Schema ini menjadi source untuk:

```text
REST API documentation
contract tests
Shuffle custom app generation/import
automated client generation bila diperlukan
```

Jangan menulis `api.yaml` Shuffle dan REST schema secara manual dengan kontrak berbeda jika dapat dihasilkan dari satu OpenAPI source.

## Security Boundary

Minimum operational controls:

```text
TLS pada deployment path
API authentication
Shuffle credential via app authentication config
payload size limit
rate limiting/backpressure
strict schema validation
structured audit log
no secrets in workflow parameters/log output
no sensitive full_log in Telegram by default
```

## Observability

Correlation IDs:

```text
wazuh_alert_id
meta_id
event_id
Shuffle workflow execution ID (operational only)
```

Tracing harus dapat mengikuti:

```text
raw Wazuh ingest
-> RBTA membership
-> meta finalization
-> score
-> decision
-> Shuffle trigger
-> custom RBTA node
-> Telegram outcome
```

## End-to-End Acceptance Test

Tanpa synthetic attack label:

```text
1. POST Wazuh-like alerts Agent A ke RBTA REST API
2. POST Agent B dengan cadence berbeda
3. verify EMA A/B independen
4. finalize bucket A
5. verify exactly 7 features
6. score dengan versioned test artifact
7. persist scored event/outbox
8. deliver ke Shuffle webhook test workflow
9. custom RBTA Shuffle node membaca/validasi event dan/atau fetch meta detail
10. action ESCALATE masuk ke Telegram test node/stub
11. retry event_id yang sama tidak menghasilkan duplicate logical notification
12. verify correlation trace dari wazuh_alert_id sampai Shuffle execution
```

## Definition of Done Operasional

Integrasi belum dianggap selesai hanya karena webhook berhasil menerima JSON.

Selesai berarti:

```text
[ ] RBTA REST API mempunyai OpenAPI versioned contract
[ ] Wazuh dapat mengirim alert langsung ke REST ingress
[ ] research core yang sama dipakai batch dan live
[ ] per-agent EMA state durable dan terisolasi
[ ] custom RBTA Shuffle App tersedia sebagai node reusable
[ ] app actions memakai REST API, bukan duplicate algorithm
[ ] scored event memulai workflow secara reliable
[ ] ESCALATE dapat mencapai Telegram
[ ] retry/idempotency terbukti
[ ] health/readiness/observability tersedia
[ ] integration tests end-to-end lulus
```
