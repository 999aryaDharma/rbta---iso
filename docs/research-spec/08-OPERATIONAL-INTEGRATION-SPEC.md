# Operational Integration Specification

## Status dan Boundary

Dokumen ini mendefinisikan **extension operasional** agar research core dapat digunakan langsung pada alur live:

```text
Wazuh Alert
    -> RBTA REST Service
    -> Agent-local RBTA state
    -> MetaAlert finalization
    -> 7-feature extraction
    -> Isolation Forest scoring
    -> Decision Matrix
    -> Shuffle SOAR
    -> Telegram Bot
```

Bagian REST/Shuffle/Telegram adalah rancangan implementasi operasional, bukan perubahan metodologi seminar. Research core pada dokumen `00`–`05` tetap menjadi source of truth algoritmik.

## Design Goal

Code hasil refactor tidak boleh menjadi script batch yang sulit diintegrasikan. Core harus dapat dipanggil dari:

```text
BatchResearchRunner
LiveAlertService
```

melalui service/domain API yang sama.

## Arsitektur Target

```text
                         +----------------------+
Wazuh -----------------> | REST Alert Ingress   |
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
                         | Shuffle SOAR Adapter |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Telegram Workflow    |
                         +----------------------+
```

## Important Streaming Constraint

Incoming raw alert **tidak selalu langsung menghasilkan scored meta-alert**.

RBTA melakukan aggregation. Scoring dilakukan saat meta-alert dianggap final.

Maka endpoint ingest seharusnya merespons status processing, bukan memaksa satu request = satu prediction.

Contoh response:

```json
{
  "status": "accepted",
  "wazuh_alert_id": "...",
  "processing": "aggregating"
}
```

Jika event menyebabkan bucket lama ditutup, response boleh menyertakan ID meta-alert yang baru difinalisasi.

## MetaAlert Finalization Policy

Bucket dapat difinalisasi ketika salah satu kondisi berikut terjadi:

1. alert berikut dengan key sama memiliki gap lebih besar dari local `delta_t`;
2. bucket mencapai batas maksimum 60 menit;
3. runtime menjalankan idle flush untuk bucket yang sudah tidak menerima event melebihi window aktifnya;
4. controlled shutdown melakukan drain/finalization sesuai lifecycle policy.

### Idle Flush adalah kebutuhan operasional

Dalam batch, akhir file dapat melakukan drain. Dalam live API tidak ada "akhir file". Karena itu service membutuhkan timer/scheduler yang memeriksa active bucket.

Idle flush tidak mengubah formula RBTA. Ia hanya memungkinkan bucket yang tidak pernah mendapat event berikutnya tetap selesai dan dapat dinilai.

## REST API Contract

### POST `/v1/alerts/wazuh`

Menerima satu alert Wazuh.

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

### POST `/v1/alerts/wazuh/batch`

Opsional untuk bridge/bulk replay. Tetap melalui canonical service yang sama, bukan code path kedua.

### GET `/v1/meta-alerts/{meta_id}`

Mengembalikan meta-alert dan scoring trace bila tersedia.

### GET `/v1/health`

Liveness process.

### GET `/v1/ready`

Readiness minimum memverifikasi:

```text
state store accessible
model loaded
scaler loaded
feature schema compatible
outbox available
```

### GET `/v1/runtime/stats`

Operational metric, misalnya:

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

Endpoint ini tidak digunakan sebagai metrik akademik kecuali eksplisit diekspor ke eksperimen.

## Idempotency

`wazuh_alert_id` adalah idempotency key utama.

Invariant:

```text
same wazuh_alert_id -> at most one state mutation
```

Retry dari Wazuh/bridge/network tidak boleh menggandakan `alert_count` pada bucket.

## Runtime State Persistence

Live service memiliki mutable state yang tidak ada pada script stateless:

```text
per-agent EMA state
active RBTA buckets
processed wazuh_alert_id registry / dedup window
pending finalized meta-alerts
outbox delivery state
```

State ini harus dapat dipulihkan setelah restart atau memiliki recovery policy eksplisit.

### Per-Agent EMA Persistence

Ini paling penting.

State Agent A disimpan terpisah dari Agent B:

```text
temporal_state:{agent_id}
```

Minimum persisted fields:

```text
last_timestamp
warmup_gap_count / required warmup state
baseline_gap
ema_gap
current_delta_t
state_version
```

Restart service tidak boleh secara diam-diam mencampur state agent.

## Concurrency Requirement

REST dapat menerima alert paralel. Mutasi state untuk agent yang sama harus serialized/atomic agar urutan state tidak race.

Model concurrency yang disarankan secara konseptual:

```text
partition by agent_id
-> one ordered mutation stream per agent
```

Agent berbeda boleh diproses paralel.

Acceptance test:

```text
100 concurrent alerts Agent A
-> deterministic result equivalent to valid ordered processing policy
```

## Model Artifact Lifecycle

Operational service tidak melakukan training ulang Isolation Forest per request.

Artifact aktif minimum:

```text
isolation_forest model
RobustScaler
threshold policy/value
feature_schema_version
model_version
metadata
```

Training/evaluation menghasilkan versioned artifact; live service memuat artifact yang sudah disetujui.

### Threshold in Live Mode

Tukey IQR membutuhkan distribusi score. Karena satu request tidak memiliki distribusi, operational runtime harus menggunakan threshold hasil calibration/training research run yang versioned, atau mekanisme rolling recalibration terpisah yang nantinya didefinisikan sebagai extension penelitian.

**Default scope saat ini:** gunakan threshold versioned dari artifact research run. Jangan recalibrate per request.

## Scored MetaAlert Event Contract

Payload internal menuju Shuffle sebaiknya stabil dan tidak bergantung pada struktur Python internal.

Contoh:

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
    "model_version": "if-...",
    "feature_schema_version": "7f-v1"
  },
  "trace": {
    "source_alert_ids": ["...", "..."]
  }
}
```

Nilai di atas hanya contoh kontrak, bukan hasil eksperimen.

## Shuffle SOAR Integration

Ada dua boundary yang didukung agar implementasi fleksibel.

### Option A — Webhook Trigger + Workflow Nodes

RBTA service POST `ScoredMetaAlert` ke webhook workflow Shuffle.

Workflow:

```text
Webhook Trigger
    -> Validate schema
    -> Switch(action)
       -> ESCALATE -> enrich/format -> Telegram
       -> DAILY_DIGEST -> store/aggregate
       -> SUPPRESS -> audit only/end
```

Ini adalah opsi awal yang direkomendasikan karena research service tetap independen dari detail internal workflow.

### Option B — Custom Shuffle App / Action Node

Buat custom Shuffle action/node yang memahami schema `rbta.meta_alert.scored` atau dapat memanggil RBTA REST API.

Contoh logical actions:

```text
RBTA - Get Meta Alert
RBTA - Get Runtime Stats
RBTA - Flush Eligible Buckets (admin/internal only)
RBTA - Score Finalized Meta Alert (internal integration only)
```

Untuk jalur alert utama tetap disarankan event push dari service ke Shuffle agar latency rendah dan tidak membutuhkan polling.

## Outbox Pattern untuk Shuffle Delivery

Jangan langsung menganggap POST ke Shuffle berhasil permanen.

Setelah scoring:

```text
save scored event
save outbox record
commit
-> delivery worker sends to Shuffle
-> mark delivered
```

Jika Shuffle down:

```text
retry with backoff
```

Idempotency downstream gunakan `event_id/meta_id` agar retry tidak menghasilkan Telegram ganda.

## Telegram Responsibility

Telegram adalah output channel, bukan bagian model.

Formatting dan delivery sebaiknya berada pada Shuffle workflow atau dedicated notification adapter.

Minimal message untuk immediate escalation:

```text
[CRITICAL/SUSPICIOUS]
Agent: ...
Rule group: ...
Alerts: ... | Max severity: ...
Anomaly score: ... | Threshold: ...
MITRE: ...
Meta ID: ...
```

### Routing

```text
CRITICAL   -> immediate Telegram
SUSPICIOUS -> immediate Telegram
NOISE_HIGH -> digest path
NOISE      -> no immediate notification
CONTEXTUAL_ANOMALY -> no immediate notification
```

Exact chat/channel policy adalah konfigurasi operasional, bukan research logic.

## Security Boundary

REST ingress tidak boleh dibuka tanpa authentication/authorization yang sesuai deployment.

Minimum operational controls:

```text
TLS pada deployment path
request authentication
payload size limit
rate limiting / backpressure
schema validation
no sensitive full_log in Telegram by default
structured audit log
```

Full raw log dan sensitive field tidak perlu diteruskan ke Shuffle/Telegram jika tidak diperlukan untuk triage.

## Observability

Gunakan correlation identifiers:

```text
wazuh_alert_id
meta_id
event_id
```

Log dapat menelusuri:

```text
raw ingest
-> bucket membership
-> meta finalization
-> score
-> decision
-> Shuffle delivery
-> Telegram workflow outcome
```

## Deployment Separation

Disarankan secara konseptual:

```text
Research CLI / experiment runner
    menggunakan package core

REST runtime service
    menggunakan package core yang sama
```

Tidak boleh copy-paste algoritma menjadi dua implementasi terpisah.

## End-to-End Acceptance Test

Gunakan Wazuh-like fixture tanpa synthetic attack label:

```text
1. POST beberapa alert Agent A + group X
2. POST traffic Agent B dengan cadence berbeda
3. verifikasi EMA A dan B independen
4. trigger/finalize bucket A
5. verify 7 features
6. score dengan artifact test model
7. verify decision payload
8. deliver ke Shuffle stub
9. retry delivery tidak membuat duplicate downstream event
```

## Future-Compatible Interface

Core service harus cukup decoupled sehingga output channel dapat ditambah tanpa mengubah penelitian:

```text
Shuffle
Telegram
email
SIEM dashboard
case management
other SOAR
```

Semua channel mengonsumsi `ScoredMetaAlert`; tidak ada channel yang memanggil atau memodifikasi EMA/feature/model internals.