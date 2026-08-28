# Dual-Mode Runtime Specification

## Goal

Codebase harus mendukung dua mode utama dengan **satu Research Core**:

```text
research-batch
replay-stream
```

Tidak boleh ada implementasi kedua untuk RBTA, EMA, tujuh feature, Isolation Forest scoring, atau Decision Matrix.

## 1. Shared Core

Semua mode memakai komponen yang sama:

```text
CanonicalRawAlert
-> RBTAEngine.process(alert)
-> finalized MetaAlert
-> SevenFeatureExtractor
-> ModelRuntime.score(meta_alert)
-> DecisionEngine
```

Boundary utama:

```text
Source/Runner berbeda
Research Core sama
```

## 2. Mode A — `research-batch`

Tujuan:

- eksperimen penelitian;
- sensitivity delta-t;
- adaptive RBTA final run;
- feature extraction;
- fit scaler + IF;
- calibration anomaly score;
- Tukey threshold;
- structural evaluation;
- artifact generation.

Input yang diperbolehkan:

```text
canonical CSV
canonical JSONL
archived Wazuh JSONL
```

Batch mode boleh memproses tanpa artificial sleep.

### Batch lifecycle

```text
load source
-> canonicalize
-> process all raw alerts
-> drain reorder buffer
-> finalize all active buckets
-> extract 7 features
-> fit/load model according to command
-> write experiment artifacts
```

Research batch yang melakukan training harus eksplisit, misalnya:

```text
rbta research train ...
```

Bukan side-effect otomatis setiap program start.

## 3. Mode B — `replay-stream`

Tujuan:

- memutar historical raw alerts seperti stream;
- demonstrasi event-by-event;
- menjalankan RBTA/EMA secara stateful;
- melakukan inference menggunakan artifact yang sudah dilatih;
- memberi output ke dashboard/DB/Shuffle.

Input:

```text
HistoricalWazuhIndexerSource
ArchivedJsonlSource
```

Replay mode **dilarang fit**:

```text
RobustScaler
IsolationForest
score calibration
Tukey threshold
```

Semua harus di-load dari artifact versioned.

## 4. Mode C — `live` Operational Extension

Walaupun CLI utama penelitian memiliki dua mode, runtime harus mendukung live source sebagai extension dari stream runner:

```text
LiveWazuhSource
-> StreamRunner
-> same Research Core
```

Jadi implementasi sebaiknya tidak membuat class `ReplayRBTAEngine` dan `LiveRBTAEngine` terpisah.

Yang berbeda hanya source dan replay clock.

## 5. Required Runtime Interfaces

Target konseptual:

```python
@dataclass(frozen=True)
class CanonicalRawAlert:
    wazuh_alert_id: str
    timestamp: datetime
    agent_id: str
    agent_name: str
    rule_group_primary: str
    rule_level: int
    rule_id: str
    mitre_tactics: tuple[str, ...]
    srcip: str | None
    agent_criticality: int
    metadata: Mapping[str, Any]
```

```python
class RBTAEngine:
    def process(self, alert: CanonicalRawAlert) -> list[MetaAlert]: ...
    def flush_idle(self, event_time: datetime) -> list[MetaAlert]: ...
    def drain(self) -> list[MetaAlert]: ...
```

`process()` boleh menghasilkan 0 atau lebih finalized meta-alert.

Incoming raw alert **tidak sama dengan** satu prediction.

## 6. Event-Time Semantics

Semua mode memakai:

```text
alert.timestamp
```

sebagai event time untuk:

```text
reorder buffer
EMA gap
RBTA merge condition
bucket start/end
```

Wall-clock hanya digunakan untuk:

```text
replay sleep
metrics operational
idle scheduler
logging
HTTP timeout
```

## 7. Replay Clock

Target abstraction:

```python
class ReplayClock:
    def wait(previous_event_time, current_event_time) -> None: ...
```

Modes:

```text
speed_factor = 1
speed_factor = 10
speed_factor = 100
speed_factor = inf/MAX
```

Formula:

```text
wall_delay = max(0, event_gap / speed_factor)
```

Tetapi calculation RBTA tetap menggunakan `event_gap` asli.

## 8. Stateful Runtime

Stream runtime menyimpan minimum:

```text
per-agent temporal state
active RBTA buckets
dedup registry
source checkpoint
finalized meta-alerts
model version loaded
outbox state
```

Batch research boleh menggunakan in-memory state selama seluruh run deterministic dan artifact/output ditulis di akhir.

## 9. Run Identity

Setiap execution memiliki `run_id`.

Contoh:

```text
research-20260828T120000Z-a1b2
replay-20260828T130000Z-c3d4
live-prod-20260828
```

`run_id` dicatat pada:

```text
logs
checkpoint
meta-alert audit
scored result
outbox event
```

Reset state harus membuat run baru kecuali recovery dari crash.

## 10. State Recovery

Crash/restart stream runner:

```text
load persisted temporal states
load active buckets
load dedup/checkpoint
load same model artifact
resume source
```

Acceptance invariant:

```text
recovery != clean reset
```

Clean reset hanya melalui explicit operator command.

## 11. Idle Flush

Live stream tidak mempunyai EOF.

Scheduler boleh memfinalisasi bucket yang idle berdasarkan active local delta-t.

Idle flush adalah operational lifecycle mechanism, bukan perubahan formula RBTA.

Historical replay dengan finite source menggunakan `drain()` pada EOF.

## 12. Batch/Replay Equivalence Requirement

Jika batch source dan replay source berisi canonical alerts yang sama, urutan event-time yang sama, research config yang sama, dan model artifact yang sama, maka hasil Research Core harus ekuivalen.

Mandatory integration test:

```text
same canonical fixture
-> BatchRunner
-> StreamRunner speed=MAX

assert finalized meta-alert structure equal
assert 7 features equal
assert anomaly score equal
assert decision equal
```

Perbedaan yang diperbolehkan hanya operational metadata seperti `ingested_at` dan wall-clock duration.

## 13. CLI Target

Contoh target, bukan nama wajib:

```text
rbta research run --input data.jsonl
rbta research train --input data.jsonl --artifact-dir artifacts/models/v1
rbta replay run --source archive --speed 100 --model artifacts/models/v1
rbta replay run --source wazuh-indexer --from 2026-04-02 --to 2026-08-28 --speed 100
rbta live run --source collector --model artifacts/models/v1
```

Semua config harus dapat berasal dari file/env/CLI tanpa hardcoded Windows path.

## 14. Forbidden Designs

Dilarang:

```text
BatchRBTA berbeda dari LiveRBTA
EMA global untuk streaming
fit model saat raw alert datang
recompute Tukey per meta-alert
normalize score dari satu meta-alert
wall-clock dipakai sebagai EMA event gap
pandas-only API sebagai contract internal
transport code berada di rbta_core
```

## 15. Acceptance Gate

Dual-mode runtime lulus jika:

```text
batch + stream memakai RBTAEngine yang sama
stream mode dapat menghasilkan 0 prediction untuk raw alert yang masih agregating
replay speed tidak mengubah result
MAX replay == batch core output pada fixture yang sama
stream mode hanya load model artifact
state recovery teruji
clean reset eksplisit
live source dapat diganti tanpa mengubah Research Core
```