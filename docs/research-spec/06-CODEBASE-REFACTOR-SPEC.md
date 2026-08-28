# Codebase Refactor Specification

## Tujuan

Menyelaraskan struktur codebase dengan metodologi penelitian, menghapus jalur legacy, mengurangi duplication/DRY violation, dan menyediakan satu core untuk batch, historical replay, serta live ingestion.

## Target Structure

```text
src/
├── config/
│   ├── domain.py
│   ├── research.py
│   └── operational.py
├── contracts/
│   ├── raw_alert.py
│   ├── meta_alert.py
│   └── scored_meta_alert.py
├── sources/
│   ├── base.py
│   ├── archive.py
│   └── wazuh/
│       ├── client.py
│       ├── historical.py
│       └── live.py
├── etl/
│   └── wazuh_canonicalizer.py
├── rbta/
│   ├── reorder_buffer.py
│   ├── temporal_state.py
│   └── engine.py
├── features/
│   └── extractor.py
├── models/
│   ├── isolation_forest.py
│   ├── score_calibration.py
│   └── registry.py
├── runners/
│   ├── batch.py
│   ├── stream.py
│   └── replay_clock.py
├── state/
│   ├── interface.py
│   └── postgres.py
├── evaluation/
│   ├── sensitivity.py
│   ├── noise_robustness.py
│   ├── runtime.py
│   └── structural_validity.py
├── operational/
│   ├── api.py
│   ├── outbox.py
│   └── shuffle.py
└── pipeline.py
```

Nama final boleh mengikuti convention project, tetapi boundary tanggung jawab harus dipertahankan.

## Single Responsibility Boundaries

### Sources

Hanya memperoleh raw events dari archive/Wazuh dan mengelola transport cursor/retry. Source tidak melakukan RBTA, feature engineering, atau model scoring.

### ETL / Canonicalizer

Hanya parse, normalize, dan validate data Wazuh menjadi `CanonicalRawAlert`. Parser file dan parser API harus menghasilkan contract setara.

### RBTA

Hanya state temporal dan pembentukan MetaAlert. RBTA tidak boleh mengetahui feature vector IF, scaler, Isolation Forest, Telegram, Shuffle, atau synthetic labels.

### Feature Extractor

Satu tempat untuk seluruh 7 feature final.

### Model

Hanya scaling, fit/load, score calibration, scoring, threshold, dan decision sesuai specification. Training dan inference lifecycle harus eksplisit.

### Runners

Mengorkestrasi source -> shared Research Core. Batch dan stream tidak memiliki algoritma RBTA/feature/model sendiri.

### State

Mengelola durable operational state: checkpoint, dedup, per-agent temporal state, active buckets, finalized meta-alerts, outbox.

### Evaluation

Tidak mengubah core algorithm. Modul evaluasi hanya menjalankan core dengan parameter eksperimen dan menghitung metrik.

### Operational Layer

Hanya API, persistence integration, SOAR delivery, retries, observability, dan deployment concerns.

## Dependency Direction

Allowed:

```text
sources -> contracts
etl -> contracts + config
rbta -> contracts + config
features -> contracts + config
models -> features/contracts
runners -> sources + etl + rbta + features + models
operational -> runners/core interfaces + state
```

Forbidden:

```text
rbta -> Shuffle
rbta -> HTTP client
features -> Wazuh API
models -> Telegram
live source -> feature calculation
frontend -> model training
```

## Domain Constants — Single Source

Pusatkan:

```text
AGENT_CRITICALITY
GROUP_SEVERITY_WEIGHT
CRITICAL_MITRE_TACTICS
```

Dilarang mendefinisikan ulang constants yang sama di parser, RBTA, IF, atau notifier.

## Feature Columns — Single Source

Hanya feature extractor/schema config yang mendefinisikan nama dan urutan tujuh fitur. Semua consumer wajib import dari source tersebut.

## Remove Legacy

Primary code path harus membersihkan:

```text
CompoundMetaAlert
Bucket B
compound window output
synthetic scenario A/B/C
ground_truth propagation
compute_dynamic_contamination
old 11-feature vector
old feature duplication in rbta_core/isolation_forest
watermark late_drop path
HIGH_FREQ/LOW_FREQ step ETW
SHRINK_RATE/EXPAND_RATE
hardcoded scenario result reports
stale SOAR fields referencing removed features
```

File legacy boleh dipindahkan ke `archive/` hanya jika diperlukan untuk histori penelitian. Archive tidak boleh diimport oleh runtime.

## DTO / Contract

Gunakan object/schema eksplisit untuk `CanonicalRawAlert`, `MetaAlert`, dan `ScoredMetaAlert` agar batch dan stream/live menggunakan kontrak identik.

Minimum `ScoredMetaAlert`:

```text
meta_id
agent_id
agent_name
rule_group_primary
start_time
end_time
alert_count
max_severity
mitre_tactics
seven_features
raw_model_score
anomaly_score
threshold_used
decision
action
model_version
feature_schema_version
score_calibration_version
source_alert_ids
```

## Error Handling

Tahap kritis harus fail-fast:

```text
invalid required schema
feature missing
model artifact incompatible
invalid state serialization
mapping integrity failure
partial PIT historical export
checkpoint corruption
```

Kegagalan delivery ke Shuffle boleh di-retry tanpa menggagalkan processing internal selama event hasil scoring tetap dapat dilacak untuk pengiriman ulang.

## Configuration

Pisahkan:

```text
Research configuration
- alpha
- warmup size
- base delta-t
- max bucket duration
- IF estimators
- permutation count

Operational configuration
- API bind address
- authentication configuration
- Wazuh source configuration
- poll interval / overlap
- request timeout / retry
- Shuffle endpoint
- state storage
- model artifact version/path
```

Operational config tidak boleh mengganti research constants tanpa explicit research/model version bump.

## Dual-Mode Rule

```text
BatchRunner -> shared RBTAEngine
StreamRunner -> shared RBTAEngine
```

Tidak boleh ada `BatchRBTA` dan `LiveRBTA` dengan formula berbeda.

Historical replay speed tidak boleh memengaruhi event-time calculation.

## Model Lifecycle Rule

Training:

```text
fit scaler
fit IF
fit score calibration
compute Tukey
publish artifact
```

Replay/live:

```text
load scaler
load IF
load score calibration
load Tukey
inference only
```

Single-event live score tidak boleh dinormalisasi menggunakan min/max request tersebut.

## Clean Code Acceptance

- tidak ada duplicate feature implementations;
- tidak ada duplicate domain mapping;
- function fokus pada satu responsibility;
- tidak ada stale docstring menyebut 9/11/12/13 fitur sebagai primary model;
- test names menggambarkan behavior;
- pure functions diprioritaskan untuk feature/evaluation calculations;
- I/O dipisahkan dari core calculation;
- tidak ada hidden global mutable EMA state;
- source/network resource ditutup deterministically;
- checkpoint/dedup tidak tersebar sebagai ad-hoc global state;
- no workstation-specific hardcoded path pada production path.