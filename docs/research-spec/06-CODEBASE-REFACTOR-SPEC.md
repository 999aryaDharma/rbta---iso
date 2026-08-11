# Codebase Refactor Specification

## Tujuan

Menyelaraskan struktur codebase dengan metodologi penelitian, menghapus jalur legacy, dan mengurangi duplication/DRY violation.

## Target Structure

```text
src/
├── config/
│   ├── domain.py
│   └── experiment.py
├── contracts/
│   ├── raw_alert.py
│   └── meta_alert.py
├── etl/
│   ├── wazuh_parser.py
│   └── preprocessing.py
├── rbta/
│   ├── reorder_buffer.py
│   ├── elastic_window.py
│   └── aggregator.py
├── features/
│   └── extractor.py
├── models/
│   └── isolation_forest.py
├── evaluation/
│   ├── sensitivity.py
│   ├── noise_robustness.py
│   ├── runtime.py
│   └── structural_validity.py
├── operational/
│   ├── service.py
│   ├── model_registry.py
│   └── adapters/
│       └── shuffle.py
└── pipeline.py
```

Nama final boleh mengikuti convention project, tetapi boundary tanggung jawab harus dipertahankan.

## Single Responsibility Boundaries

### ETL
Hanya parse, normalize, dan validate data Wazuh.

### RBTA
Hanya state temporal dan pembentukan MetaAlert. RBTA tidak boleh mengetahui feature vector IF, scaler, Isolation Forest, Telegram, Shuffle, atau synthetic labels.

### Feature Extractor
Satu tempat untuk seluruh 7 feature final.

### Model
Hanya scaling, fit/load, scoring, threshold, dan decision.

### Evaluation
Tidak mengubah core algorithm. Modul evaluasi hanya menjalankan core dengan parameter eksperimen dan menghitung metrik.

### Operational Layer
Hanya transport, persistence runtime, API, SOAR delivery, retries, observability, dan deployment concerns.

## Domain Constants — Single Source

Pusatkan:

```text
AGENT_CRITICALITY
GROUP_SEVERITY_WEIGHT
CRITICAL_MITRE_TACTICS
```

Dilarang mendefinisikan ulang constants yang sama di parser, RBTA, IF, atau notifier.

## Feature Columns — Single Source

Hanya feature extractor atau schema config yang mendefinisikan nama dan urutan tujuh fitur. Semua consumer wajib import dari source tersebut.

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
hardcoded scenario result reports
stale SOAR fields referencing removed features
```

File legacy boleh dipindahkan ke `archive/` hanya jika diperlukan untuk histori penelitian. Archive tidak boleh diimport oleh runtime.

## DTO / Contract

Gunakan object/schema eksplisit untuk `RawAlert`, `MetaAlert`, dan `ScoredMetaAlert` agar batch dan REST live mode menggunakan kontrak identik.

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
anomaly_score
threshold_used
decision
action
model_version
feature_schema_version
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
- Shuffle endpoint configuration
- retry/backoff
- state storage
- model artifact path
```

Operational config tidak boleh mengganti research constants tanpa explicit research/model version bump.

## Clean Code Acceptance

- tidak ada duplicate feature implementations;
- tidak ada duplicate domain mapping;
- function fokus pada satu responsibility;
- tidak ada stale docstring menyebut 9/11/12/13 fitur;
- test names menggambarkan behavior;
- pure functions diprioritaskan untuk feature/evaluation calculations;
- I/O dipisahkan dari core calculation;
- tidak ada hidden global mutable EMA state.