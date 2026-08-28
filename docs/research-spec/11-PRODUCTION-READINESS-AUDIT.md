# Production Readiness Audit

## Scope

Audit ini membandingkan code pada branch saat dokumentasi dibuat dengan source-of-truth `00`–`10`.

Status keseluruhan saat audit:

```text
NOT PRODUCTION READY
```

Dokumen ini bukan claim hasil penelitian. Ini adalah gap analysis untuk implementation agent.

## Severity

```text
P0 = correctness / research validity blocker
P1 = production reliability blocker
P2 = maintainability/operability improvement
```

---

# P0 Findings

## P0-01 — Final RBTA adaptive mode disabled

Current orchestrator menjalankan final RBTA dengan:

```text
enable_adaptive=False
```

Padahal final RBTA harus memakai adaptive ETW per-agent. Sensitivity experiment saja yang adaptive OFF.

Required fix:

- sensitivity runner explicit `adaptive=False`;
- final research run explicit `adaptive=True`;
- unit/integration test memastikan final run benar-benar memakai local delta-t.

## P0-02 — ElasticWindow global, bukan per agent

Current `rbta_core.py` memiliki satu `ElasticWindow` untuk seluruh stream.

Required fix:

```text
TemporalStateStore[agent_id]
```

Setiap agent mempunyai independent:

```text
last_timestamp
warmup_event_count
warmup_gaps
baseline_gap
ema_gap
current_delta_t
```

Mandatory EMA isolation test.

## P0-03 — Formula ETW salah

Current implementation memakai threshold/rate:

```text
HIGH_FREQ
LOW_FREQ
SHRINK_RATE=0.8
EXPAND_RATE=1.2
```

Source-of-truth membutuhkan:

```text
delta_t = base_dt * EMA_gap / baseline_gap
clip 0.5x .. 1.5x
alpha=0.10
```

Step-based adaptation harus dihapus dari primary path.

## P0-04 — Watermark `late_drop` membuang valid event

Current core memiliki `late_drop` dan `_process()` return tanpa processing.

Research method tidak mendefinisikan event dropping berdasarkan watermark.

Required fix:

- reorder buffer tetap boleh ada;
- seluruh valid parsed alert akhirnya diproses;
- no-event-loss test: `sum(meta.alert_count) == number valid processed raw alerts`.

## P0-05 — Bucket B / CompoundMetaAlert masih aktif

Current core menjalankan Bucket A + Bucket B secara paralel.

Primary methodology hanya:

```text
(agent_id, rule_group_primary)
```

Required fix:

- hapus `CompoundMetaAlert` dari active code path;
- hapus output compound;
- jangan gunakan behavioral sequencing lama untuk tujuh feature.

## P0-06 — IF masih 11 fitur

Current `feature_engineering.py` dan `isolation_forest.py` memakai 11-feature HIDS vector.

Final vector wajib tepat:

```text
max_severity
mitre_tactic_count
critical_mitre_tactic_present
alert_count_log
rule_diversity_shannon
severity_dispersion
agent_criticality
```

Required fix:

- satu `FEATURE_COLUMNS` source;
- missing feature fail-fast;
- jangan silently fill feature production dengan zero.

## P0-07 — Feature logic duplicated

Feature calculations tersebar di RBTA core, feature engineering, dan IF module.

Required fix:

```text
MetaAlert -> SevenFeatureExtractor -> FeatureVector
```

Model hanya consume vector.

## P0-08 — Dynamic contamination memakai ground truth

Current model path mengubah `contamination="auto"` menjadi contamination numerik dari `ground_truth` bila tersedia.

Ini bertentangan dengan unsupervised methodology.

Required fix:

```text
IsolationForest(contamination="auto", n_estimators=200)
```

Tidak ada label-derived contamination.

## P0-09 — Tukey threshold di-clamp 1.0

Current code:

```text
min(Q3 + 1.5*IQR, 1.0)
```

Required:

```text
Q3 + 1.5*IQR
```

Tanpa clamp.

## P0-10 — Single-event streaming score invalid

Current normalization menghitung min/max dari score batch yang sedang diproses.

Jika hanya satu meta-alert:

```text
min == max -> normalized score = 0.5
```

Artinya inference streaming satu event tidak comparable dengan training distribution.

Required fix:

- calibration transformation fit saat training;
- calibration parameters disimpan di model artifact;
- inference menggunakan calibration yang sama;
- tidak fit/min-max terhadap request live.

## P0-11 — Primary pipeline masih synthetic-ground-truth oriented

Current `main.py` masih memiliki:

```text
USE_INJECTED_DATA=True
attack scenarios
ground_truth propagation
PR-AUC/FPR flows
```

Required fix:

- keluarkan dari primary research path;
- archive legacy bila histori diperlukan;
- noise robustness tetap ada karena source-of-truth membutuhkannya.

## P0-12 — Tidak ada Wazuh Indexer source

Repo belum memiliki implementation untuk:

```text
daily index discovery
PIT
search_after
checkpoint
retry
live polling
```

Required implementation mengikuti `09-WAZUH-INDEXER-INGESTION-SPEC.md`.

## P0-13 — Parser API/file belum satu contract

Current ETL berorientasi local JSONL/CSV dan hardcoded path.

Raw Wazuh Search Hit memiliki envelope:

```text
_index
_id
_source
sort
```

Required fix:

- unwrap source envelope;
- preserve metadata;
- parser API dan archive menghasilkan `CanonicalRawAlert` identik.

## P0-14 — MITRE schema variants belum aman

Raw Wazuh alerts dapat mengekspos MITRE melalui shape berbeda, termasuk:

```text
rule.mitre.tactic / rule.mitre.technique
rule.mitre_tactics / rule.mitre_techniques
```

Parser harus normalize keduanya ke satu list tactics/techniques canonical.

## P0-15 — Test suite tidak tersedia

Audit code search tidak menemukan file test yang berarti.

Tidak boleh melakukan refactor research core tanpa automated tests.

Required minimum suites:

```text
contracts
canonicalizer
EMA isolation
RBTA aggregation
no event loss
7 features
IF config/threshold
artifact roundtrip
historical source
live source
batch-stream equivalence
state recovery
```

---

# P1 Findings

## P1-01 — Tidak ada package/dependency manifest

Tidak ditemukan root `pyproject.toml`/requirements production yang menjadi reproducible source.

Required:

```text
pyproject.toml
locked/reproducible dependency strategy
Python version declaration
pytest config
```

## P1-02 — Tidak ada container deployment contract

Tidak ditemukan Dockerfile production.

Required sebelum ASUS deployment:

```text
non-root runtime
healthcheck
read-only app image where practical
mounted state/artifacts
explicit env config
```

## P1-03 — Tidak ada durable runtime state

Current simulator dan research code dominan in-memory.

Live mode membutuhkan persistence untuk:

```text
dedup
source checkpoint
per-agent EMA
active buckets
meta-alert output
outbox
```

## P1-04 — No idempotent ingress

Historical resume, live overlap, dan transport retry dapat menduplikasi alert.

Required invariant:

```text
same wazuh_alert_id -> at most one core state mutation
```

## P1-05 — Stream simulator loads entire CSV via pandas

Ini cocok untuk eksperimen kecil tetapi bukan streaming source production.

Required:

- `AlertSource` iterator;
- paged network source;
- streaming archive reader;
- bounded memory.

## P1-06 — Hardcoded local paths

`json_orches.py` mempunyai Windows path khusus developer.

Required:

- Path dari config/CLI;
- no workstation-specific default untuk production path.

## P1-07 — Domain constants duplicated

Criticality/group mappings tersebar di modules.

Required:

```text
src/config/domain.py
```

single source.

## P1-08 — Model artifact lifecycle belum ada

Current pipeline mengembalikan in-memory model/scaler, tetapi live service membutuhkan versioned artifact + validation.

Ikuti `12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`.

## P1-09 — SOAR payload stale fields

Payload lama masih mereferensikan feature seperti entropy/progression/baseline lama.

Required:

- event schema hanya field current research + operational trace;
- custom Shuffle app tidak menghitung ulang research logic.

## P1-10 — Runtime health/readiness belum ada

ASUS service membutuhkan:

```text
/health
/ready
/runtime/stats
```

Readiness harus memeriksa state store + model artifact compatibility.

---

# P2 Findings

## P2-01 — Large multi-responsibility files

`main.py`, `rbta_core.py`, `isolation_forest.py`, `metrics.py` terlalu banyak responsibility.

Refactor harus mengikuti component boundaries `06-CODEBASE-REFACTOR-SPEC.md`.

## P2-02 — Legacy docstrings misleading

Banyak komentar masih menyebut Landauer alignment, 11 features, Bucket B, old scenario evaluation.

Sesudah refactor, grep acceptance harus memastikan tidak ada stale terminology di primary code.

## P2-03 — Observability belum structured

Required fields minimum log:

```text
run_id
component
event_id/meta_id
agent_id
source_mode
model_version
error_class
```

Tidak log full secret/token.

---

# Target Production Structure

Recommended target:

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
│   └── stream.py
├── state/
│   ├── interface.py
│   └── postgres.py
├── operational/
│   ├── api.py
│   ├── outbox.py
│   └── shuffle.py
└── evaluation/
```

Nama boleh sedikit berubah, tetapi dependency direction harus:

```text
sources -> contracts/etl
runners -> core
operational -> core interfaces
core -X-> transport/Shuffle/Telegram
```

---

# Production Ready Definition

Jangan menyebut system production-ready sampai seluruh kondisi berikut mempunyai evidence:

```text
all P0 closed
all P1 critical runtime items closed
unit/integration tests pass
EMA isolation pass
no-event-loss pass
batch-stream equivalence pass
artifact roundtrip pass
historical resume pass
live overlap/dedup pass
state recovery pass
Docker clean startup pass
health/readiness pass
secrets absent from repo/log
end-to-end Wazuh-like -> RBTA -> IF -> outbox pass
```

Dashboard/Telegram yang terlihat bekerja bukan bukti research core benar.