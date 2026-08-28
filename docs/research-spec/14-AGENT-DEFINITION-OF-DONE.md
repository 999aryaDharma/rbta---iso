# Agent Definition of Done

## Purpose

Dokumen ini mengatur kapan coding agent boleh mengatakan sebuah task/sprint **DONE**.

Tujuannya mencegah agent menandai selesai hanya karena code berjalan di satu happy path, tanpa membuktikan kesesuaian metodologi, test, recovery, atau integration contract.

## 1. Source-of-Truth Reading Gate

Sebelum coding, agent wajib membaca minimal:

```text
00-SOURCE-OF-TRUTH.md
spec terkait task
13-ANTIGRAVITY-SPRINT-PLAN.md
14-AGENT-DEFINITION-OF-DONE.md
```

Untuk task RBTA/IF, agent juga wajib membaca `02`, `03`, `04`.

Untuk Wazuh ingest, agent wajib membaca `09`, `10`, `12`.

Agent tidak boleh mengandalkan docstring source code lama jika bertentangan dengan docs.

## 2. TDD Gate

Untuk setiap behavior non-trivial:

```text
write failing test
run and observe expected failure
implement minimum correct behavior
run targeted test
run affected suite
commit
```

Agent harus melaporkan command dan hasil, bukan hanya berkata “tests pass”.

## 3. No Hidden Methodology Change

Agent dilarang mengubah tanpa approval:

```text
EMA alpha
warm-up size
ETW formula
ETW clamp
bucket key
max bucket duration
seven feature list/order
IF estimators
contamination mode
Tukey formula
Decision Matrix
False Positive Gate
sensitivity values
noise rates
permutation count
```

Jika implementation constraint membuat salah satu sulit, agent harus STOP dan eskalasi.

## 4. No Duplicate Algorithm

DONE gagal jika ada implementasi algoritma kedua untuk mode berbeda.

Contoh terlarang:

```text
batch_rBTA.py punya formula A
live_rbta.py punya formula B
```

Allowed:

```text
BatchRunner -> shared RBTAEngine
StreamRunner -> shared RBTAEngine
```

## 5. No Silent Fallback

Production/research path tidak boleh silently:

```text
fill missing required feature with 0
replace invalid model artifact with new random model
reset EMA after error
skip malformed required schema without accounting
ignore PIT partial shard failure
ignore failed state persistence
```

Error harus jelas dan testable.

## 6. Evidence Required Per Task

Setiap task completion report minimum:

```text
files changed
tests added/changed
exact test command
exact pass/fail summary
known limitations
commit SHA
```

Jika external integration belum dapat diuji karena credential/network belum tersedia:

```text
implementation status = code complete against fixture
integration status = BLOCKED_EXTERNAL
```

Jangan claim production verified.

## 7. Sprint Gate Evidence

Satu sprint hanya DONE bila:

- seluruh tasks sprint selesai;
- targeted tests pass;
- regression suite relevant pass;
- acceptance gate sprint terpenuhi;
- no source-of-truth contradiction;
- no secrets committed;
- code review/self-review dilakukan.

Sprint N+1 tidak dimulai sebelum gate N lulus.

## 8. Mandatory Behavioral Proofs

### EMA Isolation

Agent harus membuktikan:

```text
Agent A high frequency
Agent B low frequency
A update cannot mutate B state
B update cannot mutate A state
```

### Event Conservation

Setelah finite run + drain:

```text
sum(meta_alert.alert_count) == valid_unique_raw_alerts_processed
```

### Idempotency

Dua ingress dengan `wazuh_alert_id` sama:

```text
first -> one mutation
second -> zero additional mutation
```

### Batch/Replay Equivalence

Same canonical input + same artifact:

```text
BatchRunner == StreamRunner(speed=MAX)
```

untuk research outputs.

### Artifact Roundtrip

```text
train -> serialize -> new process load -> same vector -> same score/decision
```

## 9. Wazuh Ingestion Done Criteria

Historical ingestion tidak DONE sampai terbukti:

```text
daily index discovery
missing date handling
PIT per daily index
partial PIT rejected
search_after exact cursor
pagination >1 page
PIT close in finally
checkpoint resume
dedup on resume
```

Live ingestion tidak DONE sampai terbukti:

```text
no long-lived PIT tail
lookback overlap
late indexed fixture recovered
duplicate fixture ignored
daily rollover handled
```

## 10. Security Done Criteria

Agent harus verify:

```text
no credential in source
no token in fixture
no secret in logs
HTTP timeout configured
TLS policy explicit
least privilege documented
ASUS service not requiring public Wazuh 9200
```

Agent tidak boleh membuka firewall/port publik sebagai shortcut tanpa explicit operator approval and infrastructure authority.

## 11. Clean Code Done Criteria

Before DONE:

```text
one authoritative FEATURE_COLUMNS
one authoritative domain mapping
no global mutable EMA
no transport import inside RBTA core
no model fitting inside live runner
no stale output field referencing removed feature
no hardcoded workstation path in production code
```

Functions/classes should have one primary responsibility.

## 12. Legacy Removal Verification

Agent harus melakukan repository search dan melaporkan hasil.

Primary code path tidak boleh bergantung pada:

```text
CompoundMetaAlert
Bucket B
late_drop
HIGH_FREQ/LOW_FREQ step adaptation
SHRINK_RATE/EXPAND_RATE
compute_dynamic_contamination
ground_truth-driven contamination
11-feature primary vector
synthetic attack scenario A/B/C primary runner
```

Jika file archive tetap ada, pastikan tidak diimport oleh runtime.

## 13. Report Integrity

Agent dilarang menulis hasil tetap seperti:

```text
ARR = X%
R² = X
Silhouette = X
threshold = X
production ready = true
```

kecuali nilai dibaca/dihitung dari current run artifact.

Test fixture expected values boleh hardcoded jika jelas berada di test.

## 14. Operational State Recovery

Runtime state task tidak DONE sebelum test menunjukkan:

```text
process events
persist
simulate restart
reload
continue
no duplicate
same model version
EMA continuity preserved
active bucket continuity preserved according to lifecycle policy
```

## 15. Code Review Checklist

Agent self-review sebelum commit/sprint close:

- Apakah code mengikuti source-of-truth, bukan legacy?
- Apakah ada duplication baru?
- Apakah error path diuji?
- Apakah resource/network handle ditutup?
- Apakah retry dapat menyebabkan duplicate?
- Apakah timestamp wall-clock masuk ke research calculation secara tidak sengaja?
- Apakah model/scaler fit terjadi di replay/live?
- Apakah feature missing di-silent fallback?
- Apakah state per-agent benar-benar isolated?
- Apakah task menambahkan scope yang tidak diminta?

## 16. Stop Conditions

Agent wajib berhenti dan meminta keputusan jika:

```text
source-of-truth docs saling bertentangan
field Wazuh production berbeda sehingga research field tidak dapat dimapping aman
agent identity migration membutuhkan semantic decision
model artifact compatibility tidak dapat dibuktikan
external network policy tidak jelas
schema change akan mengubah tujuh feature
```

Jangan membuat asumsi metodologis sendiri.

## 17. Final Project Done

Project hanya boleh disebut research-core + operational production-ready setelah:

```text
S0-S11 gates passed
full pytest pass
historical ingest evidence pass
live ingest fixture + real authorized smoke test pass
ASUS restart recovery pass
model artifact validation pass
Shuffle end-to-end exactly-once logical delivery pass
no secrets findings
final docs/code consistency scan pass
```

Jika real live Wazuh smoke test belum dapat dilakukan karena agent kampus belum aktif atau jalur private belum tersedia, final status harus menyatakan hal itu secara eksplisit.