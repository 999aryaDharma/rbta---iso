# Research Implementation Specification Index

Dokumentasi ini menyelaraskan codebase dengan metodologi seminar proposal dan menyiapkan core agar dapat digunakan dalam mode batch maupun live REST integration.

## Reading Order

1. [`00-SOURCE-OF-TRUTH.md`](./00-SOURCE-OF-TRUTH.md)
   - aturan tertinggi;
   - laporan seminar sebagai source of truth;
   - EMA/ETW wajib lokal per agent;
   - daftar legacy behavior yang harus dihapus.

2. [`01-DATA-PIPELINE-SPEC.md`](./01-DATA-PIPELINE-SPEC.md)
   - canonical Wazuh alert schema;
   - batch/live equivalence;
   - parsing, deduplication, MITRE, criticality.

3. [`02-RBTA-ALGORITHM-SPEC.md`](./02-RBTA-ALGORITHM-SPEC.md)
   - bucket `(agent_id, rule_group_primary)`;
   - per-agent warm-up, baseline, EMA, dan delta-t;
   - out-of-order tanpa late-drop;
   - mapping integrity.

4. [`03-FEATURE-ENGINEERING-SPEC.md`](./03-FEATURE-ENGINEERING-SPEC.md)
   - tepat tujuh feature final;
   - formula feature;
   - single feature contract.

5. [`04-ISOLATION-FOREST-SPEC.md`](./04-ISOLATION-FOREST-SPEC.md)
   - RobustScaler;
   - IF 200 trees / contamination auto;
   - Tukey IQR;
   - Decision Matrix dan False Positive Gate.

6. [`05-EVALUATION-SPEC.md`](./05-EVALUATION-SPEC.md)
   - sensitivity;
   - ARR;
   - noise robustness;
   - runtime;
   - Silhouette + 100 permutation baseline;
   - penghapusan synthetic ground-truth evaluation dari pipeline utama.

7. [`06-CODEBASE-REFACTOR-SPEC.md`](./06-CODEBASE-REFACTOR-SPEC.md)
   - target module boundaries;
   - DRY/single responsibility;
   - legacy cleanup.

8. [`07-IMPLEMENTATION-CHECKLIST.md`](./07-IMPLEMENTATION-CHECKLIST.md)
   - urutan implementasi dan acceptance gates untuk agent coding.

9. [`08-OPERATIONAL-INTEGRATION-SPEC.md`](./08-OPERATIONAL-INTEGRATION-SPEC.md)
   - REST API live ingress dari Wazuh;
   - persistent per-agent runtime state;
   - model artifacts;
   - Shuffle SOAR event push/custom node boundary;
   - Telegram routing;
   - outbox/idempotency dan end-to-end acceptance test.

## Core Rule

Jika ada konflik:

```text
Seminar report + explicit clarification
> docs/research-spec
> source code
> old comments/docstrings
```

## Important Architectural Boundary

```text
Research Core
Wazuh canonical alert
-> RBTA per-agent ETW
-> MetaAlert
-> 7 features
-> Isolation Forest
-> Decision Matrix

Operational Layer
REST ingress
-> Research Core
-> durable scored event/outbox
-> Shuffle SOAR
-> Telegram / downstream actions
```

Operational integration tidak boleh mengubah formula EMA, feature schema, IF methodology, atau evaluation definition.