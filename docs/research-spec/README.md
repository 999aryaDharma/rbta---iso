# Research Implementation Specification Index

Dokumentasi ini menyelaraskan codebase dengan metodologi seminar proposal dan menyiapkan core agar dapat digunakan dalam mode batch research, historical replay, dan live operational integration.

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
   - high-level implementation gates;
   - research and operational verification checklist.

9. [`08-OPERATIONAL-INTEGRATION-SPEC.md`](./08-OPERATIONAL-INTEGRATION-SPEC.md)
   - REST operational boundary;
   - persistent per-agent runtime state;
   - Shuffle custom app;
   - Telegram routing;
   - outbox/idempotency.

10. [`09-WAZUH-INDEXER-INGESTION-SPEC.md`](./09-WAZUH-INDEXER-INGESTION-SPEC.md)
    - historical daily-index discovery;
    - PIT + `search_after`;
    - checkpoint/resume;
    - local archive/replay;
    - live Indexer polling;
    - campus collector push;
    - historical-to-live handoff.

11. [`10-DUAL-MODE-RUNTIME-SPEC.md`](./10-DUAL-MODE-RUNTIME-SPEC.md)
    - `research-batch` dan `replay-stream`;
    - shared Research Core;
    - replay clock;
    - state recovery;
    - batch/replay equivalence.

12. [`11-PRODUCTION-READINESS-AUDIT.md`](./11-PRODUCTION-READINESS-AUDIT.md)
    - P0/P1/P2 audit current code;
    - algorithmic blockers;
    - ingestion/runtime gaps;
    - target production structure.

13. [`12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md`](./12-MODEL-ARTIFACT-LIFECYCLE-SPEC.md)
    - immutable model/scaler artifacts;
    - single-event-safe score calibration;
    - threshold persistence;
    - feature/model version validation.

14. [`13-ANTIGRAVITY-SPRINT-PLAN.md`](./13-ANTIGRAVITY-SPRINT-PLAN.md)
    - task-by-task development plan;
    - TDD gates;
    - commits and acceptance per sprint;
    - historical + live Wazuh implementation sequence.

15. [`14-AGENT-DEFINITION-OF-DONE.md`](./14-AGENT-DEFINITION-OF-DONE.md)
    - rules for coding-agent completion claims;
    - mandatory evidence;
    - stop conditions;
    - no-shortcut production/research verification.

## Core Rule

Jika ada konflik:

```text
Seminar report + explicit researcher clarification
> docs/research-spec
> source code
> old comments/docstrings
```

## Research / Operational Boundary

```text
                    +-------------------------+
Batch/Archive/API ->| Wazuh Canonicalizer     |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Shared Research Core    |
                    | RBTA per-agent ETW      |
                    | -> MetaAlert            |
                    | -> exactly 7 features   |
                    | -> Isolation Forest     |
                    | -> Decision Matrix      |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Operational Layer       |
                    | state / API / outbox    |
                    | Shuffle -> Telegram     |
                    +-------------------------+
```

Historical replay, live polling, REST transport, persistence, dashboard, Shuffle, dan Telegram tidak boleh mengubah formula EMA, feature schema, IF methodology, atau evaluation definition.

## Development Rule for Coding Agents

Coding agent harus mulai dari:

```text
00-SOURCE-OF-TRUTH.md
13-ANTIGRAVITY-SPRINT-PLAN.md
14-AGENT-DEFINITION-OF-DONE.md
```

Lalu membaca spec yang relevan dengan sprint aktif.

Jangan menjalankan sprint berikutnya sebelum acceptance gate sprint aktif lulus dengan test evidence.