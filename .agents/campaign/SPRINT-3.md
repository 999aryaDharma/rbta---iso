# Sprint 3 Contract — Exact Seven-Feature Extractor

## Objective

Replace all legacy feature vectors with exactly seven source-of-truth features.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `03-FEATURE-ENGINEERING-SPEC.md`, `06-CODEBASE-REFACTOR-SPEC.md`, `07-IMPLEMENTATION-CHECKLIST.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Create `src/features/extractor.py` and `tests/unit/features/test_extractor.py`.

The single authoritative feature schema is exactly:

```python
FEATURE_COLUMNS = [
    "max_severity",
    "mitre_tactic_count",
    "critical_mitre_tactic_present",
    "alert_count_log",
    "rule_diversity_shannon",
    "severity_dispersion",
    "agent_criticality",
]
```

Required tests: singleton entropy 0; same-rule repeated entropy 0; balanced two-rule normalized entropy approximately 1; singleton severity dispersion 0; MITRE tactics count uniquely; critical flag correct; `alert_count_log == log1p(alert_count)`; output exactly seven columns in order; missing required aggregate raises clear error with no silent zero-fill.

Remove/archive active duplicate feature-vector logic so there is only one authoritative `FEATURE_COLUMNS`. Do not implement IF training yet.

## Gate S3

All seven-feature tests and full regression pass; repository governance proves only one active authoritative feature schema and primary runtime does not independently recalculate another vector.
