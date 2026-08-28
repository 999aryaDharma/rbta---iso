# Campaign Rules S3-S9

## Authority

Read and obey, in order:

1. `Seminar fix.pdf` where available to the agent;
2. explicit researcher clarifications already captured in `docs/research-spec/00-SOURCE-OF-TRUTH.md`;
3. `docs/research-spec/00-SOURCE-OF-TRUTH.md` through `14-AGENT-DEFINITION-OF-DONE.md`;
4. current authoritative code;
5. legacy comments/artifacts.

Never invent a compromise when higher-authority sources conflict. Stop as `METHODOLOGY_CONFLICT` with exact file/section references.

## Sequential gates

Never implement Sprint N+1 before Gate N is PASS and evidence is committed. Sprints are dependency-ordered; do not develop S3-S9 concurrently.

Parallel subagents may work only on independent tasks inside the current sprint. Only one controller owns branch advancement, gate decisions, evidence, and `STATE.json`.

## TDD and verification

For every non-trivial behavior:

1. write a failing test;
2. run it and observe the expected failure;
3. implement the minimum correct behavior;
4. run targeted tests;
5. run affected regression tests;
6. self-review;
7. commit a green unit of work.

Before Gate PASS always run:

- `git diff --check`;
- all sprint-targeted tests;
- `python -m pytest --collect-only -q`;
- `python -m pytest -q`;
- sprint-specific governance searches.

Never claim PASS from partial tests or from assumptions.

## Evidence discipline

For each sprint create/update:

`docs/research-spec/evidence/sprint-N-gate.md`

Evidence must include:

- branch;
- base SHA;
- exact Code SHA tested;
- environment/runtime;
- files changed;
- exact commands and actual results;
- sprint-specific measured invariants;
- known limitations/external blockers;
- gate decision.

Finish code/tests first, commit them, record `CODE_SHA`, run final verification against that SHA, then write a docs-only evidence commit. Never treat the docs-only evidence commit as the Code SHA tested.

## Locked research methodology

Do not silently change:

- RBTA per-agent EMA semantics established by Gate S2;
- seven feature names/order;
- IF estimators, contamination mode, scaler requirement, calibration semantics;
- Tukey formula;
- Decision Matrix / False Positive Gate;
- Wazuh event-time semantics;
- evaluation sensitivity/noise/permutation values;
- batch/replay/live shared-core requirement.

## No silent fallback

Do not silently zero-fill missing required features, fit a replacement model, recalibrate on live requests, reset EMA after errors, skip failed required schema, ignore partial PIT, ignore state persistence failure, fabricate evidence, or hardcode research results.

## External integration

Missing external credentials/network is not permission to expose Wazuh publicly or weaken TLS. Implement and test fixtures/stubs, mark only the external evidence `BLOCKED_EXTERNAL`, and continue only when the sprint gate permits fixture-backed completion.

## Scope

Stop after Gate S9. Do not start S10 deployment or S11 dashboard.
