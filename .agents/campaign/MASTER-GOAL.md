# One-Shot Antigravity Goal — Execute Sprint 3 Through Sprint 9

Use this document as the body of a single `/goal` command.

---

Execute the RBTA + Isolation Forest research campaign from Sprint 3 through Sprint 9 in repository `999aryaDharma/rbta---iso`.

Treat `.agents/campaign/STATE.json` as persistent orchestration state, `.agents/campaign/RULES.md` as mandatory campaign policy, and `.agents/campaign/SPRINT-N.md` as the execution contract for the current sprint.

Before any code change, read the current state and the authoritative research documents required by that sprint. Never rely on legacy code when it contradicts the research specification.

For each sprint N from the current state through 9:

1. Confirm Gate N-1 is PASS and its evidence exists.
2. Set state to `IMPLEMENTING` and use a branch named `refactor/sprint-N-<short-scope>` based on the previous passed sprint/evidence head. Never overwrite a passed sprint branch.
3. Implement only Sprint N. Use TDD for every non-trivial behavior.
4. Use parallel subagents only for independent work inside Sprint N, such as implementation research, test design, code review, or governance review. Do not let multiple agents modify campaign state or advance gates.
5. Set state to `SELF_REVIEW`; review source-of-truth compliance, clean architecture, DRY violations, hidden global mutable state, duplicate algorithms, silent fallbacks, and scope creep.
6. Set state to `VERIFYING`; run all targeted tests, full pytest collect/run, `git diff --check`, and sprint-specific governance/invariant checks.
7. If verification fails, set state to `REMEDIATING`, fix within Sprint N, and repeat verification. Never advance with a failed test or unresolved Important/Critical review issue.
8. When all Sprint N acceptance criteria pass, finish code/test commits, record the exact Code SHA tested, create `docs/research-spec/evidence/sprint-N-gate.md`, commit evidence separately, push the branch, then update state with `last_passed_gate`, `last_completed_sprint`, Code SHA, evidence SHA, and next sprint.
9. Continue automatically with Sprint N+1. Do not ask the user for routine implementation confirmation.
10. Ask/stop only when authoritative sources contradict each other, a locked methodology change is necessary, a destructive infrastructure/security action would be required, or an external dependency prevents a mandatory gate. Record `METHODOLOGY_CONFLICT` or `BLOCKED_EXTERNAL` with exact reason.

Do not implement Sprint 10 or Sprint 11. Do not merge to main automatically.

When Gate S9 passes, set `status` to `COMPLETE` and stop. Produce a concise final campaign report linking each Sprint 3-9 branch, Code SHA tested, evidence SHA, and Gate result. Do not perform a substitute audit of earlier sprints; ChatGPT's scheduled audit process will independently audit each completed sprint.
