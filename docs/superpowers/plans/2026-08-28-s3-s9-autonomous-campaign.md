# S3-S9 Autonomous Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a persistent Antigravity campaign controller for sequential Sprint 3-9 execution with gate evidence and per-sprint external auditing.

**Architecture:** `.agents/campaign` is orchestration metadata only; it does not implement research algorithms. One controller advances a durable JSON state machine, reads a sprint-specific contract, delegates independent in-sprint tasks when useful, verifies the gate, records evidence, and proceeds. ChatGPT independently watches GitHub for newly completed sprint gates.

**Tech Stack:** Git, Markdown, JSON, Antigravity agent/goals, pytest, existing Python research repository.

**Spec:** `docs/superpowers/specs/2026-08-28-s3-s9-autonomous-campaign-design.md`

## Global Constraints

- Sprints S3-S9 are sequential; only in-sprint tasks may run in parallel.
- Existing research specifications remain authoritative; orchestration files may not redefine methodology.
- Gate evidence must reference the exact Code SHA tested and be a separate docs commit.
- Stop after Gate S9; do not start S10/S11.
- Never weaken external security to satisfy a gate.

---

### Task 1: Install persistent campaign control plane

**Files:** Create `.agents/campaign/README.md`, `RULES.md`, `STATE.json`, `MASTER-GOAL.md`, and `WATCHDOG.md`.

**Interfaces:** Consumes the passed S2 head and existing research specs. Produces one durable campaign state and one master AGY instruction.

- [ ] Validate `STATE.json` begins at `last_passed_gate=S2`, `current_sprint=3`, `status=READY`.
- [ ] Verify campaign rules enforce sequential gates, TDD, code-SHA evidence, no silent methodology changes, and stop after S9.
- [ ] Verify the master goal resumes from state rather than assuming a fresh run.
- [ ] Commit as orchestration-only metadata.

### Task 2: Bind S3-S9 to sprint-specific contracts

**Files:** Create `.agents/campaign/SPRINT-3.md` through `SPRINT-9.md`.

**Interfaces:** Each contract consumes the authoritative docs named in its Required authority section and produces Gate SN evidence only.

- [ ] Compare every contract against `13-ANTIGRAVITY-SPRINT-PLAN.md`.
- [ ] Ensure no sprint contract introduces later-sprint implementation.
- [ ] Ensure each gate names measurable acceptance criteria.
- [ ] Commit with Task 1 in the same orchestration package unless review requires separation.

### Task 3: Verify orchestration package

**Files:** No research source changes.

- [ ] Parse `.agents/campaign/STATE.json` with a standard JSON parser.
- [ ] Search orchestration files for `Sprint 10`/`Sprint 11` implementation instructions and confirm they occur only as explicit stop rules.
- [ ] Confirm campaign package contains no credentials, workstation paths, hardcoded research result claims, or commands that modify main automatically.
- [ ] Confirm existing S2 research source files are unchanged from the campaign base.
- [ ] Record the orchestration commit SHA and push the campaign branch.
