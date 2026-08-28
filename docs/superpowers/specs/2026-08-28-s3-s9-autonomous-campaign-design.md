# S3-S9 Autonomous Campaign Design

## Goal

Allow Antigravity to develop Sprint 3 through Sprint 9 without manual per-sprint prompts while preserving sequential research gates and enabling independent ChatGPT audit after each completed sprint.

## Architecture

The repository contains a persistent campaign control plane under `.agents/campaign/`. `STATE.json` records the current sprint and campaign status. One Antigravity controller owns state transitions and gate advancement. Sprint contracts bind each iteration to the existing research specifications. Parallel subagents are allowed only inside the current sprint.

The development dependency graph is sequential across sprints, so S3-S9 are not executed concurrently. A failed gate returns the controller to remediation for that sprint. The controller stops only for a genuine methodology conflict, a mandatory external blocker, or after Gate S9.

ChatGPT runs independently as a condition watcher against GitHub. When a newly completed Sprint N gate is detected, it audits Sprint N only against the previous passed baseline, its source-of-truth documents, code, tests, and gate evidence. It does not wait until S9 and does not repeatedly perform a full S0-N audit unless a cross-sprint regression makes that necessary.

## State transitions

`READY -> IMPLEMENTING -> SELF_REVIEW -> VERIFYING -> GATE_PASS -> READY(next sprint)`

Verification failure becomes `REMEDIATING -> VERIFYING`. External blockers use `BLOCKED_EXTERNAL`; source-of-truth contradictions use `METHODOLOGY_CONFLICT`; successful S9 completion uses `COMPLETE`.

## Safety and research integrity

The controller may not change locked methodology silently, fabricate evidence, weaken external security to obtain integration evidence, duplicate the Research Core for different modes, or advance with failed regression tests. Evidence is tied to an exact tested Code SHA and committed separately.

## Audit integration

The scheduled ChatGPT watcher polls no more frequently than hourly. It should notify only when it detects a newly completed S3-S9 gate requiring audit or a material blocker. Each audit reports PASS/FAIL for that sprint and identifies remediation before the next audit notification.
