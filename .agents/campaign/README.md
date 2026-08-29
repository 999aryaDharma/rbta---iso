# Autonomous Research Campaign S3-S9

This directory is the persistent control plane for Antigravity development from Sprint 3 through Sprint 9.

## Start

Run the contents of `MASTER-GOAL.md` once as the Antigravity `/goal` instruction while checked out on the campaign branch.

The controller must read `STATE.json`, `RULES.md`, the current `SPRINT-N.md`, and the authoritative research specification before modifying code.

## Execution model

Sprint order is strictly sequential:

`S3 -> S4 -> S5 -> S6 -> S7 -> S8 -> S9`

Parallel subagents are allowed only inside the current sprint for independent implementation, review, test, and evidence work. Only the campaign controller may update `STATE.json` or decide a gate.

## Gate rule

A sprint advances only after:

1. its required implementation is complete;
2. targeted tests pass;
3. the full regression suite passes;
4. the sprint-specific gate passes;
5. code SHA tested is recorded;
6. gate evidence is committed separately;
7. the branch is pushed;
8. `STATE.json` is updated.

If a gate fails, remain on that sprint and remediate. If authoritative documents conflict, stop with `METHODOLOGY_CONFLICT`. If external credentials/network block only external integration, continue fixture-backed implementation and record `BLOCKED_EXTERNAL` without weakening security.

## Human/ChatGPT audit handoff

After AGY records a sprint Gate PASS and pushes it, ChatGPT's scheduled watcher audits that sprint only. The AGY campaign does not need to wait for ChatGPT unless the repository contains an explicit audit-block marker or the researcher manually pauses the campaign.

## Files

- `MASTER-GOAL.md`: one-shot Antigravity campaign instruction.
- `RULES.md`: invariant and stop rules.
- `STATE.json`: durable machine-readable campaign state.
- `SPRINT-3.md` ... `SPRINT-9.md`: sprint execution contracts derived from the authoritative sprint plan.
- `WATCHDOG.md`: optional Antigravity resume/watchdog prompt.
