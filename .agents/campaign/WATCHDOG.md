# Optional Antigravity Campaign Watchdog

Use this only as a resume/check task, not as the primary development prompt.

Read `.agents/campaign/STATE.json` and inspect the repository.

- If status is `COMPLETE`, do nothing.
- If status is `METHODOLOGY_CONFLICT`, do not modify code; report the recorded conflict.
- If status is `BLOCKED_EXTERNAL`, check whether the external blocker is now resolvable without weakening security. If not, do nothing beyond reporting it.
- If status is `READY`, `IMPLEMENTING`, `SELF_REVIEW`, `VERIFYING`, or `REMEDIATING` and no active campaign work is progressing, resume the exact current sprint using `MASTER-GOAL.md`, `RULES.md`, and `SPRINT-N.md`.
- Never skip a failed or incomplete gate.
- Never start Sprint 10/11.
