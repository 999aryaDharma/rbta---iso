# Hybrid Lenovo–ASUS Runtime Implementation Plan

Date: 2026-09-07  
Design: `docs/superpowers/specs/2026-09-07-hybrid-lenovo-asus-runtime-design.md`

## 1. Corpus discovery and research ingestion

- Add failing tests for file/directory discovery, lexical ordering, `.meta` exclusion, gzip reading, malformed-line provenance, and duplicate IDs.
- Add a focused corpus loader used by the research orchestrator.
- Record per-file and combined provenance in the research manifest.
- Update CLI help and retain single-file compatibility.

## 2. Non-blocking replay catalog

- Add failing tests proving listing does not inspect uncached large files.
- Add fast pending manifests, thread-safe cache persistence, and background refresh status.
- Add controller and authenticated API contracts for refresh and status.
- Require a current catalog for `ALL`, while preserving lazy single-file inspection.

## 3. Demo user interface

- Add typed catalog status API hooks and polling.
- Show a clear indexing card, progress, current file, errors, and refresh action.
- Explain `.meta` exclusion and the difference between dataset indexing and replay evaluation.
- Disable full-corpus replay until indexing succeeds.

## 4. Lenovo deployment

- Add hardened `deploy/local/compose.yml` and `.env.example` with the supplied Windows dataset path.
- Add a cross-platform launcher with deterministic validation and post-start indexing.
- Add unit tests for Windows and WSL path handling plus supported-file discovery.

## 5. ASUS compatibility and operator documentation

- Accept `.jsonl.gz` consistently in container validation and ASUS preflight.
- Resolve the documented ASUS demo port mismatch.
- Add a hybrid runbook with exact PowerShell, WSL, and ASUS commands and model placement guidance.
- State the 60/20/20 temporal split and resource expectations explicitly.

## 6. Verification and delivery

- Run targeted tests during each red/green cycle.
- Run complete backend tests with coverage.
- Run frontend lint, typecheck, unit tests, production build, and dependency audit.
- Render Compose configuration if Docker is available; otherwise report the environment limitation precisely.
- Review the final diff, verify `main` is unchanged, commit, and push `prod/final-dashboard-demo`.
