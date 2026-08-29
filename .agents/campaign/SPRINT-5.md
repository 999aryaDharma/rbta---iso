# Sprint 5 Contract — Historical Wazuh Indexer Source

## Objective

Implement efficient, resumable historical Wazuh Indexer acquisition without weakening security.

## Required authority

Read `00-SOURCE-OF-TRUTH.md`, `01-DATA-PIPELINE-SPEC.md`, `09-WAZUH-INDEXER-INGESTION-SPEC.md`, `10-DUAL-MODE-RUNTIME-SPEC.md`, `13-ANTIGRAVITY-SPRINT-PLAN.md`, and `14-AGENT-DEFINITION-OF-DONE.md`.

## Deliverables

Implement source/client boundaries, daily index discovery, PIT-per-daily-index + `search_after`, and historical checkpoint/resume.

Required behavior: explicit connect/read timeouts; secure TLS default; credentials from env/config; redacted logging; retry only transient failures; 401/403 fail fast; discover real daily indices and accept missing dates; sorted ascending; PIT per daily index; partial PIT rejected; exact sort `@timestamp ASC, id ASC`; default page size 500; one transport request per page, not alert; exact last-hit `sort` cursor; PIT close in `finally`; persist index + last_sort + count + last Wazuh ID; resume with a new PIT; overlap/resume dedup prevents double core mutation.

If real Wazuh credentials/network are unavailable, fixture-backed implementation may proceed, but never expose Indexer publicly or disable security merely to produce evidence.

## Gate S5

Multi-page fixture exports every logical alert once, resume is duplicate-safe, PIT always closes, missing dates are safe, targeted/full regression pass, and any unavailable external verification is clearly marked `BLOCKED_EXTERNAL` rather than faked.
