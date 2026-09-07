# RBTA + Isolation Forest Security Log Alert Fatigue Mitigation

Research Implementation:
**Rule-Based Temporal Aggregation (RBTA) dan Isolation Forest untuk Mitigasi Alert Fatigue pada Log Keamanan SIEM Wazuh**

## Operational & Deployment Status

- **Research Core & Evaluation (S1–S9)**: Verified and audited.
- **Pre-Deployment CI/CD Tooling (S10)**: Prepared and automated.
- **GitHub Actions CI**: Active (full test regression, Compose validation, clean Docker build, non-root smoke).
- **Thesis Demo UI**: `/dashboard/demo` with live RBTA/model telemetry and post-replay evaluation artifacts.
- **ASUS Demo Port**: host port `8010` (container port `8000`).
- **Hybrid Runtime**: the same frozen model and read-only replay corpus can run on Lenovo/Docker Desktop or ASUS/Linux.
- **External Wazuh / Shuffle / Telegram**: deferred; not represented as validated by the historical replay demo.

## Quickstart

### Running Full Test Suite
```bash
python -m pytest -v
```

For the defense, freeze the exact Git SHA, model artifact, and a manifest-backed golden dataset. See [`docs/demo/SIDANG-RUNBOOK.md`](docs/demo/SIDANG-RUNBOOK.md) and [`docs/thesis/BAB-IV-V-KERANGKA.md`](docs/thesis/BAB-IV-V-KERANGKA.md).

### Lenovo local demo

Copy `deploy/local/.env.example` to `deploy/local/.env`, verify the replay/model paths, then run:

```powershell
py scripts/deploy/local.py up
```

The launcher starts the service and pre-indexes the daily Wazuh exports without reading `.meta` sidecars as alerts. See [`docs/deployment/HYBRID-LENOVO-ASUS.md`](docs/deployment/HYBRID-LENOVO-ASUS.md).

### Pre-Deployment & Operations Documentation
- [`docs/deployment/ASUS.md`](docs/deployment/ASUS.md): Phased ASUS server deployment guide.
- [`docs/deployment/HYBRID-LENOVO-ASUS.md`](docs/deployment/HYBRID-LENOVO-ASUS.md): PowerShell, WSL2, and ASUS hybrid workflow.
- [`docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md`](docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md): Operational parameter checklist for live Wazuh connection.
- [`docs/deployment/CD-FUTURE-CONTRACT.md`](docs/deployment/CD-FUTURE-CONTRACT.md): Future continuous deployment workflow specification.
