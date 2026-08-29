# RBTA + Isolation Forest Security Log Alert Fatigue Mitigation

Research Implementation:
**Rule-Based Temporal Aggregation (RBTA) dan Isolation Forest untuk Mitigasi Alert Fatigue pada Log Keamanan SIEM Wazuh**

## Operational & Deployment Status

- **Research Core & Evaluation (S1–S9)**: Verified and audited.
- **Pre-Deployment CI/CD Tooling (S10)**: Prepared and automated.
- **GitHub Actions CI**: Active (full test regression, Compose validation, clean Docker build, non-root smoke).
- **Actual ASUS Server Deployment**: **DEFERRED** (pending campus Wazuh network route confirmation and deployment session).

## Quickstart

### Running Full Test Suite
```bash
python -m pytest -v
```

### Pre-Deployment & Operations Documentation
- [`docs/deployment/ASUS.md`](docs/deployment/ASUS.md): Phased ASUS server deployment guide.
- [`docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md`](docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST.md): Operational parameter checklist for live Wazuh connection.
- [`docs/deployment/CD-FUTURE-CONTRACT.md`](docs/deployment/CD-FUTURE-CONTRACT.md): Future continuous deployment workflow specification.
