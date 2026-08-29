# RBTA + Isolation Forest Security Log Alert Fatigue Mitigation

Research Implementation:
**Rule-Based Temporal Aggregation (RBTA) dan Isolation Forest untuk Mitigasi Alert Fatigue pada Log Keamanan SIEM Wazuh**

## Overview
This repository implements the research pipeline and dual-mode operational service combining agent-local Elastic Time Window (ETW) temporal aggregation with Isolation Forest contextual anomaly detection.

## Quickstart

### Running Tests
```bash
python -m pytest -v
```

### Production Deployment
See [`docs/deployment/ASUS.md`](docs/deployment/ASUS.md) for full deployment instructions, Docker Compose configuration, host persistence layout, and verification smoke testing on the ASUS server.
