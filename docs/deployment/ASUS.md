# ASUS Server Deployment Guide — RBTA + Isolation Forest Service

## 1. Architecture Overview

The RBTA + Isolation Forest operational service runs as a containerized FastAPI application via Docker Compose on the ASUS server:

```text
Host ASUS Server
 └── /srv/rbta-iso/
      ├── deploy/asus/compose.yml
      ├── .env
      ├── state/                      <-- Host-mounted RW durable state
      │    └── state.json
      └── models/                     <-- Host-mounted RO model registry
           └── <model_version>/
                ├── isolation_forest.joblib
                ├── robust_scaler.joblib
                ├── score_calibration.json
                ├── threshold.json
                ├── feature_schema.json
                ├── metadata.json
                └── manifest.json
```

---

## 2. Directory Layout & Setup

On the ASUS server:

```bash
# 1. Create isolated deployment root
mkdir -p /srv/rbta-iso/{state,models}
cd /srv/rbta-iso

# 2. Ensure non-root container user (UID 10001) can write to state directory
chmod 775 state
```

---

## 3. Environment Configuration

Copy `.env.example` to `.env` in the deployment directory:

```bash
cp .env.example .env
```

Configure the following variables:

| Variable | Description | Default / Example |
|---|---|---|
| `RBTA_API_KEY` | Secret API key for `/api/v1/*` endpoints | Random 32+ char token |
| `RBTA_MODEL_VERSION` | Explicit active model version folder | `deploy-smoke-v1` |
| `RBTA_HOST_PORT` | Local host port bound on loopback | `8000` |
| `RBTA_LOG_LEVEL` | Application logging level | `INFO` |

---

## 4. Model Artifact Preparation

Before starting the service, ensure an immutable model version bundle is present in `models/<model_version>`:

```text
models/deploy-smoke-v1/
├── isolation_forest.joblib
├── robust_scaler.joblib
├── score_calibration.json
├── threshold.json
├── feature_schema.json
├── metadata.json
└── manifest.json
```

---

## 5. Deployment Commands

### Initial Deployment / Update
```bash
# Build and start container in detached mode
docker compose -f deploy/asus/compose.yml build
docker compose -f deploy/asus/compose.yml up -d
```

### Automated Deployment Script
```bash
bash scripts/deploy/asus-deploy.sh
```

---

## 6. Probes & Health Verification

- **Liveness Probe**: `GET http://127.0.0.1:8000/health` (Returns HTTP 200 when container is responsive)
- **Readiness Probe**: `GET http://127.0.0.1:8000/ready` (Returns HTTP 200 when valid active model bundle is loaded; HTTP 503 if missing or invalid)
- **Runtime Stats**: `GET http://127.0.0.1:8000/runtime/stats` (Requires `Authorization: Bearer <RBTA_API_KEY>`)

Run verification smoke:
```bash
bash scripts/deploy/smoke.sh
```

---

## 7. State Persistence & Restart Semantics

- **Crash & Restart Survival**: The container stores `seen_alert_ids`, active RBTA buckets, per-agent temporal state, and outbox queues in `/app/data/runtime/state.json` (mounted to host `./state/state.json`).
- **Restart**:
  ```bash
  docker compose -f deploy/asus/compose.yml restart
  ```
  Active buckets and processed IDs are preserved without creating synthetic forced MetaAlerts.
- **Recreation**:
  ```bash
  docker compose -f deploy/asus/compose.yml up -d --force-recreate
  ```
  State persists on the host filesystem across container teardown and recreation.

---

## 8. Rollback & Backup

### State Snapshot Before Upgrade
```bash
cp /srv/rbta-iso/state/state.json /srv/rbta-iso/state/state.json.bak.$(date +%Y%m%d%H%M%S)
```

### Rollback Procedure
```bash
# 1. Stop current container
docker compose -f deploy/asus/compose.yml down

# 2. Checkout previous Git revision or switch image tag
git checkout <PREVIOUS_COMMIT_SHA>

# 3. Rebuild and launch
docker compose -f deploy/asus/compose.yml up -d --build
```

---

## 9. Security Baseline

- **Non-Root Execution**: Runs strictly as `appuser:appgroup` (`UID 10001:10001`).
- **Capability Dropping**: `cap_drop: ALL`, `no-new-privileges: true`.
- **Read-Only Models**: `./models` is mounted read-only (`:ro`).
- **Network Isolation**: Binds exclusively to host loopback (`127.0.0.1:8000`).
- **Bounded Logs**: Docker JSON logging capped at 10MB x 3 rotations.
