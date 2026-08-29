# ASUS Server Deployment Guide — RBTA + Isolation Forest Service

## 1. Operational Overview & Phased Deployment Model

To guarantee methodological integrity and prevent speculative operational claims, deployment to the ASUS server is structured into four distinct phases:

```text
+-------------------------------------------------------------+
| Phase A: Repository & CI/CD Readiness (CURRENT: PASS)       |
|  - Full test suite passing (234+ tests)                     |
|  - Production Dockerfile (non-root UID 10001)               |
|  - Fail-closed Compose manifest (requires key & model)      |
|  - Automated preflight & smoke scripts verified             |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Phase B: Infrastructure & Wazuh Discovery (DEFERRED)        |
|  - Confirm ASUS host network route & firewall rules         |
|  - Confirm Wazuh Indexer URL, TLS certificates & creds      |
|  - Complete docs/deployment/WAZUH-LIVE-INTEGRATION-CHECKLIST|
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Phase C: Live ASUS Deployment Execution (DEFERRED)          |
|  - Deploy immutable container revision on ASUS              |
|  - Mount state directory (RW) and model directory (RO)      |
|  - Verify /health (200) and /ready (200)                    |
|  - Run authenticated smoke test & measure real resource use |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Phase D: Final S10 Production Gate Closeout (DEFERRED)      |
|  - Document live restart & force-recreate persistence       |
|  - Record actual ASUS memory, CPU, and startup latency      |
+-------------------------------------------------------------+
```

---

## 2. Directory Layout on ASUS Host

On the ASUS host machine:

```text
/srv/rbta-iso/
├── deploy/asus/
│    ├── compose.yml              <-- Production Docker Compose manifest
│    └── .env                     <-- Mandatory production environment (mode 0600)
├── state/                        <-- Host-mounted RW durable state
│    └── state.json
└── models/                       <-- Host-mounted RO model registry
     └── <model_version>/
          ├── isolation_forest.joblib
          ├── robust_scaler.joblib
          ├── score_calibration.json
          ├── threshold.json
          ├── feature_schema.json
          ├── metadata.json
          └── manifest.json
```

### Initial Host Directory Setup
```bash
mkdir -p /srv/rbta-iso/{state,models}
cd /srv/rbta-iso

# Configure non-root container ownership (UID/GID 10001:10001)
sudo chown -R 10001:10001 /srv/rbta-iso/state
sudo chmod 0750 /srv/rbta-iso/state
```

---

## 3. Mandatory Environment Configuration

Copy `.env.example` to `deploy/asus/.env`:

```bash
cp .env.example deploy/asus/.env
chmod 0600 deploy/asus/.env
```

| Variable | Requirement | Description | Example / Notes |
|---|---|---|---|
| `RBTA_API_KEY` | **MANDATORY** | Secret token for `/api/v1/*` & `/runtime/stats`. Fail-closed if empty. | Random 32+ char token |
| `RBTA_MODEL_VERSION` | **MANDATORY** | Explicit active model version subdirectory. Fail-closed if empty. | e.g. `seminar-model-v1` |
| `RBTA_HOST_PORT` | Configurable | Loopback port bound on host (Default: `8000`). | `8000` |
| `RBTA_LOG_LEVEL` | Configurable | Log verbosity (`DEBUG`, `INFO`, `WARNING`). | `INFO` |
| `RBTA_CPU_LIMIT` | Provisional | Maximum CPU allocation for container. | `2.00` |
| `RBTA_MEMORY_LIMIT`| Provisional | Maximum RAM allocation for container. | `1024M` |

---

## 4. Deployment Execution Sequence

### Step 1: Run Preflight Validation
```bash
bash scripts/deploy/asus-preflight.sh
```
*Validates environment variables, model bundle manifest checksums, state directory write permissions, and Docker Compose syntax before touching any running services.*

### Step 2: Deploy Container
```bash
bash scripts/deploy/asus-deploy.sh
```
*Builds container, launches via Compose in detached mode, and sequentially gates on `/health` (HTTP 200) followed by `/ready` (HTTP 200).*

### Step 3: Run Authenticated Smoke Test
```bash
export RBTA_API_KEY="<your-configured-api-key>"
bash scripts/deploy/smoke.sh
```

---

## 5. State Persistence & Restart Semantics

- **Durable Runtime Core**: Engine seen alert IDs, active ETW aggregation buckets, per-agent temporal state, and outbox queues are persisted to `/app/data/runtime/state.json` (mapped to host `./state/state.json`).
- **Crash & Restart Survival**:
  ```bash
  docker compose -f deploy/asus/compose.yml restart
  ```
  Active buckets and processed alert IDs survive untouched without generating synthetic forced MetaAlerts.
- **Container Re-creation**:
  ```bash
  docker compose -f deploy/asus/compose.yml up -d --force-recreate
  ```
  State remains intact on the host filesystem across complete container teardown.

---

## 6. Security Baseline

- **Non-Root Runtime**: Strictly runs as `appuser:appgroup` (`UID 10001:10001`).
- **Dropped Capabilities**: `cap_drop: ALL`, `no-new-privileges: true`.
- **Immutable Models**: Model directory mounted strictly read-only (`:ro`).
- **Network Isolation**: Binds strictly to host loopback (`127.0.0.1:${RBTA_HOST_PORT:-8000}`).
- **Bounded Logging**: Docker JSON logging capped at 10MB x 3 rotations.

---

## 7. Deferred Deployment Items

The following items are intentionally marked **DEFERRED** pending the future ASUS live operational session:

1. **Actual Wazuh Indexer Endpoint & Route**: Real campus IP/hostname, TLS CA bundle, and firewall access.
2. **Actual Wazuh Authentication**: Service account credentials for indexer access.
3. **Production Research Model Activation**: Final seminar model bundle published to registry.
4. **ASUS Real Hardware Resource Benchmarks**: Idle memory, active scoring CPU, and startup latency.
5. **Final S10 Gate Acceptance**: Only closed after actual Phase C deployment is completed on physical ASUS host.
