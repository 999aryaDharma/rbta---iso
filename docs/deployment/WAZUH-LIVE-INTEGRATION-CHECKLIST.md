# Wazuh Live Integration Operational Checklist

This checklist tracks infrastructure and network parameters for the future live deployment session connecting the ASUS deployment instance to the campus Wazuh SIEM cluster.

> **Current Status:** All live integration fields remain **UNKNOWN** and **DEFERRED** until the live deployment session. No values are assumed or hardcoded into the application codebase.

---

## 1. Network & Routing Configuration

| Parameter | Status | Value | Evidence / Notes |
|---|---|---|---|
| **Live Source Ingestion Mode** | `UNKNOWN` | | Option A: Indexer Poller, Option B: Collector Ingress |
| **Wazuh Indexer Base URL** | `UNKNOWN` | | e.g. `https://<ip-or-domain>:9200` |
| **Network Path from ASUS Server** | `UNKNOWN` | | Direct LAN, VLAN routing, Tailscale, WireGuard, or SSH Tunnel |
| **Campus Firewall / ACL Openings** | `UNKNOWN` | | Port 9200 / Port 55000 firewall rules confirmed |
| **VPN / Overlay Tunnel Requirements** | `UNKNOWN` | | Tailscale/WireGuard service active on ASUS |
| **Request Timeout & TCP Keepalive** | `UNKNOWN` | | Default 30s timeout recommended |

---

## 2. Authentication, TLS & Security

| Parameter | Status | Value | Evidence / Notes |
|---|---|---|---|
| **Authentication Mechanism** | `UNKNOWN` | | Basic Auth, Client Certs, or API Token |
| **Wazuh Service Account Username** | `UNKNOWN` | | Read-only security log analyst account |
| **Wazuh Service Account Password** | `UNKNOWN` | | Injected securely via `.env` / secret store |
| **TLS Certificate Verification Policy** | `UNKNOWN` | | Strict CA verification / Custom root CA |
| **Custom CA / Certificate Path** | `UNKNOWN` | | Path to mounted PEM certificate on ASUS |
| **Credential Storage Location** | `UNKNOWN` | | `/srv/rbta-iso/deploy/asus/.env` (mode `0600`) |

---

## 3. Indexer Schema & Data Retention

| Parameter | Status | Value | Evidence / Notes |
|---|---|---|---|
| **Actual Wazuh / OpenSearch Version** | `UNKNOWN` | | e.g. Wazuh 4.x / OpenSearch 2.x |
| **Daily Index Naming Pattern** | `UNKNOWN` | | e.g. `wazuh-alerts-4.x-YYYY.MM.DD` |
| **Index Retention Period (Days)** | `UNKNOWN` | | Daily indices retention on Indexer |
| **Minimum Required RBAC Permissions** | `UNKNOWN` | | `read` and `indices:data/read/search` on `wazuh-alerts-*` |
| **Sample Real Search Hit Payload** | `UNKNOWN` | | To be captured and verified against `CanonicalRawAlert` schema |

---

## 4. Operational Schedules & Monitoring

| Parameter | Status | Value | Evidence / Notes |
|---|---|---|---|
| **Live Fast Poll Interval** | `UNKNOWN` | | Default 5 seconds |
| **Recent Reconciliation Interval** | `UNKNOWN` | | Default 60 seconds (lookback 15m) |
| **Full Retention Reconciliation Schedule** | `UNKNOWN` | | Default daily cron (midnight sweep) |
| **Failure & Dead Letter Notification** | `UNKNOWN` | | Outbox alert to SOC / Telegram channel |
| **Process / Service Recovery Mode** | `UNKNOWN` | | Systemd / Docker Compose auto-restart |

---

## 5. Live Deployment Session Verification Gate

- [ ] Network ping and port 9200 reachability from ASUS container.
- [ ] TLS handshake succeeds with specified CA bundle.
- [ ] Read-only authentication verified against Wazuh Indexer.
- [ ] Search query returns actual daily index alert documents.
- [ ] `Canonicalizer` cleanly processes live alerts without schema validation errors.
- [ ] Reconciler scans retained daily indices and detects zero dropped alerts.
