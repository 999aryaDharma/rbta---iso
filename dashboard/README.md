# RBTA Security Analytics Dashboard

Operational demonstration, monitoring, and raw alert investigation layer for:
**Rule-Based Temporal Aggregation and Isolation Forest for Wazuh SIEM Security Logs**

---

## 1. Architecture & Design

The dashboard is built following Cloudflare-inspired infrastructure dashboard principles:
- **Light-first design**: Clean white and warm-neutral surfaces (\#f7f7f8\ app background, \#ffffff\ card surfaces).
- **Subtle borders**: Thin 1px borders (\#e2e5e9\) with clean 7px card corner radius.
- **Accents**: Brand orange (\#f6821f\) for identity/active states, Action blue (\#0055dc\) for primary actions.
- **No browser-side research metric calculation**: All metrics (ARR, 7 features, anomaly scores, Tukey IQR thresholds, quadrants, actions) are authoritatively computed by the backend.

---

## 2. Tech Stack

- **Framework**: React 18+ with TypeScript (strict mode)
- **Bundler & Dev Server**: Vite 5
- **Styling**: Tailwind CSS v4 with CSS custom properties
- **Data Fetching & Caching**: TanStack Query v5 (polling interval model)
- **Table Controls**: TanStack Table v8
- **Visualizations**: Recharts
- **Icons**: Lucide React
- **Runtime Validation**: Zod

---

## 3. Directory Structure

\dashboard/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
└── src/
    ├── api/
    │   ├── client.ts         # Authenticated fetch client with sessionStorage bearer token
    │   ├── dashboard.ts      # Summary, timeseries, system, agents, buckets queries
    │   ├── metaAlerts.ts     # MetaAlerts list, detail, and trace queries
    │   ├── rawAlerts.ts      # Raw alerts pagination and single alert detail queries
    │   ├── replay.ts         # Demonstration replay lifecycle controls
    │   └── schemas.ts        # Strict Zod schemas for all DTOs
    ├── app/
    │   └── App.tsx           # Router with AuthGate and route definitions
    ├── components/
    │   └── shared/
    │       ├── AppShell.tsx       # Root layout with Sidebar and Topbar
    │       ├── AuthGate.tsx       # SessionStorage API key authentication barrier
    │       ├── DecisionBadge.tsx  # Color-coded action & decision indicator
    │       ├── MetricCard.tsx     # Standard KPI metric card
    │       ├── PageHeader.tsx     # Standard header with breadcrumbs and actions
    │       ├── Sidebar.tsx        # Navigation sidebar with categorized groups
    │       └── Topbar.tsx         # Live status indicator and model metadata bar
    ├── features/
    │   ├── integrations/     # End-to-end pipeline visualization
    │   ├── meta-alerts/      # MetaAlerts list & deep detail inspection
    │   ├── overview/         # Executive KPIs, time series reduction, recent events
    │   ├── raw-alerts/       # Raw alert member grid & JSON viewer
    │   ├── rbta/             # Per-agent temporal states & active buckets
    │   ├── replay/           # Replay demonstration controller & telemetry
    │   └── system/           # Health, runtime configuration & schema versions
    ├── hooks/
    │   └── usePolling.ts     # Polling query wrapper for auto-refresh
    ├── lib/
    │   ├── auth.ts           # SessionStorage token accessor
    │   └── utils.ts          # Formatting utilities (numbers, percentages, dates)
    ├── styles/
    │   └── index.css         # Theme variables and global stylesheet
    └── main.tsx              # Application entrypoint
\
---

## 4. Authentication

- API Key is stored exclusively in \sessionStorage\ (bta.dashboard.apiKey\).
- It is never written to \localStorage\, cookies, or compiled into the client bundle.
- Any ā Unauthorized\ response immediately purges the stored key and redirects to the login barrier.

---

## 5. Running Locally

### Development Mode:
\\ash
cd dashboard
npm install
npm run dev
\Vite dev server starts on port ŏ3\ and proxies \/api\, \/health\, \/ready\ to \http://localhost:8000\.

### Production Build:
\\ash
npm run typecheck
npm run build
\Generates production assets in \dashboard/dist/\.

---

## 6. Docker & Production Serving

In production, the dashboard is served directly by the FastAPI application:
- Built in Stage 1 of the multi-stage \Dockerfile\.
- Copied to \/app/dashboard/dist\ in Stage 2.
- Mounted at \/dashboard\ with SPA fallback routing.
- Navigating to \/\ redirects to \/dashboard/\.
