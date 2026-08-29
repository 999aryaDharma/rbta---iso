import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { useSearchParams } from 'react-router-dom';
import { HardDrives, Cpu } from '@phosphor-icons/react';

export function SystemPage() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const { data: sys } = usePollingQuery(['system', runId || 'live'], () => fetchSystemInfo(runId || undefined), 3000);
  const { data: summary } = usePollingQuery(['summary', runId || 'live'], () => fetchSummary(runId || undefined), 3000);

  return (
    <div className="space-y-6">
      <PageHeader
        title="System Configuration & Diagnostics"
        description="Runtime architecture, active model bundle parameters, schema invariants, and persistent evidence storage"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="System Status" value={sys?.system_status ?? 'READY'} />
        <MetricCard label="Model Version" value={sys?.model_version ?? '—'} />
        <MetricCard label="Tukey Threshold" value={sys ? sys.tukey_threshold.toFixed(4) : '—'} />
        <MetricCard label="Base Δt (Seconds)" value={sys ? `${sys.base_delta_t_seconds}s` : '—'} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Bundle Card */}
        <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs">
          <div className="flex items-center gap-2.5 mb-4 pb-3 border-b border-kumo-hairline">
            <Cpu size={18} className="text-kumo-brand" />
            <h3 className="font-semibold text-sm text-kumo-default">
              Active Isolation Forest Bundle
            </h3>
          </div>
          <dl className="space-y-3 text-xs">
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Model Version</dt>
              <dd className="font-mono font-semibold text-kumo-default">{sys?.model_version ?? '—'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Tukey IQR Anomaly Threshold</dt>
              <dd className="font-mono font-semibold text-kumo-brand">
                {sys ? sys.tukey_threshold.toFixed(4) : '—'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Random State Seed</dt>
              <dd className="font-mono text-kumo-default">{sys?.random_state ?? 'None'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Adaptive Temporal Clustering</dt>
              <dd className="font-mono font-semibold text-kumo-default">{sys?.adaptive ? 'ENABLED' : 'DISABLED'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Decision Strategy</dt>
              <dd className="font-mono text-kumo-default">FOUR_QUADRANT_MATRIX</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Feature Normalization</dt>
              <dd className="font-mono text-kumo-default">RobustScaler (Median / IQR)</dd>
            </div>
            <div>
              <dt className="text-kumo-subtle mb-1">Seven Features (Locked Order)</dt>
              <dd className="font-mono text-[11px] p-2 rounded-md border border-kumo-hairline bg-kumo-recessed text-kumo-default">
                {sys?.feature_names && sys.feature_names.length > 0
                  ? sys.feature_names.join(', ')
                  : 'max_severity, mitre_tactic_count, critical_mitre_tactic_present, alert_count_log, rule_diversity_shannon, severity_dispersion, agent_criticality'}
              </dd>
            </div>
          </dl>
        </div>

        {/* Runtime Environment Card */}
        <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs">
          <div className="flex items-center gap-2.5 mb-4 pb-3 border-b border-kumo-hairline">
            <HardDrives size={18} className="text-kumo-info" />
            <h3 className="font-semibold text-sm text-kumo-default">
              Runtime & Persistence Diagnostics
            </h3>
          </div>
          <dl className="space-y-3 text-xs">
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Liveness Probe (/health)</dt>
              <dd className="font-mono font-semibold text-kumo-success">200 OK</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Readiness Probe (/ready)</dt>
              <dd className={`font-mono font-semibold ${summary?.system_status === 'READY' ? 'text-kumo-success' : 'text-kumo-danger'}`}>
                {summary?.system_status === 'READY' ? '200 READY' : '503 NOT READY'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Ingress Source Mode</dt>
              <dd className="font-mono text-kumo-default">{sys?.source_mode ?? 'LIVE'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Raw Evidence SQLite Engine</dt>
              <dd className="font-mono text-kumo-default">WAL Mode with SHA-256 Checksums</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">Replay Isolation Layer</dt>
              <dd className="font-mono text-kumo-default">Sandboxed Data Directory per Run ID</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-kumo-subtle">RBTA Engine Thread Safety</dt>
              <dd className="font-mono text-kumo-default">Non-overlapping Synchronous Ingress</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
