import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { useSearchParams } from 'react-router-dom';
import { Server, Cpu } from 'lucide-react';

export function SystemPage() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const { data: sys } = usePollingQuery(['system', runId || 'live'], () => fetchSystemInfo(runId || undefined), 3000);
  const { data: summary } = usePollingQuery(['summary', runId || 'live'], () => fetchSummary(runId || undefined), 3000);

  return (
    <div>
      <PageHeader
        title="System Configuration & Diagnostics"
        description="Runtime architecture, active model bundle parameters, schema invariants, and persistent evidence storage"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="System Status" value={sys?.system_status ?? 'READY'} />
        <MetricCard label="Model Version" value={sys?.model_version ?? '—'} />
        <MetricCard label="Tukey Threshold" value={sys ? sys.tukey_threshold.toFixed(4) : '—'} />
        <MetricCard label="Base Δt (Seconds)" value={sys ? `${sys.base_delta_t_seconds}s` : '—'} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Bundle Card */}
        <div
          className="p-5 rounded-[7px] border"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="flex items-center gap-2.5 mb-4 pb-3 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <Cpu size={18} style={{ color: 'var(--brand-orange)' }} />
            <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
              Active Isolation Forest Bundle
            </h3>
          </div>
          <dl className="space-y-3 text-xs">
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Model Version</dt>
              <dd className="font-mono font-semibold">{sys?.model_version ?? '—'}</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Tukey IQR Anomaly Threshold</dt>
              <dd className="font-mono font-semibold" style={{ color: 'var(--brand-orange)' }}>
                {sys ? sys.tukey_threshold.toFixed(4) : '—'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Random State Seed</dt>
              <dd className="font-mono">{sys?.random_state ?? 'None'}</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Adaptive Temporal Clustering</dt>
              <dd className="font-mono font-semibold">{sys?.adaptive ? 'ENABLED' : 'DISABLED'}</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Decision Strategy</dt>
              <dd className="font-mono">FOUR_QUADRANT_MATRIX</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Feature Normalization</dt>
              <dd className="font-mono">RobustScaler (Median / IQR)</dd>
            </div>
            <div>
              <dt style={{ color: 'var(--text-tertiary)' }} className="mb-1">Seven Features (Locked Order)</dt>
              <dd className="font-mono text-[11px] p-2 rounded border bg-[var(--bg-subtle)]" style={{ borderColor: 'var(--border-subtle)' }}>
                {sys?.feature_names && sys.feature_names.length > 0
                  ? sys.feature_names.join(', ')
                  : 'max_severity, mitre_tactic_count, critical_mitre_tactic_present, alert_count_log, rule_diversity_shannon, severity_dispersion, agent_criticality'}
              </dd>
            </div>
          </dl>
        </div>

        {/* Runtime Environment Card */}
        <div
          className="p-5 rounded-[7px] border"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="flex items-center gap-2.5 mb-4 pb-3 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <Server size={18} style={{ color: 'var(--action-blue)' }} />
            <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
              Runtime & Persistence Diagnostics
            </h3>
          </div>
          <dl className="space-y-3 text-xs">
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Liveness Probe (/health)</dt>
              <dd className="font-mono font-semibold" style={{ color: 'var(--success)' }}>200 OK</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Readiness Probe (/ready)</dt>
              <dd className="font-mono font-semibold" style={{ color: summary?.system_status === 'READY' ? 'var(--success)' : 'var(--danger)' }}>
                {summary?.system_status === 'READY' ? '200 READY' : '503 NOT READY'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Source Mode</dt>
              <dd className="font-mono font-semibold">{sys?.source_mode ?? summary?.source_mode ?? 'LIVE'}</dd>
            </div>
            <div>
              <dt style={{ color: 'var(--text-tertiary)' }} className="mb-1">Durable State Path</dt>
              <dd className="font-mono text-[11px] p-2 rounded border truncate bg-[var(--bg-subtle)]" style={{ borderColor: 'var(--border-subtle)' }}>
                {sys?.durable_state_path ?? '—'}
              </dd>
            </div>
            <div>
              <dt style={{ color: 'var(--text-tertiary)' }} className="mb-1">Raw Evidence SQLite DB</dt>
              <dd className="font-mono text-[11px] p-2 rounded border truncate bg-[var(--bg-subtle)]" style={{ borderColor: 'var(--border-subtle)' }}>
                {sys?.raw_evidence_db_path ?? 'SQLite (WAL mode enabled)'}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
