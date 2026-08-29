import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber } from '@/lib/utils';
import { Server, Cpu } from 'lucide-react';

export function SystemPage() {
  const { data: sys } = usePollingQuery(['system'], fetchSystemInfo, 3000);
  const { data: summary } = usePollingQuery(['summary'], fetchSummary, 3000);

  return (
    <div>
      <PageHeader
        title="System Information"
        description="Runtime architecture, active model bundle parameters, schema invariants, and endpoint health"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="API Status" value={sys?.api_status ?? '—'} />
        <MetricCard label="Model Version" value={sys?.model_version ?? '—'} />
        <MetricCard label="Feature Schema" value={sys?.feature_schema_version ?? '—'} />
        <MetricCard label="Tukey Threshold" value={sys ? sys.threshold.toFixed(4) : '—'} />
      </div>

      <div className="grid grid-cols-2 gap-6">
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
              <dt style={{ color: 'var(--text-tertiary)' }}>Feature Schema Version</dt>
              <dd className="font-mono">{sys?.feature_schema_version ?? '—'}</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Tukey IQR Anomaly Threshold</dt>
              <dd className="font-mono font-semibold" style={{ color: 'var(--brand-orange)' }}>
                {sys ? sys.threshold.toFixed(4) : '—'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Decision Strategy</dt>
              <dd className="font-mono">FOUR_QUADRANT_MATRIX</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Feature Normalization</dt>
              <dd className="font-mono">RobustScaler (Median / IQR)</dd>
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
              Runtime & Persistence Health
            </h3>
          </div>
          <dl className="space-y-3 text-xs">
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Liveness Probe (/health)</dt>
              <dd className="font-mono font-semibold" style={{ color: 'var(--success)' }}>200 OK</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Readiness Probe (/ready)</dt>
              <dd className="font-mono font-semibold" style={{ color: summary?.ready ? 'var(--success)' : 'var(--danger)' }}>
                {summary?.ready ? '200 READY' : '503 NOT READY'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Source Mode</dt>
              <dd className="font-mono">{summary?.source_mode ?? 'STANDALONE'}</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Raw Alert Evidence DB</dt>
              <dd className="font-mono">SQLite (WAL mode enabled)</dd>
            </div>
            <div className="flex justify-between">
              <dt style={{ color: 'var(--text-tertiary)' }}>Outbox Queue Depth</dt>
              <dd className="font-mono">{summary ? formatNumber(summary.outbox_depth) : 0}</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
