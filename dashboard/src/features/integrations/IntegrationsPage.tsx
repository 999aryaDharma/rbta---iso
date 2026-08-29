import { usePollingQuery } from '@/hooks/usePolling';
import { fetchIntegrations, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber } from '@/lib/utils';
import { useSearchParams } from 'react-router-dom';
import { ArrowRight, Database, Layers, ShieldCheck, Send, Network, MessageSquare } from 'lucide-react';

export function IntegrationsPage() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const { data: integrations } = usePollingQuery(['integrations'], fetchIntegrations, 5000);
  const { data: summary } = usePollingQuery(['summary', runId || 'live'], () => fetchSummary(runId || undefined), 3000);

  const getBadgeStyle = (status: string) => {
    if (status === 'READY' || status === 'ONLINE' || status === 'ACTIVE') {
      return { background: 'var(--success-soft)', color: 'var(--success)', border: '1px solid var(--success)' };
    }
    if (status === 'DEFERRED') {
      return { background: 'var(--warning-soft)', color: 'var(--warning)', border: '1px solid var(--warning)' };
    }
    return { background: 'var(--bg-subtle)', color: 'var(--text-tertiary)', border: '1px solid var(--border-default)' };
  };

  const getIcon = (key: string) => {
    if (key === 'wazuh') return Database;
    if (key === 'rbta') return Layers;
    if (key === 'model') return ShieldCheck;
    if (key === 'outbox') return Send;
    if (key === 'shuffle') return Network;
    return MessageSquare;
  };

  return (
    <div>
      <PageHeader
        title="Pipeline Integrations"
        description="End-to-end telemetry from raw event canonicalization through RBTA, Isolation Forest scoring, and downstream dispatch"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Raw Events Ingested" value={summary ? formatNumber(summary.raw_alert_count) : '—'} />
        <MetricCard label="MetaAlerts Generated" value={summary ? formatNumber(summary.meta_alert_count) : '—'} />
        <MetricCard
          label="Reduction Achieved"
          value={
            summary && summary.alert_reduction_rate_percent !== null && summary.alert_reduction_rate_percent !== undefined
              ? `${summary.alert_reduction_rate_percent}%`
              : '—'
          }
        />
        <MetricCard label="Active Buckets" value={summary ? formatNumber(summary.active_buckets_count) : '—'} />
      </div>

      <div className="space-y-4">
        {integrations &&
          Object.entries(integrations).map(([key, item], idx, arr) => {
            const Icon = getIcon(key);
            const badgeStyle = getBadgeStyle(item.status);
            return (
              <div
                key={key}
                className="p-5 rounded-[7px] border flex items-start justify-between transition-colors"
                style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
              >
                <div className="flex items-start gap-4">
                  <div
                    className="p-2.5 rounded-[5px] border shrink-0 mt-0.5"
                    style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--action-blue)' }}
                  >
                    <Icon size={20} />
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-1">
                      <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
                        {item.name || key.toUpperCase()}
                      </h3>
                      <span className="px-2 py-0.5 rounded-[3px] text-[11px] font-mono font-semibold" style={badgeStyle}>
                        {item.status}
                      </span>
                    </div>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {item.detail || 'Operational service integration component'}
                    </p>
                  </div>
                </div>
                {idx < arr.length - 1 && (
                  <div className="hidden lg:flex items-center text-xs font-mono shrink-0 pl-4 self-center" style={{ color: 'var(--text-disabled)' }}>
                    <span>CASCADE</span>
                    <ArrowRight size={14} className="ml-1" />
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </div>
  );
}
