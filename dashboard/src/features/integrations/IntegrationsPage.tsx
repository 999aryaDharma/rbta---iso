import { usePollingQuery } from '@/hooks/usePolling';
import { fetchIntegrations, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber } from '@/lib/formatters';
import { useSearchParams } from 'react-router-dom';
import { Badge } from '@cloudflare/kumo/components/badge';
import { ArrowRight, Database, Stack, ShieldCheck, PaperPlaneTilt, Plugs, ChatTeardropText } from '@phosphor-icons/react';

export function IntegrationsPage() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const { data: integrations } = usePollingQuery(['integrations'], fetchIntegrations, 5000);
  const { data: summary } = usePollingQuery(['summary', runId || 'live'], () => fetchSummary(runId || undefined), 3000);

  const getBadgeVariant = (status: string) => {
    if (status === 'READY' || status === 'ONLINE' || status === 'ACTIVE') {
      return 'success';
    }
    if (status === 'DEFERRED') {
      return 'warning';
    }
    return 'secondary';
  };

  const getIcon = (key: string) => {
    if (key === 'wazuh') return Database;
    if (key === 'rbta') return Stack;
    if (key === 'model') return ShieldCheck;
    if (key === 'outbox') return PaperPlaneTilt;
    if (key === 'shuffle') return Plugs;
    return ChatTeardropText;
  };

  return (
    <>
      <PageHeader
        title="Pipeline Integrations"
        description="End-to-end telemetry from raw event canonicalization through RBTA, Isolation Forest scoring, and downstream dispatch"
      />

      <div className="px-6 py-4 space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
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

        <div className="space-y-3">
          {integrations &&
            Object.entries(integrations).map(([key, item], idx, arr) => {
              const Icon = getIcon(key);
              const variant = getBadgeVariant(item.status);
              return (
                <div
                  key={key}
                  className="p-4 rounded-lg border border-kumo-hairline bg-kumo-base flex items-start justify-between transition-colors hover:border-kumo-line"
                >
                  <div className="flex items-start gap-3.5">
                    <div className="p-2 rounded-md border border-kumo-hairline bg-kumo-recessed text-kumo-brand shrink-0 mt-0.5">
                      <Icon size={18} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2.5 mb-0.5">
                        <h3 className="font-semibold text-xs text-kumo-default">
                          {item.name || key.toUpperCase()}
                        </h3>
                        <Badge variant={variant as any}>
                          {item.status}
                        </Badge>
                      </div>
                      <p className="text-xs text-kumo-subtle">
                        {item.detail || 'Operational service integration component'}
                      </p>
                    </div>
                  </div>
                  {idx < arr.length - 1 && (
                    <div className="hidden lg:flex items-center text-xs font-mono shrink-0 pl-4 self-center text-kumo-inactive">
                      <span>CASCADE</span>
                      <ArrowRight size={14} className="ml-1" />
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      </div>
    </>
  );
}
