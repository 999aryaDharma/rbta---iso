import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo, fetchSummary } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber } from '@/lib/utils';
import { ArrowRight, ShieldCheck, Database, Layers, Send } from 'lucide-react';

export function IntegrationsPage() {
  const { data: sys } = usePollingQuery(['system'], fetchSystemInfo, 3000);
  const { data: summary } = usePollingQuery(['summary'], fetchSummary, 3000);

  const stages = [
    {
      title: '1. Wazuh Live Source',
      status: 'DEFERRED / REPLAY',
      badgeClass: 'var(--warning-soft)',
      badgeColor: 'var(--warning)',
      icon: Database,
      desc: 'Physical endpoint connectivity deferred. Operating in synthetic replay & local streaming mode.',
      stat: `Source Mode: ${summary?.source_mode ?? 'STANDALONE'}`,
    },
    {
      title: '2. RBTA Engine',
      status: 'ACTIVE',
      badgeClass: 'var(--success-soft)',
      badgeColor: 'var(--success)',
      icon: Layers,
      desc: 'Per-agent temporal aggregation with 100-event warmup and dynamic EMA gap windowing.',
      stat: `${summary?.active_agents_count ?? 0} active agents · ${summary?.active_buckets_count ?? 0} open buckets`,
    },
    {
      title: '3. Isolation Forest',
      status: 'READY',
      badgeClass: 'var(--success-soft)',
      badgeColor: 'var(--success)',
      icon: ShieldCheck,
      desc: 'Seven-feature anomaly detection scoring against calibrated Tukey IQR threshold.',
      stat: `Model Version: ${sys?.model_version ?? '—'} (Threshold: ${sys?.threshold.toFixed(4) ?? '—'})`,
    },
    {
      title: '4. SOAR & Telegram Outbox',
      status: 'ONLINE',
      badgeClass: 'var(--success-soft)',
      badgeColor: 'var(--success)',
      icon: Send,
      desc: 'Escalation router with atomic dispatch queue and durable crash recovery.',
      stat: `Outbox Depth: ${summary?.outbox_depth ?? 0} queued alerts`,
    },
  ];

  return (
    <div>
      <PageHeader
        title="Pipeline Integrations"
        description="End-to-end telemetry from raw event canonicalization through RBTA, Isolation Forest scoring, and downstream dispatch"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Raw Events Ingested" value={summary ? formatNumber(summary.raw_alert_count) : '—'} />
        <MetricCard label="MetaAlerts Generated" value={summary ? formatNumber(summary.meta_alert_count) : '—'} />
        <MetricCard label="Reduction Achieved" value={summary ? `${(summary.alert_reduction_rate * 100).toFixed(1)}%` : '—'} />
        <MetricCard label="Outbox Dispatch Queue" value={summary ? formatNumber(summary.outbox_depth) : '—'} />
      </div>

      <div className="space-y-4">
        {stages.map((stage, idx) => (
          <div
            key={stage.title}
            className="p-5 rounded-[7px] border flex items-start justify-between transition-colors"
            style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
          >
            <div className="flex items-start gap-4">
              <div
                className="p-2.5 rounded-[5px] border shrink-0 mt-0.5"
                style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--action-blue)' }}
              >
                <stage.icon size={20} />
              </div>
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{stage.title}</h3>
                  <span
                    className="px-2 py-0.5 rounded-[3px] text-[11px] font-semibold tracking-wide"
                    style={{ background: stage.badgeClass, color: stage.badgeColor }}
                  >
                    {stage.status}
                  </span>
                </div>
                <p className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>{stage.desc}</p>
                <span className="font-mono text-xs" style={{ color: 'var(--text-tertiary)' }}>{stage.stat}</span>
              </div>
            </div>
            {idx < stages.length - 1 && (
              <div className="hidden lg:flex items-center text-xs font-mono shrink-0 pl-4 self-center" style={{ color: 'var(--text-disabled)' }}>
                <span>CASCADE</span>
                <ArrowRight size={14} className="ml-1" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
