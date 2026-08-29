import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlert, fetchMetaAlertTrace } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime, formatScore } from '@/lib/formatters';
import { Tabs } from '@cloudflare/kumo/components/tabs';
import { Button } from '@cloudflare/kumo/components/button';
import { ArrowRight } from '@phosphor-icons/react';

const SEVEN_FEATURE_KEYS = [
  'max_severity',
  'mitre_tactic_count',
  'critical_mitre_tactic_present',
  'alert_count_log',
  'rule_diversity_shannon',
  'severity_dispersion',
  'agent_criticality',
] as const;

export function MetaAlertDetailPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState('overview');
  const runId = searchParams.get('run_id');
  const id = Number(metaId);

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data } = useQuery({
    queryKey: ['meta-alert', id, runId || 'live'],
    queryFn: () => fetchMetaAlert(id, runId || undefined),
  });

  const { data: trace } = useQuery({
    queryKey: ['meta-alert-trace', id, runId || 'live'],
    queryFn: () => fetchMetaAlertTrace(id, runId || undefined),
  });

  if (!data) {
    return <div className="p-6 text-xs text-kumo-subtle">Loading MetaAlert #{id}...</div>;
  }

  const tabsConfig = [
    { value: 'overview', label: 'Overview & Detection' },
    { value: 'features', label: 'Seven Features' },
    { value: 'provenance', label: `Provenance Trace (${data.alert_count})` },
  ];

  return (
    <>
      <div className="px-6 pt-4 text-xs font-mono text-kumo-subtle flex items-center gap-2">
        <Link to={withRunId('/meta-alerts')} className="hover:underline text-kumo-default">MetaAlerts</Link>
        <span>/</span>
        <span className="text-kumo-strong font-semibold">#{id}</span>
      </div>

      <PageHeader
        title={`MetaAlert #${id}`}
        description={`Agent: ${data.agent_name} (${data.agent_id}) · Primary Rule Group: ${data.rule_group_primary}`}
        actions={
          <div className="flex items-center gap-3">
            <DecisionBadge decision={data.decision} action={data.action} />
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate(withRunId(`/meta-alerts/${id}/raw-alerts`))}
            >
              Investigate {data.alert_count} Raw Alerts <ArrowRight size={14} />
            </Button>
          </div>
        }
      />

      <div className="px-6">
        <Tabs
          tabs={tabsConfig}
          value={activeTab}
          onValueChange={setActiveTab}
          variant="underline"
        />
      </div>

      <div className="px-6 py-4 space-y-4">
        {/* Tab 1: Overview & Detection */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Aggregation Profile */}
            <div>
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">
                Temporal Aggregation Profile
              </h3>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt className="text-kumo-subtle">Cluster Window:</dt> <dd className="font-mono text-kumo-default">{formatDateTime(data.start_time)} → {formatDateTime(data.end_time)}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Raw Alert Count:</dt> <dd className="font-mono font-semibold text-kumo-default">{data.alert_count} events</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Max Rule Severity:</dt> <dd className="font-mono font-semibold text-kumo-default">{data.max_severity}/15</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Agent Criticality:</dt> <dd className="font-mono text-kumo-default">{data.seven_features.agent_criticality ?? 1.0}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">MITRE Tactics:</dt> <dd className="font-mono text-kumo-default">{data.mitre_tactics.length ? data.mitre_tactics.join(', ') : 'None'}</dd></div>
              </dl>
            </div>

            {/* Isolation Forest Decision */}
            <div>
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">
                Isolation Forest Evaluation
              </h3>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt className="text-kumo-subtle">Anomaly Score:</dt> <dd className="font-mono font-semibold text-kumo-default">{formatScore(data.anomaly_score)}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Tukey IQR Threshold:</dt> <dd className="font-mono text-kumo-default">{formatScore(data.threshold_used)}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Classification Decision:</dt> <dd className="font-semibold text-kumo-default">{data.decision}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Operational Action:</dt> <dd className="font-semibold text-kumo-default">{data.action}</dd></div>
                <div className="flex justify-between"><dt className="text-kumo-subtle">Model Version:</dt> <dd className="font-mono text-kumo-default">{data.model_version}</dd></div>
              </dl>
            </div>
          </div>
        )}

        {/* Tab 2: Seven Features */}
        {activeTab === 'features' && (
          <div>
            <h3 className="font-semibold text-xs uppercase tracking-wider mb-4 text-kumo-default pb-2 border-b border-kumo-hairline">
              Canonical 7-Feature Vector (Locked Research Order)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {SEVEN_FEATURE_KEYS.map((key, idx) => {
                const val = data.seven_features[key];
                return (
                  <div
                    key={key}
                    className="p-3 rounded-lg border border-kumo-hairline bg-kumo-recessed"
                  >
                    <div className="text-[11px] font-mono mb-1 text-kumo-subtle">
                      #{idx + 1} {key}
                    </div>
                    <div className="text-sm font-mono font-semibold text-kumo-default">
                      {val !== undefined ? Number(val).toFixed(4) : '—'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab 3: Provenance Trace */}
        {activeTab === 'provenance' && trace && (
          <div>
            <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">
              Audit Provenance Trace ({trace.source_alert_ids.length} member alert IDs)
            </h3>
            <p className="text-xs mb-3 text-kumo-subtle">
              Click any raw alert ID below to inspect its individual cryptographic evidence and canonical payload.
            </p>
            <div className="flex flex-wrap gap-2 max-h-60 overflow-auto">
              {trace.source_alert_ids.map((aid, i) => (
                <button
                  key={aid}
                  onClick={() => navigate(withRunId(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(aid)}`))}
                  className="px-2.5 py-1 text-xs font-mono rounded-md border border-kumo-hairline bg-kumo-recessed text-kumo-default hover:border-kumo-brand transition-colors cursor-pointer"
                >
                  <span className="text-[10px] mr-1 text-kumo-subtle">#{i + 1}</span>
                  {aid}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
