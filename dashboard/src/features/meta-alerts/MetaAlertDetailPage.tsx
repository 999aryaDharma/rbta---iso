import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
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
      <PageHeader
        breadcrumbs={['Security Analytics', 'MetaAlerts', `#${id}`]}
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
              Investigate {data.alert_count} Raw Alerts <ArrowRight size={14} className="ml-1" />
            </Button>
          </div>
        }
      />

      <div className="px-6 lg:px-8 border-b border-kumo-hairline bg-kumo-canvas">
        <Tabs
          tabs={tabsConfig}
          value={activeTab}
          onValueChange={setActiveTab}
          variant="underline"
        />
      </div>

      <div className="px-6 py-8 lg:px-10 space-y-8">
        {/* Tab 1: Overview & Detection */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Aggregation Profile Card */}
            <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs">
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong mb-4 pb-3 border-b border-kumo-hairline">
                Temporal Aggregation Profile
              </h3>
              <dl className="space-y-3 text-xs">
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Cluster Window:</dt>
                  <dd className="font-mono text-kumo-default">{formatDateTime(data.start_time)} → {formatDateTime(data.end_time)}</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Aggregated Event Count:</dt>
                  <dd className="font-mono font-bold text-kumo-strong">{data.alert_count} events</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Max Rule Severity:</dt>
                  <dd className="font-mono font-semibold text-kumo-default">{data.max_severity} / 15</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Agent Criticality Weight:</dt>
                  <dd className="font-mono text-kumo-default">{data.seven_features.agent_criticality ?? 1.0}</dd>
                </div>
                <div className="flex justify-between items-center py-1">
                  <dt className="text-kumo-subtle font-medium">MITRE Tactics Present:</dt>
                  <dd className="font-mono text-kumo-default text-right">{data.mitre_tactics.length ? data.mitre_tactics.join(', ') : 'None'}</dd>
                </div>
              </dl>
            </div>

            {/* Isolation Forest Evaluation Card */}
            <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs">
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong mb-4 pb-3 border-b border-kumo-hairline">
                Isolation Forest Evaluation & Scoring
              </h3>
              <dl className="space-y-3 text-xs">
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Calibrated Anomaly Score:</dt>
                  <dd className="font-mono font-bold text-kumo-strong text-sm">{formatScore(data.anomaly_score)}</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Deterministic Tukey Threshold:</dt>
                  <dd className="font-mono font-medium text-kumo-default">{formatScore(data.threshold_used)}</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">Classification Decision:</dt>
                  <dd className="font-semibold text-kumo-default">{data.decision}</dd>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <dt className="text-kumo-subtle font-medium">SOC Action Trigger:</dt>
                  <dd><DecisionBadge decision={data.decision} action={data.action} /></dd>
                </div>
                <div className="flex justify-between items-center py-1">
                  <dt className="text-kumo-subtle font-medium">Model Artifact Registry:</dt>
                  <dd className="font-mono text-kumo-subtle">{data.model_version}</dd>
                </div>
              </dl>
            </div>
          </div>
        )}

        {/* Tab 2: Seven Features */}
        {activeTab === 'features' && (
          <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-5">
            <div>
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong pb-2 border-b border-kumo-hairline">
                Canonical 7-Feature Vector (Locked Research Specification)
              </h3>
              <p className="text-xs text-kumo-subtle mt-1.5">
                Exact numerical features extracted from the temporal episode and fed into Isolation Forest
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pt-2">
              {SEVEN_FEATURE_KEYS.map((key, idx) => {
                const val = data.seven_features[key];
                return (
                  <div
                    key={key}
                    className="p-5 rounded-xl border border-kumo-hairline bg-kumo-recessed/30 space-y-2.5"
                  >
                    <div className="text-[11px] font-mono text-kumo-subtle flex items-center justify-between">
                      <span className="font-semibold text-kumo-default">#{idx + 1}</span>
                      <span className="truncate ml-1">{key}</span>
                    </div>
                    <div className="text-base font-mono font-bold text-kumo-strong">
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
          <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-5">
            <div>
              <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong pb-2 border-b border-kumo-hairline">
                Cryptographic Audit Provenance Trace ({trace.source_alert_ids.length} member alerts)
              </h3>
              <p className="text-xs text-kumo-subtle mt-1.5">
                Every raw Wazuh event aggregated into this MetaAlert is cryptographically hashed and indexed in SQLite evidence storage
              </p>
            </div>
            <div className="flex flex-wrap gap-2.5 max-h-80 overflow-y-auto p-4 rounded-xl border border-kumo-hairline bg-kumo-recessed/20">
              {trace.source_alert_ids.map((aid, i) => (
                <button
                  key={aid}
                  onClick={() => navigate(withRunId(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(aid)}`))}
                  className="px-3 py-1.5 text-xs font-mono rounded-lg border border-kumo-hairline bg-kumo-canvas text-kumo-default hover:border-kumo-strong hover:text-kumo-strong transition-all cursor-pointer shadow-2xs"
                >
                  <span className="text-[10px] mr-1.5 text-kumo-subtle">#{i + 1}</span>
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
