import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlert, fetchMetaAlertTrace } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime } from '@/lib/utils';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Shield, ArrowRight, Layers, Fingerprint } from 'lucide-react';

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
    return <div className="p-6 text-xs" style={{ color: 'var(--text-tertiary)' }}>Loading MetaAlert #{id}...</div>;
  }

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
        <Link to={withRunId('/meta-alerts')} className="hover:underline">MetaAlerts</Link>
        <span>/</span>
        <span style={{ color: 'var(--text-primary)' }}>#{id}</span>
      </div>

      <PageHeader
        title={`MetaAlert #${id}`}
        description={`Agent: ${data.agent_name} (${data.agent_id}) · Primary Rule Group: ${data.rule_group_primary}`}
        actions={
          <div className="flex items-center gap-3">
            <DecisionBadge decision={data.decision} action={data.action} />
            <button
              onClick={() => navigate(withRunId(`/meta-alerts/${id}/raw-alerts`))}
              className="flex items-center gap-1.5 px-3 py-1.5 text-white rounded-[5px] text-xs font-medium cursor-pointer"
              style={{ background: 'var(--action-blue)' }}
            >
              Investigate {data.alert_count} Raw Alerts <ArrowRight size={14} />
            </button>
          </div>
        }
      />

      <Tabs defaultValue="overview" className="mt-4">
        <TabsList>
          <TabsTrigger value="overview">Overview & Detection</TabsTrigger>
          <TabsTrigger value="features">Seven Features</TabsTrigger>
          <TabsTrigger value="provenance">Provenance Trace ({data.alert_count})</TabsTrigger>
        </TabsList>

        {/* Tab 1: Overview & Detection */}
        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Aggregation Profile */}
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Layers size={16} style={{ color: 'var(--brand-orange)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Temporal Aggregation Profile</h3>
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Cluster Window:</dt> <dd className="font-mono">{formatDateTime(data.start_time)} → {formatDateTime(data.end_time)}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Raw Alert Count:</dt> <dd className="font-mono font-semibold">{data.alert_count} events</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Max Rule Severity:</dt> <dd className="font-mono font-semibold">{data.max_severity}/15</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Agent Criticality:</dt> <dd className="font-mono">{data.seven_features.agent_criticality ?? 1.0}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>MITRE Tactics:</dt> <dd className="font-mono">{data.mitre_tactics.length ? data.mitre_tactics.join(', ') : 'None'}</dd></div>
              </dl>
            </div>

            {/* Isolation Forest Decision */}
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Shield size={16} style={{ color: 'var(--action-blue)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Isolation Forest Evaluation</h3>
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Anomaly Score:</dt> <dd className="font-mono font-semibold">{data.anomaly_score.toFixed(4)}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Tukey IQR Threshold:</dt> <dd className="font-mono">{data.threshold_used.toFixed(4)}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Classification Decision:</dt> <dd className="font-semibold">{data.decision}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Operational Action:</dt> <dd className="font-semibold">{data.action}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Model Version:</dt> <dd className="font-mono">{data.model_version}</dd></div>
              </dl>
            </div>
          </div>
        </TabsContent>

        {/* Tab 2: Seven Features */}
        <TabsContent value="features">
          <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
            <h3 className="font-semibold text-xs uppercase tracking-wider mb-4" style={{ color: 'var(--text-secondary)' }}>
              Canonical 7-Feature Vector (Locked Research Order)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {SEVEN_FEATURE_KEYS.map((key, idx) => {
                const val = data.seven_features[key];
                return (
                  <div
                    key={key}
                    className="p-3 rounded-[5px] border"
                    style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-subtle)' }}
                  >
                    <div className="text-[11px] font-mono mb-1" style={{ color: 'var(--text-tertiary)' }}>
                      #{idx + 1} {key}
                    </div>
                    <div className="text-sm font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {val !== undefined ? Number(val).toFixed(4) : '—'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </TabsContent>

        {/* Tab 3: Provenance Trace */}
        <TabsContent value="provenance">
          {trace && (
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Fingerprint size={16} style={{ color: 'var(--success)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                  Audit Provenance Trace ({trace.source_alert_ids.length} member alert IDs)
                </h3>
              </div>
              <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
                Click any raw alert ID below to inspect its individual cryptographic evidence and canonical payload.
              </p>
              <div className="flex flex-wrap gap-2 max-h-60 overflow-auto">
                {trace.source_alert_ids.map((aid, i) => (
                  <button
                    key={aid}
                    onClick={() => navigate(withRunId(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(aid)}`))}
                    className="px-2 py-1 text-xs font-mono rounded-[4px] border hover:border-[var(--brand-orange)] transition-colors cursor-pointer"
                    style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
                  >
                    <span className="text-[10px] mr-1" style={{ color: 'var(--text-tertiary)' }}>#{i + 1}</span>
                    {aid}
                  </button>
                ))}
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
