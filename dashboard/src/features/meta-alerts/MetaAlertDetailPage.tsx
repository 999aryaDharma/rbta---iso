import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchMetaAlert, fetchMetaAlertTrace } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime } from '@/lib/utils';

export function MetaAlertDetailPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const id = Number(metaId);

  const { data } = useQuery({ queryKey: ['meta-alert', id], queryFn: () => fetchMetaAlert(id) });
  const { data: trace } = useQuery({ queryKey: ['meta-alert-trace', id], queryFn: () => fetchMetaAlertTrace(id) });

  if (!data) return <div className="p-4 text-sm" style={{ color: 'var(--text-tertiary)' }}>Loading MetaAlert...</div>;

  return (
    <div>
      <PageHeader
        title={`MetaAlert #${id}`}
        description={`Agent ${data.agent_name} (${data.agent_id}) · ${data.rule_group_primary}`}
        actions={<DecisionBadge decision={data.decision} action={data.action} />}
      />
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="p-5 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h3 className="font-semibold text-sm mb-3" style={{ color: 'var(--text-primary)' }}>Aggregation & Detection</h3>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Time Window:</span> <span>{formatDateTime(data.start_time)} → {formatDateTime(data.end_time)}</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Raw Alert Count:</span> <span className="font-semibold">{data.alert_count}</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Max Severity:</span> <span>{data.max_severity}/15</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>MITRE Tactics:</span> <span>{data.mitre_tactics.length ? data.mitre_tactics.join(', ') : 'None'}</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Anomaly Score:</span> <span className="font-mono">{data.anomaly_score.toFixed(4)}</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Tukey Threshold:</span> <span className="font-mono">{data.threshold_used.toFixed(4)}</span></div>
            <div className="flex justify-between"><span style={{ color: 'var(--text-tertiary)' }}>Model Version:</span> <span className="font-mono">{data.model_version}</span></div>
          </div>
        </div>
        <div className="p-5 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h3 className="font-semibold text-sm mb-3" style={{ color: 'var(--text-primary)' }}>Seven Features</h3>
          <div className="space-y-2 text-xs font-mono">
            {Object.entries(data.seven_features).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span style={{ color: 'var(--text-tertiary)' }}>{k}:</span>
                <span className="font-semibold">{Number(v).toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {trace && (
        <div className="p-5 border rounded-[7px] mb-6" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h3 className="font-semibold text-sm mb-2" style={{ color: 'var(--text-primary)' }}>Provenance Trace ({trace.source_alert_ids.length} member alerts)</h3>
          <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>Model: <span className="font-mono">{trace.model_version}</span> · Action: {trace.action}</p>
          <div className="flex flex-wrap gap-1.5 max-h-36 overflow-auto">
            {trace.source_alert_ids.map((aid) => (
              <span key={aid} className="px-2 py-0.5 text-xs font-mono rounded-[3px] border" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                {aid}
              </span>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={() => navigate(`/meta-alerts/${id}/raw-alerts`)}
        className="px-4 py-2 text-white rounded-[5px] text-sm font-medium cursor-pointer"
        style={{ background: 'var(--action-blue)' }}
      >
        Investigate {data.alert_count} Raw Alerts
      </button>
    </div>
  );
}
