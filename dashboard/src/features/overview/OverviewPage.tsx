import { useSearchParams, useNavigate } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary, fetchTimeseries } from '@/api/dashboard';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatNumber, formatDateTime } from '@/lib/utils';
import { AlertTriangle, ArrowRight } from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

export function OverviewPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data: summary } = usePollingQuery(['summary', runId || 'live'], () => fetchSummary(runId || undefined), 3000);
  const { data: timeseries } = usePollingQuery(['timeseries', runId || 'live'], () => fetchTimeseries(24, runId || undefined), 5000);
  const { data: recentMetas } = usePollingQuery(
    ['meta-alerts-recent', runId || 'live'],
    () => fetchMetaAlerts({ page: 1, page_size: 10, sort_by: 'end_time', sort_order: 'desc', run_id: runId || undefined }),
    5000,
  );

  const needsInvestigation = recentMetas?.items.filter((m) => m.action === 'ESCALATE') || [];

  return (
    <div>
      <PageHeader
        title="Security Analytics Overview"
        description="Continuous Rule-Based Temporal Aggregation and Isolation Forest Anomaly Detection"
      />

      {/* Needs Investigation Panel */}
      {needsInvestigation.length > 0 && (
        <div
          className="p-5 rounded-[7px] border mb-6"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="flex items-center justify-between pb-3 mb-3 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex items-center gap-2">
              <AlertTriangle size={16} style={{ color: 'var(--danger)' }} />
              <h2 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-primary)' }}>
                Needs Investigation ({needsInvestigation.length} Active Escalations)
              </h2>
            </div>
            <button
              onClick={() => navigate(withRunId('/meta-alerts?action=ESCALATE'))}
              className="text-xs font-medium flex items-center gap-1 cursor-pointer hover:underline"
              style={{ color: 'var(--action-blue)' }}
            >
              View all escalated alerts <ArrowRight size={12} />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {needsInvestigation.slice(0, 3).map((m) => (
              <div
                key={m.meta_id}
                onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                className="p-3.5 rounded-[5px] border cursor-pointer transition-all hover:border-[var(--brand-orange)]"
                style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs font-semibold">#{m.meta_id}</span>
                  <DecisionBadge decision={m.decision} action={m.action} />
                </div>
                <div className="space-y-1 text-xs mb-3">
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Agent:</span>
                    <span>{m.agent_name} ({m.agent_id})</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Group:</span>
                    <span className="font-mono">{m.rule_group_primary}</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Score / Thresh:</span>
                    <span className="font-mono">{m.anomaly_score.toFixed(3)} / {m.threshold_used.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Member Alerts:</span>
                    <span className="font-mono font-semibold">{m.alert_count}</span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(withRunId(`/meta-alerts/${m.meta_id}`));
                  }}
                  className="w-full py-1.5 px-2 rounded-[4px] text-xs font-medium text-white flex items-center justify-center gap-1.5 cursor-pointer"
                  style={{ background: 'var(--action-blue)' }}
                >
                  Investigate {m.alert_count} Raw Alerts <ArrowRight size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Raw Ingested Alerts" value={summary ? formatNumber(summary.raw_alert_count) : '—'} />
        <MetricCard label="Finalized MetaAlerts" value={summary ? formatNumber(summary.meta_alert_count) : '—'} />
        <MetricCard
          label="Alert Reduction Rate"
          value={
            summary && summary.alert_reduction_rate_percent !== null && summary.alert_reduction_rate_percent !== undefined
              ? `${summary.alert_reduction_rate_percent}%`
              : '—'
          }
        />
        <div
          onClick={() => navigate(withRunId('/meta-alerts?action=ESCALATE'))}
          className="cursor-pointer hover:opacity-90"
        >
          <MetricCard
            label="Escalated Incidents"
            value={summary ? formatNumber(summary.escalate_count) : '—'}
          />
        </div>
        <MetricCard label="Contextual Anomalies" value={summary ? formatNumber(summary.anomalies_detected) : '—'} />
        <div
          onClick={() => navigate(withRunId('/rbta'))}
          className="cursor-pointer hover:opacity-90"
        >
          <MetricCard
            label="Active Open Buckets"
            value={summary ? formatNumber(summary.active_buckets_count) : '—'}
          />
        </div>
        <MetricCard label="Digest Queue" value={summary ? formatNumber(summary.digest_count) : '—'} />
        <MetricCard label="Suppressed Noise" value={summary ? formatNumber(summary.suppress_count) : '—'} />
      </div>

      {/* Timeseries Chart */}
      {timeseries && Array.isArray(timeseries) && timeseries.length > 0 && (
        <div
          className="p-5 rounded-[7px] border mb-6"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <h2 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Raw Alerts vs MetaAlerts Ingestion Timeline
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={timeseries}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} stroke="var(--text-disabled)" />
              <YAxis tick={{ fontSize: 10 }} stroke="var(--text-disabled)" />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="raw_alerts" stroke="var(--brand-orange)" fill="var(--brand-orange-soft)" name="Raw Alerts" />
              <Area type="monotone" dataKey="meta_alerts" stroke="var(--action-blue)" fill="var(--action-blue-soft)" name="MetaAlerts" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Latest MetaAlerts Table */}
      {recentMetas && recentMetas.items.length > 0 && (
        <div
          className="rounded-[7px] border overflow-hidden"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="px-5 py-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--border-default)' }}>
            <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              Recent MetaAlert Events
            </h2>
            <button
              onClick={() => navigate(withRunId('/meta-alerts'))}
              className="text-xs font-medium flex items-center gap-1 cursor-pointer hover:underline"
              style={{ color: 'var(--action-blue)' }}
            >
              View all <ArrowRight size={12} />
            </button>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: 'var(--bg-subtle)' }}>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Meta ID</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Time</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
                <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Raw Count</th>
                <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Anomaly Score</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {recentMetas.items.map((m) => (
                <tr
                  key={m.meta_id}
                  className="cursor-pointer hover:bg-[var(--bg-hover)] border-b"
                  style={{ borderColor: 'var(--border-subtle)' }}
                  onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                >
                  <td className="px-4 py-2.5 font-mono text-xs font-semibold">#{m.meta_id}</td>
                  <td className="px-4 py-2.5 text-xs">{formatDateTime(m.end_time)}</td>
                  <td className="px-4 py-2.5 text-xs">{m.agent_name} ({m.agent_id})</td>
                  <td className="px-4 py-2.5 text-xs font-mono">{m.rule_group_primary}</td>
                  <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold">{m.alert_count}</td>
                  <td className="px-4 py-2.5 text-xs text-right font-mono">{m.anomaly_score.toFixed(4)}</td>
                  <td className="px-4 py-2.5"><DecisionBadge decision={m.decision} action={m.action} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
