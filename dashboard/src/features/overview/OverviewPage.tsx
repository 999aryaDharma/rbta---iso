import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary, fetchTimeseries } from '@/api/dashboard';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatNumber, formatPercent, formatDateTime } from '@/lib/utils';
import { useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

export function OverviewPage() {
  const navigate = useNavigate();
  const { data: summary } = usePollingQuery(['summary'], fetchSummary, 3000);
  const { data: timeseries } = usePollingQuery(['timeseries'], fetchTimeseries, 5000);
  const { data: recentMetas } = usePollingQuery(
    ['meta-alerts-recent'],
    () => fetchMetaAlerts({ page: 1, page_size: 10, sort_by: 'end_time', sort_order: 'desc' }),
    5000,
  );

  return (
    <div>
      <PageHeader title="Overview" description="Real-time operational summary" />

      {/* KPI Row 1 */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        <MetricCard label="Raw Alerts" value={summary ? formatNumber(summary.raw_alert_count) : '—'} />
        <MetricCard label="MetaAlerts" value={summary ? formatNumber(summary.meta_alert_count) : '—'} />
        <MetricCard label="Alert Reduction Rate" value={summary ? formatPercent(summary.alert_reduction_rate) : '—'} />
        <MetricCard label="Escalated" value={summary ? formatNumber(summary.escalate_count) : '—'} />
      </div>

      {/* KPI Row 2 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Active Agents" value={summary ? formatNumber(summary.active_agents_count) : '—'} />
        <MetricCard label="Active Buckets" value={summary ? formatNumber(summary.active_buckets_count) : '—'} />
        <MetricCard label="Outbox Depth" value={summary ? formatNumber(summary.outbox_depth) : '—'} />
        <MetricCard label="Source Mode" value={summary?.source_mode ?? '—'} />
      </div>

      {/* Timeseries Chart */}
      {timeseries && timeseries.series.length > 0 && (
        <div className="p-5 rounded-[7px] border mb-6" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h2 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Raw Alerts vs MetaAlerts</h2>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={timeseries.series}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="var(--text-disabled)" />
              <YAxis tick={{ fontSize: 11 }} stroke="var(--text-disabled)" />
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
        <div className="rounded-[7px] border overflow-hidden" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <div className="px-5 py-3 border-b" style={{ borderColor: 'var(--border-default)' }}>
            <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Latest MetaAlerts</h2>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: 'var(--bg-subtle)' }}>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Meta ID</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Time</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
                <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Raw</th>
                <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Score</th>
                <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {recentMetas.items.map((m) => (
                <tr
                  key={m.meta_id}
                  className="cursor-pointer hover:bg-[var(--bg-hover)] border-b"
                  style={{ borderColor: 'var(--border-subtle)' }}
                  onClick={() => navigate(`/meta-alerts/${m.meta_id}`)}
                >
                  <td className="px-4 py-2.5 font-mono text-xs">{m.meta_id}</td>
                  <td className="px-4 py-2.5 text-xs">{formatDateTime(m.end_time)}</td>
                  <td className="px-4 py-2.5 text-xs">{m.agent_name}</td>
                  <td className="px-4 py-2.5 text-xs font-mono">{m.rule_group_primary}</td>
                  <td className="px-4 py-2.5 text-xs text-right">{m.alert_count}</td>
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
