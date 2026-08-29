import { useSearchParams, useNavigate } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary, fetchTimeseries } from '@/api/dashboard';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatNumber, formatDateTime, formatScore } from '@/lib/formatters';
import { Button } from '@cloudflare/kumo/components/button';
import { Table } from '@cloudflare/kumo/components/table';
import { WarningCircle, ArrowRight } from '@phosphor-icons/react';
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
    <div className="space-y-6">
      <PageHeader
        title="Security Analytics Overview"
        description="Continuous Rule-Based Temporal Aggregation and Isolation Forest Anomaly Detection"
      />

      {/* Needs Investigation Panel */}
      {needsInvestigation.length > 0 && (
        <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-kumo-hairline">
            <div className="flex items-center gap-2">
              <WarningCircle size={18} className="text-kumo-danger" weight="fill" />
              <h2 className="text-xs font-semibold uppercase tracking-wider text-kumo-default">
                Needs Investigation ({needsInvestigation.length} Active Escalations)
              </h2>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(withRunId('/meta-alerts?action=ESCALATE'))}
            >
              View all escalated alerts <ArrowRight size={14} />
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {needsInvestigation.slice(0, 3).map((m) => (
              <div
                key={m.meta_id}
                onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                className="p-4 rounded-lg border border-kumo-hairline bg-kumo-recessed cursor-pointer transition-all hover:border-kumo-brand"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs font-semibold text-kumo-default">#{m.meta_id}</span>
                  <DecisionBadge decision={m.decision} action={m.action} />
                </div>
                <div className="space-y-1 text-xs mb-3 text-kumo-subtle">
                  <div className="flex justify-between">
                    <span>Agent:</span>
                    <span className="text-kumo-default">{m.agent_name} ({m.agent_id})</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Group:</span>
                    <span className="font-mono text-kumo-default">{m.rule_group_primary}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Score / Thresh:</span>
                    <span className="font-mono text-kumo-default">{m.anomaly_score.toFixed(3)} / {m.threshold_used.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Member Alerts:</span>
                    <span className="font-mono font-semibold text-kumo-default">{m.alert_count}</span>
                  </div>
                </div>
                <Button
                  variant="primary"
                  size="sm"
                  className="w-full justify-center"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(withRunId(`/meta-alerts/${m.meta_id}`));
                  }}
                >
                  Investigate {m.alert_count} Raw Alerts <ArrowRight size={14} />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
          className="cursor-pointer transition-opacity hover:opacity-90"
        >
          <MetricCard
            label="Escalated Incidents"
            value={summary ? formatNumber(summary.escalate_count) : '—'}
          />
        </div>
        <MetricCard label="Contextual Anomalies" value={summary ? formatNumber(summary.anomalies_detected) : '—'} />
        <div
          onClick={() => navigate(withRunId('/rbta'))}
          className="cursor-pointer transition-opacity hover:opacity-90"
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
        <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs">
          <h2 className="text-sm font-semibold mb-4 text-kumo-default">
            Raw Alerts vs MetaAlerts Ingestion Timeline
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={timeseries}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="raw_alerts" stroke="#f6821f" fill="#f6821f20" name="Raw Alerts" />
              <Area type="monotone" dataKey="meta_alerts" stroke="#0055dc" fill="#0055dc20" name="MetaAlerts" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Latest MetaAlerts Table */}
      {recentMetas && recentMetas.items.length > 0 && (
        <div className="rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden shadow-xs">
          <div className="px-5 py-3 border-b border-kumo-hairline flex items-center justify-between">
            <h2 className="text-sm font-semibold text-kumo-default">
              Recent MetaAlert Events
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(withRunId('/meta-alerts'))}
            >
              View all <ArrowRight size={14} />
            </Button>
          </div>
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Meta ID</Table.Head>
                <Table.Head>Time</Table.Head>
                <Table.Head>Agent</Table.Head>
                <Table.Head>Rule Group</Table.Head>
                <Table.Head className="text-right">Raw Count</Table.Head>
                <Table.Head className="text-right">Anomaly Score</Table.Head>
                <Table.Head>Decision</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {recentMetas.items.map((m) => (
                <Table.Row
                  key={m.meta_id}
                  className="cursor-pointer hover:bg-kumo-tint"
                  onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                >
                  <Table.Cell className="font-mono text-xs font-semibold">#{m.meta_id}</Table.Cell>
                  <Table.Cell className="text-xs">{formatDateTime(m.end_time)}</Table.Cell>
                  <Table.Cell className="text-xs">{m.agent_name} ({m.agent_id})</Table.Cell>
                  <Table.Cell className="text-xs font-mono">{m.rule_group_primary}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono font-semibold">{m.alert_count}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatScore(m.anomaly_score)}</Table.Cell>
                  <Table.Cell><DecisionBadge decision={m.decision} action={m.action} /></Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table>
        </div>
      )}
    </div>
  );
}
