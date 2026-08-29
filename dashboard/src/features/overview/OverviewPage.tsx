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
    <>
      <PageHeader
        breadcrumbs={['Security Analytics', 'Overview']}
        title="Security Analytics Overview"
        description="Continuous Rule-Based Temporal Aggregation (RBTA) and Isolation Forest Anomaly Detection Engine"
      />

      <div className="px-6 py-8 lg:px-10 space-y-8">
        {/* Needs Investigation Banner */}
        {needsInvestigation.length > 0 && (
          <div className="p-6 rounded-xl border border-rose-500/30 border-l-4 border-l-rose-500 bg-rose-500/5 shadow-xs space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-rose-500/20">
              <div className="flex items-center gap-3">
                <WarningCircle size={22} className="text-rose-600 dark:text-rose-400 shrink-0" weight="fill" />
                <h2 className="text-xs font-semibold uppercase tracking-wider text-kumo-strong">
                  Needs Investigation ({needsInvestigation.length} Active Escalations)
                </h2>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-rose-600 dark:text-rose-400 hover:text-rose-700"
                onClick={() => navigate(withRunId('/meta-alerts?action=ESCALATE'))}
              >
                View all escalated incidents <ArrowRight size={14} className="ml-1" />
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {needsInvestigation.slice(0, 3).map((m) => (
                <div
                  key={m.meta_id}
                  onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                  className="p-5 rounded-xl border border-kumo-hairline bg-kumo-canvas cursor-pointer transition-all hover:border-kumo-strong hover:shadow-xs space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-kumo-strong">#{m.meta_id}</span>
                    <DecisionBadge decision={m.decision} action={m.action} />
                  </div>
                  <div className="space-y-2 text-xs text-kumo-subtle">
                    <div className="flex justify-between items-center">
                      <span>Agent:</span>
                      <span className="text-kumo-default font-medium truncate max-w-[160px]">{m.agent_name} ({m.agent_id})</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Rule Group:</span>
                      <span className="font-mono text-kumo-default bg-kumo-recessed px-2 py-0.5 rounded text-[11px] border border-kumo-hairline">{m.rule_group_primary}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Score / Thresh:</span>
                      <span className="font-mono text-kumo-strong font-semibold">{m.anomaly_score.toFixed(3)} / {m.threshold_used.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Aggregated Alerts:</span>
                      <span className="font-mono font-bold text-kumo-strong">{m.alert_count}</span>
                    </div>
                  </div>
                  <Button
                    variant="primary"
                    size="sm"
                    className="w-full justify-center text-xs mt-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(withRunId(`/meta-alerts/${m.meta_id}`));
                    }}
                  >
                    Investigate {m.alert_count} Raw Alerts <ArrowRight size={13} className="ml-1" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* KPI Cards Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          <MetricCard
            label="Raw Ingested Alerts"
            value={summary ? formatNumber(summary.raw_alert_count) : '—'}
            sub="Incoming Wazuh events"
          />
          <MetricCard
            label="Finalized MetaAlerts"
            value={summary ? formatNumber(summary.meta_alert_count) : '—'}
            sub="Clustered temporal episodes"
          />
          <MetricCard
            label="Alert Reduction Rate"
            value={
              summary && summary.alert_reduction_rate_percent !== null && summary.alert_reduction_rate_percent !== undefined
                ? `${summary.alert_reduction_rate_percent}%`
                : '—'
            }
            sub="SOC noise elimination"
          />
          <div
            onClick={() => navigate(withRunId('/meta-alerts?action=ESCALATE'))}
            className="cursor-pointer transition-opacity hover:opacity-95"
          >
            <MetricCard
              label="Escalated Incidents"
              value={summary ? formatNumber(summary.escalate_count) : '—'}
              sub="High-priority anomalies"
            />
          </div>
          <MetricCard
            label="Contextual Anomalies"
            value={summary ? formatNumber(summary.anomalies_detected) : '—'}
            sub="Outliers detected by IF"
          />
          <div
            onClick={() => navigate(withRunId('/rbta'))}
            className="cursor-pointer transition-opacity hover:opacity-95"
          >
            <MetricCard
              label="Active Open Buckets"
              value={summary ? formatNumber(summary.active_buckets_count) : '—'}
              sub="Live temporal windows"
            />
          </div>
          <MetricCard
            label="Digest Queue"
            value={summary ? formatNumber(summary.digest_count) : '—'}
            sub="Low-frequency routine batches"
          />
          <MetricCard
            label="Suppressed Noise"
            value={summary ? formatNumber(summary.suppress_count) : '—'}
            sub="Benign repetitive patterns"
          />
        </div>

        {/* Timeseries Ingestion Chart */}
        {timeseries && Array.isArray(timeseries) && timeseries.length > 0 && (
          <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-kumo-hairline">
              <div>
                <h2 className="text-sm font-semibold text-kumo-strong">
                  Ingestion & Aggregation Velocity
                </h2>
                <p className="text-xs text-kumo-subtle mt-0.5">
                  Raw Wazuh alerts stream vs finalized MetaAlerts over time
                </p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={timeseries}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Area type="monotone" dataKey="raw_alerts" stroke="#64748b" fill="#64748b20" name="Raw Alerts" />
                <Area type="monotone" dataKey="meta_alerts" stroke="#0f172a" fill="#0f172a20" name="MetaAlerts" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Recent MetaAlerts Table */}
        {recentMetas && recentMetas.items.length > 0 && (
          <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs overflow-hidden">
            <div className="px-6 py-4 border-b border-kumo-hairline flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-kumo-strong">
                  Recent MetaAlert Episodes
                </h2>
                <p className="text-xs text-kumo-subtle mt-0.5">
                  Latest aggregated alert groups evaluated by Isolation Forest
                </p>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => navigate(withRunId('/meta-alerts'))}
              >
                View all MetaAlerts <ArrowRight size={13} className="ml-1" />
              </Button>
            </div>
            <Table>
              <Table.Header>
                <Table.Row className="bg-kumo-recessed/50 text-[11px] uppercase tracking-wider">
                  <Table.Head>Meta ID</Table.Head>
                  <Table.Head>Timestamp</Table.Head>
                  <Table.Head>Agent / Host</Table.Head>
                  <Table.Head>Rule Name</Table.Head>
                  <Table.Head className="text-right">Alert Count</Table.Head>
                  <Table.Head className="text-right">Anomaly Score</Table.Head>
                  <Table.Head>SOC Decision</Table.Head>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {recentMetas.items.map((m) => (
                  <Table.Row
                    key={m.meta_id}
                    className="cursor-pointer hover:bg-kumo-recessed/40 transition-colors text-xs"
                    onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                  >
                    <Table.Cell className="font-mono font-semibold text-kumo-strong">#{m.meta_id}</Table.Cell>
                    <Table.Cell className="text-kumo-subtle">{formatDateTime(m.end_time)}</Table.Cell>
                    <Table.Cell className="font-medium text-kumo-default">{m.agent_name} <span className="text-kumo-subtle text-[11px]">({m.agent_id})</span></Table.Cell>
                    <Table.Cell className="font-mono text-kumo-default">{m.rule_group_primary}</Table.Cell>
                    <Table.Cell className="text-right font-mono font-bold text-kumo-strong">{m.alert_count}</Table.Cell>
                    <Table.Cell className="text-right font-mono text-kumo-default">{formatScore(m.anomaly_score)}</Table.Cell>
                    <Table.Cell><DecisionBadge decision={m.decision} action={m.action} /></Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table>
          </div>
        )}
      </div>
    </>
  );
}
