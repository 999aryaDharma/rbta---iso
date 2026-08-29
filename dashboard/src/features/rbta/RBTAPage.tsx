import { useSearchParams } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchAgents, fetchBuckets } from '@/api/dashboard';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatNumber, formatSeconds, formatDateTime } from '@/lib/formatters';
import { Table } from '@cloudflare/kumo/components/table';
import { Badge } from '@cloudflare/kumo/components/badge';

export function RBTAPage() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const { data: agents = [] } = usePollingQuery(['agents', runId || 'live'], () => fetchAgents(runId || undefined), 3000);
  const { data: buckets = [] } = usePollingQuery(['buckets', runId || 'live'], () => fetchBuckets(runId || undefined), 3000);

  const activeAgents = agents.length;
  const warmedUp = agents.filter((a) => a.is_warmed_up).length;
  const seenAlerts = agents.reduce((acc, a) => acc + a.event_count, 0);
  const activeBuckets = buckets.length;

  return (
    <div className="space-y-6">
      <PageHeader
        title="RBTA Engine Telemetry"
        description="Real-time agent temporal state, dynamic aggregation windows, and active open buckets"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Active Agents" value={formatNumber(activeAgents)} />
        <MetricCard label="Warmed-up Agents" value={formatNumber(warmedUp)} />
        <MetricCard label="Seen Alerts" value={formatNumber(seenAlerts)} />
        <MetricCard label="Open Active Buckets" value={formatNumber(activeBuckets)} />
      </div>

      <div>
        <h2 className="text-sm font-semibold mb-3 text-kumo-default">
          Agent Temporal States ({agents.length})
        </h2>
        <div className="rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden shadow-xs">
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Agent ID</Table.Head>
                <Table.Head>Name</Table.Head>
                <Table.Head className="text-right">Events</Table.Head>
                <Table.Head className="text-center">Warmup</Table.Head>
                <Table.Head className="text-right">Baseline Gap</Table.Head>
                <Table.Head className="text-right">EMA Gap</Table.Head>
                <Table.Head className="text-right">Base Δt</Table.Head>
                <Table.Head className="text-right">Current Δt</Table.Head>
                <Table.Head className="text-right">Buckets</Table.Head>
                <Table.Head className="text-center">Status</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {agents.map((a) => (
                <Table.Row key={a.agent_id} className="hover:bg-kumo-tint">
                  <Table.Cell className="font-mono text-xs font-semibold">{a.agent_id}</Table.Cell>
                  <Table.Cell className="text-xs">{a.agent_name}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatNumber(a.event_count)}</Table.Cell>
                  <Table.Cell className="text-xs text-center font-mono">
                    {a.warmup_progress}/{a.warmup_required}
                  </Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatSeconds(a.baseline_gap_seconds)}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatSeconds(a.ema_gap_seconds)}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatSeconds(a.base_delta_t_seconds)}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono font-semibold text-kumo-brand">
                    {formatSeconds(a.current_delta_t_seconds)}
                  </Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{a.active_bucket_count}</Table.Cell>
                  <Table.Cell className="text-xs text-center">
                    <Badge variant={a.is_warmed_up ? 'success' : 'warning'}>
                      {a.status}
                    </Badge>
                  </Table.Cell>
                </Table.Row>
              ))}
              {agents.length === 0 && (
                <Table.Row>
                  <Table.Cell colSpan={10} className="p-6 text-center text-xs text-kumo-subtle">
                    No agent states active yet.
                  </Table.Cell>
                </Table.Row>
              )}
            </Table.Body>
          </Table>
        </div>
      </div>

      <div>
        <h2 className="text-sm font-semibold mb-3 text-kumo-default">
          Active Aggregation Buckets ({buckets.length})
        </h2>
        <div className="rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden shadow-xs">
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Agent</Table.Head>
                <Table.Head>Rule Group</Table.Head>
                <Table.Head>Opened At</Table.Head>
                <Table.Head>Last Alert</Table.Head>
                <Table.Head className="text-right">Count</Table.Head>
                <Table.Head className="text-right">Max Sev</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {buckets.map((b, idx) => (
                <Table.Row key={`${b.agent_id}-${b.rule_group_primary}-${idx}`} className="hover:bg-kumo-tint">
                  <Table.Cell className="text-xs font-mono">{b.agent_name ? `${b.agent_name} (${b.agent_id})` : b.agent_id}</Table.Cell>
                  <Table.Cell className="text-xs font-mono">{b.rule_group_primary}</Table.Cell>
                  <Table.Cell className="text-xs">{formatDateTime(b.start_time)}</Table.Cell>
                  <Table.Cell className="text-xs">{formatDateTime(b.end_time)}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono font-semibold">{b.alert_count}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{b.max_severity}</Table.Cell>
                </Table.Row>
              ))}
              {buckets.length === 0 && (
                <Table.Row>
                  <Table.Cell colSpan={6} className="p-6 text-center text-xs text-kumo-subtle">
                    No open buckets currently active in runtime memory.
                  </Table.Cell>
                </Table.Row>
              )}
            </Table.Body>
          </Table>
        </div>
      </div>
    </div>
  );
}
