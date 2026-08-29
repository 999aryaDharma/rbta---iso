import { useSearchParams } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchAgents, fetchBuckets } from '@/api/dashboard';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatNumber, formatSeconds } from '@/lib/utils';

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
    <div>
      <PageHeader
        title="RBTA Engine Telemetry"
        description="Real-time agent temporal state, dynamic aggregation windows, and active open buckets"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Active Agents" value={formatNumber(activeAgents)} />
        <MetricCard label="Warmed-up Agents" value={formatNumber(warmedUp)} />
        <MetricCard label="Seen Alerts" value={formatNumber(seenAlerts)} />
        <MetricCard label="Open Active Buckets" value={formatNumber(activeBuckets)} />
      </div>

      <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
        Agent Temporal States ({agents.length})
      </h2>
      <div
        className="rounded-[7px] border overflow-hidden mb-6"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Agent ID</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Name</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Events</th>
              <th className="text-center px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Warmup</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Baseline Gap</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>EMA Gap</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Base Δt</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Current Δt</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Buckets</th>
              <th className="text-center px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a) => (
              <tr key={a.agent_id} className="border-b hover:bg-[var(--bg-hover)]" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="px-4 py-2.5 font-mono text-xs font-semibold">{a.agent_id}</td>
                <td className="px-4 py-2.5 text-xs">{a.agent_name}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{formatNumber(a.event_count)}</td>
                <td className="px-4 py-2.5 text-xs text-center font-mono">
                  {a.warmup_progress}/{a.warmup_required}
                </td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{formatSeconds(a.baseline_gap_seconds)}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{formatSeconds(a.ema_gap_seconds)}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{formatSeconds(a.base_delta_t_seconds)}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold" style={{ color: 'var(--brand-orange)' }}>
                  {formatSeconds(a.current_delta_t_seconds)}
                </td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{a.active_bucket_count}</td>
                <td className="px-4 py-2.5 text-xs text-center">
                  <span
                    className="px-2 py-0.5 rounded-[3px] text-[11px] font-semibold tracking-wide"
                    style={{
                      background: a.is_warmed_up ? 'var(--success-soft)' : 'var(--warning-soft)',
                      color: a.is_warmed_up ? 'var(--success)' : 'var(--warning)',
                    }}
                  >
                    {a.status}
                  </span>
                </td>
              </tr>
            ))}
            {agents.length === 0 && (
              <tr>
                <td colSpan={10} className="p-6 text-center text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  No agent states active yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
        Active Aggregation Buckets ({buckets.length})
      </h2>
      <div
        className="rounded-[7px] border overflow-hidden"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Meta ID</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Alert Count</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Max Severity</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Window Start</th>
            </tr>
          </thead>
          <tbody>
            {buckets.map((b) => (
              <tr key={b.meta_id || b.agent_id} className="border-b hover:bg-[var(--bg-hover)]" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="px-4 py-2.5 font-mono text-xs font-semibold">#{b.meta_id}</td>
                <td className="px-4 py-2.5 text-xs">{b.agent_name} ({b.agent_id})</td>
                <td className="px-4 py-2.5 font-mono text-xs">{b.rule_group_primary}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold">{b.alert_count}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{b.max_severity}/15</td>
                <td className="px-4 py-2.5 text-xs font-mono">{b.start_time || '—'}</td>
              </tr>
            ))}
            {buckets.length === 0 && (
              <tr>
                <td colSpan={6} className="p-6 text-center text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  No open aggregation buckets currently active.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
