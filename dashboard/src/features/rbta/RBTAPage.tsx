import { usePollingQuery } from '@/hooks/usePolling';
import { fetchAgents, fetchBuckets } from '@/api/dashboard';
import { MetricCard } from '@/components/shared/MetricCard';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatNumber } from '@/lib/utils';

export function RBTAPage() {
  const { data: agents = [] } = usePollingQuery(['agents'], fetchAgents, 5000);
  const { data: buckets = [] } = usePollingQuery(['buckets'], fetchBuckets, 5000);

  const activeAgents = agents.length;
  const warmedUp = agents.filter(a => a.is_warmed_up).length;
  const seenAlerts = agents.reduce((acc, a) => acc + a.event_count, 0);
  const activeBuckets = buckets.length;

  return (
    <div>
      <PageHeader title="RBTA Engine" description="Real-time Agent Temporal State" />
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Active Agents" value={formatNumber(activeAgents)} />
        <MetricCard label="Warmed-up Agents" value={formatNumber(warmedUp)} />
        <MetricCard label="Seen Alerts" value={formatNumber(seenAlerts)} />
        <MetricCard label="Active Buckets" value={formatNumber(activeBuckets)} />
      </div>

      <h2 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Agent States</h2>
      <div className="rounded-[7px] border overflow-hidden mb-6" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Agent ID</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Status</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Events</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Warmup</th>
            </tr>
          </thead>
          <tbody>
            {agents.map(a => (
              <tr key={a.agent_id} className="border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="px-4 py-2 font-mono text-xs">{a.agent_id}</td>
                <td className="px-4 py-2">{a.status}</td>
                <td className="px-4 py-2">{a.event_count}</td>
                <td className="px-4 py-2">{(a.warmup_progress * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Active Buckets</h2>
      <div className="rounded-[7px] border overflow-hidden" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Meta ID</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
              <th className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-tertiary)' }}>Alerts</th>
            </tr>
          </thead>
          <tbody>
            {buckets.map(b => (
              <tr key={b.meta_id} className="border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="px-4 py-2 font-mono text-xs">{b.meta_id}</td>
                <td className="px-4 py-2">{b.agent_id}</td>
                <td className="px-4 py-2 font-mono text-xs">{b.rule_group_primary}</td>
                <td className="px-4 py-2">{b.alert_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
