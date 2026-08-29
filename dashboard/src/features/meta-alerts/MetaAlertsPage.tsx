import { useQuery } from '@tanstack/react-query';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime } from '@/lib/utils';
import { useNavigate, useSearchParams } from 'react-router-dom';

export function MetaAlertsPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);

  const { data } = useQuery({
    queryKey: ['meta-alerts', page],
    queryFn: () => fetchMetaAlerts({ page, page_size: 20 })
  });

  return (
    <div>
      <PageHeader title="MetaAlerts" description="Aggregated and scored alerts" />
      <div className="rounded-[7px] border overflow-hidden" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>ID</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Time</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
              <th className="text-right px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Score</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Decision</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map(m => (
              <tr key={m.meta_id} className="border-b cursor-pointer hover:bg-[var(--bg-hover)]" style={{ borderColor: 'var(--border-subtle)' }} onClick={() => navigate(`/meta-alerts/${m.meta_id}`)}>
                <td className="px-4 py-2 font-mono text-xs">{m.meta_id}</td>
                <td className="px-4 py-2 text-xs">{formatDateTime(m.end_time)}</td>
                <td className="px-4 py-2 text-xs">{m.agent_id}</td>
                <td className="px-4 py-2 font-mono text-xs">{m.rule_group_primary}</td>
                <td className="px-4 py-2 text-right font-mono text-xs">{m.anomaly_score.toFixed(4)}</td>
                <td className="px-4 py-2"><DecisionBadge decision={m.decision} action={m.action} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="p-4 flex gap-2 border-t" style={{ borderColor: 'var(--border-default)' }}>
          <button disabled={page <= 1} onClick={() => setSearchParams({ page: String(page - 1) })} className="px-3 py-1 border rounded-[5px] text-sm" style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}>Prev</button>
          <button disabled={!data || data.items.length < 20} onClick={() => setSearchParams({ page: String(page + 1) })} className="px-3 py-1 border rounded-[5px] text-sm" style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}>Next</button>
        </div>
      </div>
    </div>
  );
}
