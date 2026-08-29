import { useQuery } from '@tanstack/react-query';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime } from '@/lib/utils';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Search, ChevronLeft, ChevronRight } from 'lucide-react';

export function MetaAlertsPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const decision = searchParams.get('decision') || '';
  const search = searchParams.get('search') || '';

  const { data } = useQuery({
    queryKey: ['meta-alerts', page, decision, search],
    queryFn: () =>
      fetchMetaAlerts({
        page,
        page_size: 20,
        decision: decision || undefined,
        search: search || undefined,
        sort_by: 'end_time',
        sort_order: 'desc',
      }),
  });

  const handleDecisionChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('decision', val);
    else params.delete('decision');
    params.set('page', '1');
    setSearchParams(params);
  };

  const handleSearchChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('search', val);
    else params.delete('search');
    params.set('page', '1');
    setSearchParams(params);
  };

  return (
    <div>
      <PageHeader
        title="MetaAlerts"
        description="Aggregated security alert clusters scored by Isolation Forest with deterministic Tukey IQR thresholding"
      />

      {/* Toolbar / Filters */}
      <div
        className="p-4 rounded-[7px] border mb-6 flex flex-wrap items-center justify-between gap-3"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex items-center gap-3 flex-1 min-w-[280px]">
          <div className="relative flex-1 max-w-sm">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
            <input
              type="text"
              placeholder="Search Meta ID, Agent, Rule Group..."
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 border rounded-[5px] text-xs bg-white"
              style={{ borderColor: 'var(--border-default)' }}
            />
          </div>

          <select
            value={decision}
            onChange={(e) => handleDecisionChange(e.target.value)}
            className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-white"
            style={{ borderColor: 'var(--border-default)' }}
          >
            <option value="">All Decisions</option>
            <option value="CRITICAL">CRITICAL (Escalate)</option>
            <option value="SUSPICIOUS">SUSPICIOUS (Escalate)</option>
            <option value="CONTEXTUAL_ANOMALY">CONTEXTUAL_ANOMALY (Escalate)</option>
            <option value="NOISE_HIGH">NOISE_HIGH (Daily Digest)</option>
            <option value="NOISE">NOISE (Suppress)</option>
          </select>
        </div>

        <div className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
          {data ? `Showing page ${page} of ${Math.ceil(data.total / 20) || 1} (${data.total} total)` : 'Loading...'}
        </div>
      </div>

      {/* MetaAlerts Table */}
      <div
        className="rounded-[7px] border overflow-hidden"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Meta ID</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>End Time</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Agent</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Rule Group</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Raw Count</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Max Sev</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Score</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Threshold</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Decision</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((m) => (
              <tr
                key={m.meta_id}
                className="border-b cursor-pointer hover:bg-[var(--bg-hover)]"
                style={{ borderColor: 'var(--border-subtle)' }}
                onClick={() => navigate(`/meta-alerts/${m.meta_id}`)}
              >
                <td className="px-4 py-2.5 font-mono text-xs font-semibold">{m.meta_id}</td>
                <td className="px-4 py-2.5 text-xs">{formatDateTime(m.end_time)}</td>
                <td className="px-4 py-2.5 text-xs">{m.agent_name} ({m.agent_id})</td>
                <td className="px-4 py-2.5 font-mono text-xs">{m.rule_group_primary}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold">{m.alert_count}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{m.max_severity}/15</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{m.anomaly_score.toFixed(4)}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono" style={{ color: 'var(--text-tertiary)' }}>
                  {m.threshold_used.toFixed(4)}
                </td>
                <td className="px-4 py-2.5"><DecisionBadge decision={m.decision} action={m.action} /></td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination Bar */}
        <div
          className="p-4 flex items-center justify-between border-t"
          style={{ borderColor: 'var(--border-default)' }}
        >
          <button
            disabled={page <= 1}
            onClick={() => {
              const params = new URLSearchParams(searchParams);
              params.set('page', String(page - 1));
              setSearchParams(params);
            }}
            className="flex items-center gap-1 px-3 py-1.5 border rounded-[5px] text-xs font-medium bg-white disabled:opacity-50 cursor-pointer"
            style={{ borderColor: 'var(--border-default)' }}
          >
            <ChevronLeft size={14} /> Previous
          </button>

          <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
            Page {page} of {Math.ceil((data?.total ?? 0) / 20) || 1}
          </span>

          <button
            disabled={!data || page >= Math.ceil(data.total / 20)}
            onClick={() => {
              const params = new URLSearchParams(searchParams);
              params.set('page', String(page + 1));
              setSearchParams(params);
            }}
            className="flex items-center gap-1 px-3 py-1.5 border rounded-[5px] text-xs font-medium bg-white disabled:opacity-50 cursor-pointer"
            style={{ borderColor: 'var(--border-default)' }}
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
