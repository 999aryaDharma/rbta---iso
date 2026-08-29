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
  const action = searchParams.get('action') || '';
  const search = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data } = useQuery({
    queryKey: ['meta-alerts', page, decision, action, search, runId || 'live'],
    queryFn: () =>
      fetchMetaAlerts({
        page,
        page_size: 20,
        decision: decision || undefined,
        action: action || undefined,
        search: search || undefined,
        sort_by: 'end_time',
        sort_order: 'desc',
        run_id: runId || undefined,
      }),
  });

  const handleDecisionChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('decision', val);
    else params.delete('decision');
    params.set('page', '1');
    setSearchParams(params);
  };

  const handleActionChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('action', val);
    else params.delete('action');
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

  const totalPages = data ? Math.ceil(data.total / 20) || 1 : 1;

  return (
    <div>
      <PageHeader
        title="MetaAlerts Investigation Table"
        description="Aggregated security alert clusters scored by Isolation Forest with deterministic Tukey IQR thresholding"
      />

      {/* Toolbar / Filters */}
      <div
        className="p-4 rounded-[7px] border mb-6 flex flex-wrap items-center justify-between gap-3"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
          <div className="relative flex-1 max-w-sm">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
            <input
              type="text"
              placeholder="Search Meta ID, Agent, Rule Group..."
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 border rounded-[5px] text-xs bg-[var(--bg-surface)]"
              style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
            />
          </div>

          <select
            value={action}
            onChange={(e) => handleActionChange(e.target.value)}
            className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-[var(--bg-surface)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
          >
            <option value="">All Actions</option>
            <option value="ESCALATE">ESCALATE</option>
            <option value="DAILY_DIGEST">DAILY_DIGEST</option>
            <option value="SUPPRESS">SUPPRESS</option>
          </select>

          <select
            value={decision}
            onChange={(e) => handleDecisionChange(e.target.value)}
            className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-[var(--bg-surface)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
          >
            <option value="">All Decisions</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="SUSPICIOUS">SUSPICIOUS</option>
            <option value="CONTEXTUAL_ANOMALY">CONTEXTUAL_ANOMALY</option>
            <option value="NOISE_HIGH">NOISE_HIGH</option>
            <option value="NOISE">NOISE</option>
          </select>
        </div>

        <div className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
          {data ? `Showing page ${page} of ${totalPages} (${data.total} total)` : 'Loading...'}
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
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Primary Group</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Raw Count</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Max Sev</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Score</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Decision</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((m) => (
              <tr
                key={m.meta_id}
                className="border-b cursor-pointer hover:bg-[var(--bg-hover)]"
                style={{ borderColor: 'var(--border-subtle)' }}
                onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
              >
                <td className="px-4 py-2.5 font-mono text-xs font-semibold">#{m.meta_id}</td>
                <td className="px-4 py-2.5 text-xs">{formatDateTime(m.end_time)}</td>
                <td className="px-4 py-2.5 text-xs">{m.agent_name} ({m.agent_id})</td>
                <td className="px-4 py-2.5 text-xs font-mono">{m.rule_group_primary}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold">{m.alert_count}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{m.max_severity}/15</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono">{m.anomaly_score.toFixed(4)}</td>
                <td className="px-4 py-2.5"><DecisionBadge decision={m.decision} action={m.action} /></td>
              </tr>
            ))}
            {data && data.items.length === 0 && (
              <tr>
                <td colSpan={8} className="p-8 text-center text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  No MetaAlerts match the specified filters.
                </td>
              </tr>
            )}
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
            className="flex items-center gap-1 px-3 py-1.5 border rounded-[5px] text-xs font-medium disabled:opacity-50 cursor-pointer"
            style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
          >
            <ChevronLeft size={14} /> Previous
          </button>

          <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
            Page {page} of {totalPages}
          </span>

          <button
            disabled={!data || page >= totalPages}
            onClick={() => {
              const params = new URLSearchParams(searchParams);
              params.set('page', String(page + 1));
              setSearchParams(params);
            }}
            className="flex items-center gap-1 px-3 py-1.5 border rounded-[5px] text-xs font-medium disabled:opacity-50 cursor-pointer"
            style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
