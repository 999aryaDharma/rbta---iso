import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlertRawAlerts } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/utils';
import { Alert } from '@/components/ui/alert';
import { Search, ChevronLeft, ChevronRight, ArrowLeft, AlertTriangle } from 'lucide-react';

export function RawAlertsPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const search = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');
  const id = Number(metaId);

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data } = useQuery({
    queryKey: ['raw-alerts', id, page, search, runId || 'live'],
    queryFn: () =>
      fetchMetaAlertRawAlerts(id, {
        page,
        page_size: 20,
        search: search || undefined,
        run_id: runId || undefined,
      }),
  });

  const handleSearchChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('search', val);
    else params.delete('search');
    params.set('page', '1');
    setSearchParams(params);
  };

  const totalPages = data ? Math.ceil(data.filtered_total / 20) || 1 : 1;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
        <Link to={withRunId('/meta-alerts')} className="hover:underline">MetaAlerts</Link>
        <span>/</span>
        <Link to={withRunId(`/meta-alerts/${id}`)} className="hover:underline">#{id}</Link>
        <span>/</span>
        <span style={{ color: 'var(--text-primary)' }}>Raw Alerts</span>
      </div>

      <PageHeader
        title={`Member Raw Alerts for MetaAlert #${id}`}
        description="Individual security log events aggregated into this temporal cluster"
        actions={
          <button
            onClick={() => navigate(withRunId(`/meta-alerts/${id}`))}
            className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium cursor-pointer"
            style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-secondary)' }}
          >
            <ArrowLeft size={14} /> Back to MetaAlert
          </button>
        }
      />

      {/* Unresolved Evidence Warning Banner */}
      {data && data.unresolved_alert_ids && data.unresolved_alert_ids.length > 0 && (
        <Alert variant="warning" className="mb-4">
          <AlertTriangle size={16} className="shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">
              Partial Local Evidence: {data.resolved_total} of {data.source_total} source alerts resolved.
            </div>
            <div className="mt-0.5">
              {data.unresolved_alert_ids.length} source alert(s) remain referenced by the MetaAlert trace but local audit evidence is unavailable:
              <span className="font-mono ml-1 font-medium">{data.unresolved_alert_ids.join(', ')}</span>
            </div>
          </div>
        </Alert>
      )}

      {/* Filter Toolbar */}
      <div
        className="p-4 rounded-[7px] border mb-6 flex flex-wrap items-center justify-between gap-3"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="relative flex-1 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
          <input
            type="text"
            placeholder="Search Alert ID, Rule ID, Description, IP..."
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 border rounded-[5px] text-xs"
            style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
          />
        </div>

        <div className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
          {data ? `Showing page ${page} of ${totalPages} (${data.filtered_total} matching / ${data.source_total} total)` : 'Loading...'}
        </div>
      </div>

      {/* Raw Alerts Table */}
      <div
        className="rounded-[7px] border overflow-hidden"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Timestamp</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Wazuh Alert ID</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Rule ID</th>
              <th className="text-right px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Level</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Description</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>Source IP</th>
              <th className="text-left px-4 py-2.5 font-medium text-xs" style={{ color: 'var(--text-tertiary)' }}>MITRE Tactics</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((a) => (
              <tr
                key={a.wazuh_alert_id}
                className="border-b cursor-pointer hover:bg-[var(--bg-hover)]"
                style={{ borderColor: 'var(--border-subtle)' }}
                onClick={() =>
                  navigate(withRunId(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(a.wazuh_alert_id)}`))
                }
              >
                <td className="px-4 py-2.5 text-xs font-mono">{formatDateTime(a.timestamp)}</td>
                <td className="px-4 py-2.5 font-mono text-xs font-semibold text-blue-600 truncate max-w-[140px]">
                  {a.wazuh_alert_id}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs">{a.rule_id}</td>
                <td className="px-4 py-2.5 text-xs text-right font-mono font-semibold">{a.rule_level}</td>
                <td className="px-4 py-2.5 text-xs truncate max-w-[220px]">{a.rule_description}</td>
                <td className="px-4 py-2.5 font-mono text-xs">{a.srcip || '—'}</td>
                <td className="px-4 py-2.5 text-xs">
                  {a.mitre_tactics && a.mitre_tactics.length > 0 ? (
                    <span className="px-1.5 py-0.5 rounded-[3px] text-[11px] font-mono border" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                      {a.mitre_tactics.join(', ')}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-disabled)' }}>None</span>
                  )}
                </td>
              </tr>
            ))}
            {data && data.items.length === 0 && (
              <tr>
                <td colSpan={7} className="p-6 text-center text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  No matching raw alerts found.
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
