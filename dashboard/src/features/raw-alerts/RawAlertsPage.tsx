import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlertRawAlerts } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/utils';
import { Search, ChevronLeft, ChevronRight, ArrowLeft } from 'lucide-react';

export function RawAlertsPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const search = searchParams.get('search') || '';
  const id = Number(metaId);

  const { data } = useQuery({
    queryKey: ['raw-alerts', id, page, search],
    queryFn: () =>
      fetchMetaAlertRawAlerts(id, {
        page,
        page_size: 20,
        search: search || undefined,
      }),
  });

  const handleSearchChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('search', val);
    else params.delete('search');
    params.set('page', '1');
    setSearchParams(params);
  };

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
        <Link to="/meta-alerts" className="hover:underline">MetaAlerts</Link>
        <span>/</span>
        <Link to={`/meta-alerts/${id}`} className="hover:underline">#{id}</Link>
        <span>/</span>
        <span style={{ color: 'var(--text-primary)' }}>Raw Alerts</span>
      </div>

      <PageHeader
        title={`Member Raw Alerts for MetaAlert #${id}`}
        description="Individual security log events aggregated into this temporal cluster"
        actions={
          <button
            onClick={() => navigate(`/meta-alerts/${id}`)}
            className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium bg-white cursor-pointer"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
          >
            <ArrowLeft size={14} /> Back to MetaAlert
          </button>
        }
      />

      {/* Filter Toolbar */}
      <div
        className="p-4 rounded-[7px] border mb-6 flex items-center justify-between gap-3"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="relative flex-1 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
          <input
            type="text"
            placeholder="Search Alert ID, Rule ID, Description, IP..."
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 border rounded-[5px] text-xs bg-white"
            style={{ borderColor: 'var(--border-default)' }}
          />
        </div>

        <div className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
          {data ? `Showing page ${page} of ${Math.ceil(data.total / 20) || 1} (${data.total} member alerts)` : 'Loading...'}
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
                  navigate(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(a.wazuh_alert_id)}`)
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
                  {a.mitre_tactics.length > 0 ? (
                    <span className="px-1.5 py-0.5 rounded-[3px] text-[11px] font-mono border" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                      {a.mitre_tactics.join(', ')}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-disabled)' }}>None</span>
                  )}
                </td>
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
