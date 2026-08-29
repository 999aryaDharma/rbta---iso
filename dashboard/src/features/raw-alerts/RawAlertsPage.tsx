import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { fetchMetaAlertRawAlerts } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/utils';

export function RawAlertsPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const id = Number(metaId);

  const { data } = useQuery({
    queryKey: ['raw-alerts', id, page],
    queryFn: () => fetchMetaAlertRawAlerts(id, { page, page_size: 20 })
  });

  return (
    <div>
      <PageHeader title={`Raw Alerts for MetaAlert ${id}`} />
      <div className="rounded-[7px] border overflow-hidden" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <table className="w-full text-sm">
          <thead className="border-b" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
            <tr>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Alert ID</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Time</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Rule ID</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Level</th>
              <th className="text-left px-4 py-2" style={{ color: 'var(--text-tertiary)' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map(a => (
              <tr key={a.wazuh_alert_id} className="border-b cursor-pointer hover:bg-[var(--bg-hover)]" style={{ borderColor: 'var(--border-subtle)' }} onClick={() => navigate(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(a.wazuh_alert_id)}`)}>
                <td className="px-4 py-2 font-mono text-xs truncate max-w-[100px]">{a.wazuh_alert_id}</td>
                <td className="px-4 py-2 text-xs">{formatDateTime(a.timestamp)}</td>
                <td className="px-4 py-2 font-mono text-xs">{a.rule_id}</td>
                <td className="px-4 py-2 text-xs">{a.rule_level}</td>
                <td className="px-4 py-2 text-xs truncate max-w-[200px]">{a.rule_description}</td>
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
