import { useQuery } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { fetchRawAlert } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';

export function RawAlertDetailPage() {
  const { alertId } = useParams();
  const { data } = useQuery({
    queryKey: ['raw-alert', alertId],
    queryFn: () => fetchRawAlert(alertId as string)
  });

  if (!data) return <div>Loading...</div>;

  return (
    <div>
      <PageHeader title="Raw Alert Detail" description={data.wazuh_alert_id} />
      <div className="grid grid-cols-2 gap-6">
        <div className="p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Details</h3>
          <p className="text-sm">Rule: <span className="font-mono">{data.rule_id} (Level {data.rule_level})</span></p>
          <p className="text-sm">Group: <span className="font-mono">{data.rule_group_primary}</span></p>
          <p className="text-sm">Description: {data.rule_description}</p>
        </div>
        <div className="p-4 border rounded-[7px] overflow-auto max-h-[600px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
          <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>JSON Profile</h3>
          <pre className="text-xs font-mono">{JSON.stringify(data, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
