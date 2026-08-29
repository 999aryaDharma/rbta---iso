import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';

export function SystemPage() {
  const { data } = usePollingQuery(['system'], fetchSystemInfo, 5000);

  return (
    <div>
      <PageHeader title="System Information" description="Backend operational status" />
      {data && (
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>API Status</p>
            <p className="font-mono">{data.api_status}</p>
          </div>
          <div className="p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>Model Version</p>
            <p className="font-mono">{data.model_version}</p>
          </div>
          <div className="p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>Schema Version</p>
            <p className="font-mono">{data.feature_schema_version}</p>
          </div>
          <div className="p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>Threshold</p>
            <p className="font-mono">{data.threshold}</p>
          </div>
        </div>
      )}
    </div>
  );
}
