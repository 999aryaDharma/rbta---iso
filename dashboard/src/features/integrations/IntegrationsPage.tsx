import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSystemInfo } from '@/api/dashboard';
import { PageHeader } from '@/components/shared/PageHeader';

export function IntegrationsPage() {
  const { data } = usePollingQuery(['system'], fetchSystemInfo, 5000);
  
  return (
    <div>
      <PageHeader title="Integrations" description="Pipeline components" />
      <div className="p-6 border rounded-[7px] flex flex-col gap-4 max-w-xl" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <div className="p-4 border rounded-[5px] text-center font-mono" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>Wazuh Source (DEFERRED)</div>
        <div className="text-center">↓</div>
        <div className="p-4 border rounded-[5px] text-center font-mono" style={{ background: 'var(--action-blue-soft)', borderColor: 'var(--action-blue)' }}>RBTA Engine</div>
        <div className="text-center">↓</div>
        <div className="p-4 border rounded-[5px] text-center font-mono" style={{ background: 'var(--brand-orange-soft)', borderColor: 'var(--brand-orange)' }}>Isolation Forest (v{data?.model_version})</div>
        <div className="text-center">↓</div>
        <div className="p-4 border rounded-[5px] text-center font-mono" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>Outbox (Depth: {data?.outbox_depth})</div>
        <div className="text-center">↓</div>
        <div className="p-4 border rounded-[5px] text-center font-mono" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>Telegram Alerts</div>
      </div>
    </div>
  );
}
