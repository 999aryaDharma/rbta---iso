import { Activity } from 'lucide-react';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary } from '@/api/dashboard';

export function Topbar() {
  const { data } = usePollingQuery(['summary-topbar'], fetchSummary, 3000);

  return (
    <header
      className="h-14 flex items-center justify-between px-4 border-b shrink-0"
      style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
    >
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Activity size={18} style={{ color: 'var(--brand-orange)' }} />
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>RBTA</span>
        </div>
        {data && (
          <div className="flex items-center gap-3 ml-4 text-xs" style={{ color: 'var(--text-tertiary)' }}>
            <span className="font-mono">{data.model_version}</span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: data.ready ? 'var(--success)' : 'var(--danger)' }}
              />
              {data.ready ? 'READY' : 'NOT READY'}
            </span>
            <span>{data.source_mode}</span>
          </div>
        )}
      </div>
    </header>
  );
}
