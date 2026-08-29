import { useQueryClient } from '@tanstack/react-query';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchReplayStatus, startReplay, pauseReplay, resumeReplay, stopReplay, resetReplay } from '@/api/replay';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber } from '@/lib/utils';
import { useState } from 'react';

export function ReplayPage() {
  const queryClient = useQueryClient();
  const { data: status } = usePollingQuery(['replay'], fetchReplayStatus, 1000);
  const [dataset, setDataset] = useState('demo_dataset.json');

  const handleAction = async (action: () => Promise<any>) => {
    await action();
    queryClient.invalidateQueries({ queryKey: ['replay'] });
  };

  return (
    <div>
      <PageHeader title="Demonstration Replay" description="Control synthetic workload playback" />
      <div className="flex gap-4 mb-6 p-4 border rounded-[7px]" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <select value={dataset} onChange={e => setDataset(e.target.value)} className="border p-2 rounded-[5px] text-sm" style={{ borderColor: 'var(--border-default)' }}>
          <option value="demo_dataset.json">Demo Dataset</option>
        </select>
        <button onClick={() => handleAction(() => startReplay(dataset, 'MAX'))} className="px-4 py-2 text-white rounded-[5px] text-sm cursor-pointer" style={{ background: 'var(--action-blue)' }}>Start</button>
        <button onClick={() => handleAction(pauseReplay)} className="px-4 py-2 border rounded-[5px] text-sm cursor-pointer" style={{ borderColor: 'var(--border-default)' }}>Pause</button>
        <button onClick={() => handleAction(resumeReplay)} className="px-4 py-2 border rounded-[5px] text-sm cursor-pointer" style={{ borderColor: 'var(--border-default)' }}>Resume</button>
        <button onClick={() => handleAction(stopReplay)} className="px-4 py-2 text-white rounded-[5px] text-sm cursor-pointer" style={{ background: 'var(--danger)' }}>Stop</button>
        <button onClick={() => handleAction(resetReplay)} className="px-4 py-2 border rounded-[5px] text-sm cursor-pointer" style={{ borderColor: 'var(--border-default)' }}>Reset</button>
      </div>
      
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Status" value={status?.status || 'UNKNOWN'} />
        <MetricCard label="Processed" value={status ? formatNumber(status.processed_count) : 0} />
        <MetricCard label="Total" value={status ? formatNumber(status.total_count) : 0} />
        <MetricCard label="Events/sec" value={status ? formatNumber(status.events_per_second) : 0} />
      </div>
    </div>
  );
}
