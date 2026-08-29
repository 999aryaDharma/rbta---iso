import { useQueryClient } from '@tanstack/react-query';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchReplayStatus, startReplay, pauseReplay, resumeReplay, stopReplay, resetReplay } from '@/api/replay';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { formatNumber, formatDuration } from '@/lib/utils';
import { useState } from 'react';
import { Play, Pause, Square, RotateCcw, FastForward } from 'lucide-react';

export function ReplayPage() {
  const queryClient = useQueryClient();
  const { data: status } = usePollingQuery(['replay'], fetchReplayStatus, 1000);
  const [dataset, setDataset] = useState('demo_dataset.json');
  const [speed, setSpeed] = useState<'1' | '10' | '100' | 'MAX'>('MAX');
  const [isLoading, setIsLoading] = useState(false);

  const handleAction = async (action: () => Promise<any>) => {
    setIsLoading(true);
    try {
      await action();
      await queryClient.invalidateQueries({ queryKey: ['replay'] });
    } finally {
      setIsLoading(false);
    }
  };

  const isIdle = !status || status.status === 'IDLE' || status.status === 'COMPLETED' || status.status === 'ERROR';
  const isRunning = status?.status === 'RUNNING';
  const isPaused = status?.status === 'PAUSED';
  const progress = status && status.total_count > 0 ? (status.processed_count / status.total_count) * 100 : 0;

  return (
    <div>
      <PageHeader
        title="Demonstration Replay Controller"
        description="Deterministic replay streaming with calibrated playback speed, state isolation, and dual evidence logging"
      />

      {/* Control Panel Card */}
      <div
        className="p-5 rounded-[7px] border mb-6"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div>
              <label className="block text-[11px] font-semibold mb-1" style={{ color: 'var(--text-tertiary)' }}>
                Workload Dataset
              </label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={!isIdle || isLoading}
                className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-white"
                style={{ borderColor: 'var(--border-default)' }}
              >
                <option value="demo_dataset.json">demo_dataset.json (Evaluation Set)</option>
              </select>
            </div>

            <div>
              <label className="block text-[11px] font-semibold mb-1" style={{ color: 'var(--text-tertiary)' }}>
                Speed Multiplier
              </label>
              <select
                value={speed}
                onChange={(e) => setSpeed(e.target.value as any)}
                disabled={!isIdle || isLoading}
                className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-white"
                style={{ borderColor: 'var(--border-default)' }}
              >
                <option value="1">1x (Realtime)</option>
                <option value="10">10x Throttled</option>
                <option value="100">100x Fast</option>
                <option value="MAX">MAX (Unthrottled)</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {isIdle && (
              <button
                onClick={() => handleAction(() => startReplay(dataset, speed))}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-4 py-2 text-white rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ background: 'var(--action-blue)' }}
              >
                <Play size={14} /> Start Replay
              </button>
            )}

            {isRunning && (
              <button
                onClick={() => handleAction(pauseReplay)}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-4 py-2 border rounded-[5px] text-xs font-medium cursor-pointer bg-white"
                style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
              >
                <Pause size={14} /> Pause
              </button>
            )}

            {isPaused && (
              <button
                onClick={() => handleAction(resumeReplay)}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-4 py-2 text-white rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ background: 'var(--brand-orange)' }}
              >
                <FastForward size={14} /> Resume
              </button>
            )}

            {(isRunning || isPaused) && (
              <button
                onClick={() => handleAction(stopReplay)}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-4 py-2 text-white rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ background: 'var(--danger)' }}
              >
                <Square size={14} /> Stop
              </button>
            )}

            <button
              onClick={() => handleAction(resetReplay)}
              disabled={isLoading || status?.status === 'IDLE'}
              className="flex items-center gap-1.5 px-3 py-2 border rounded-[5px] text-xs font-medium cursor-pointer bg-white"
              style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
            >
              <RotateCcw size={14} /> Reset
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {status && status.total_count > 0 && (
          <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex justify-between text-xs mb-1 font-mono">
              <span style={{ color: 'var(--text-tertiary)' }}>Replay Progress</span>
              <span className="font-semibold">{progress.toFixed(1)}%</span>
            </div>
            <div className="w-full h-2 rounded-[3px] overflow-hidden" style={{ background: 'var(--bg-subtle)' }}>
              <div
                className="h-full transition-all duration-300"
                style={{ width: `${progress}%`, background: 'var(--brand-orange)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Telemetry Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Playback Status" value={status?.status || 'IDLE'} />
        <MetricCard label="Processed Events" value={status ? formatNumber(status.processed_count) : 0} />
        <MetricCard label="Total Events" value={status ? formatNumber(status.total_count) : 0} />
        <MetricCard label="Throughput" value={status ? `${formatNumber(status.events_per_second)} ev/s` : '0 ev/s'} />
      </div>

      {/* Replay Details */}
      {status && status.run_id && (
        <div
          className="p-5 rounded-[7px] border text-xs font-mono space-y-2"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Active Run ID:</span>
            <span className="font-semibold">{status.run_id}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Dataset Path:</span>
            <span>{status.dataset}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Elapsed Time:</span>
            <span>{formatDuration(status.wall_clock_elapsed_seconds)}</span>
          </div>
          {status.error && (
            <div className="flex justify-between text-red-600">
              <span>Error:</span>
              <span>{status.error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
