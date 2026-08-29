import React, { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import {
  fetchReplayDatasets,
  fetchReplayStatus,
  startReplay,
  pauseReplay,
  resumeReplay,
  stopReplay,
  resetReplay,
} from '@/api/replay';
import { PageHeader } from '@/components/shared/PageHeader';
import { MetricCard } from '@/components/shared/MetricCard';
import { Dialog } from '@/components/ui/dialog';
import { Alert } from '@/components/ui/alert';
import { formatNumber, formatDuration } from '@/lib/utils';
import { Play, Pause, Square, RotateCcw, FastForward, AlertTriangle, ArrowRight, CheckCircle2 } from 'lucide-react';

export function ReplayPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: datasetsData } = useQuery({ queryKey: ['replay-datasets'], queryFn: fetchReplayDatasets });
  const { data: status } = usePollingQuery(['replay'], fetchReplayStatus, 1000);

  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [speed, setSpeed] = useState<'1' | '10' | '100' | 'MAX'>('MAX');
  const [isLoading, setIsLoading] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Set default dataset once loaded
  React.useEffect(() => {
    if (datasetsData?.items && datasetsData.items.length > 0 && !selectedDataset) {
      setSelectedDataset(datasetsData.items[0].name);
    }
  }, [datasetsData, selectedDataset]);

  // Sync run_id with URL when replay is active
  React.useEffect(() => {
    if (status?.run_id && searchParams.get('run_id') !== status.run_id) {
      const params = new URLSearchParams(searchParams);
      params.set('run_id', status.run_id);
      setSearchParams(params, { replace: true });
    }
  }, [status?.run_id, searchParams, setSearchParams]);

  const handleAction = async (action: () => Promise<any>) => {
    setIsLoading(true);
    try {
      await action();
      await queryClient.invalidateQueries({ queryKey: ['replay'] });
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfirmReset = async () => {
    setShowResetConfirm(false);
    await handleAction(resetReplay);
    const params = new URLSearchParams(searchParams);
    params.delete('run_id');
    setSearchParams(params, { replace: true });
  };

  const isIdle = !status || status.status === 'IDLE' || status.status === 'COMPLETED' || status.status === 'ERROR' || status.status === 'STOPPED';
  const isRunning = status?.status === 'RUNNING';
  const isPaused = status?.status === 'PAUSED';
  const progressPercent = status && status.total_count > 0 ? (status.processed_count / status.total_count) * 100 : 0;

  const withRunId = (path: string) => (status?.run_id ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(status.run_id)}` : path);

  return (
    <div>
      <PageHeader
        title="Demonstration Replay Controller"
        description="Deterministic historical workload replay streaming with calibrated pacing, session isolation, and strict evidence logging"
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
                Replay Dataset (.jsonl)
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={!isIdle || isLoading || !datasetsData?.items?.length}
                className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-[var(--bg-surface)]"
                style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
              >
                {datasetsData?.items && datasetsData.items.length > 0 ? (
                  datasetsData.items.map((ds) => (
                    <option key={ds.name} value={ds.name}>
                      {ds.name} ({Math.round(ds.size_bytes / 1024)} KB)
                    </option>
                  ))
                ) : (
                  <option value="">No replay datasets available</option>
                )}
              </select>
            </div>

            <div>
              <label className="block text-[11px] font-semibold mb-1" style={{ color: 'var(--text-tertiary)' }}>
                Playback Speed
              </label>
              <select
                value={speed}
                onChange={(e) => setSpeed(e.target.value as any)}
                disabled={!isIdle || isLoading}
                className="px-3 py-1.5 border rounded-[5px] text-xs font-mono bg-[var(--bg-surface)]"
                style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
              >
                <option value="1">1x (Realtime Clock)</option>
                <option value="10">10x Throttled</option>
                <option value="100">100x Fast Pacing</option>
                <option value="MAX">MAX (Unthrottled Throughput)</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {isIdle && (
              <button
                onClick={() => handleAction(() => startReplay(selectedDataset, speed))}
                disabled={isLoading || !selectedDataset}
                className="flex items-center gap-1.5 px-4 py-2 text-white rounded-[5px] text-xs font-medium cursor-pointer disabled:opacity-50"
                style={{ background: 'var(--action-blue)' }}
              >
                <Play size={14} /> Start Replay
              </button>
            )}

            {isRunning && (
              <button
                onClick={() => handleAction(pauseReplay)}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-4 py-2 border rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
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
              onClick={() => setShowResetConfirm(true)}
              disabled={isLoading || status?.status === 'IDLE'}
              className="flex items-center gap-1.5 px-3 py-2 border rounded-[5px] text-xs font-medium cursor-pointer disabled:opacity-40"
              style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-secondary)' }}
            >
              <RotateCcw size={14} /> Reset New Run
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {status && status.total_count > 0 && (
          <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex justify-between text-xs mb-1 font-mono">
              <span style={{ color: 'var(--text-tertiary)' }}>Replay Progress</span>
              <span className="font-semibold">{progressPercent.toFixed(1)}%</span>
            </div>
            <div className="w-full h-2 rounded-[3px] overflow-hidden" style={{ background: 'var(--bg-subtle)' }}>
              <div
                className="h-full transition-all duration-300"
                style={{ width: `${progressPercent}%`, background: 'var(--brand-orange)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Error Alert if replay encountered a malformed line */}
      {status?.status === 'ERROR' && status.last_error && (
        <Alert variant="danger" className="mb-6">
          <AlertTriangle size={18} className="shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-xs">Replay Execution Failed</div>
            <div className="mt-1 text-xs font-mono">
              Dataset: {String(status.last_error.dataset)} · Line: {String(status.last_error.line_number)}
            </div>
            <div className="mt-0.5 text-xs">
              Error: {String(status.last_error.error_message)}
            </div>
          </div>
        </Alert>
      )}

      {/* Replay Completed Banner */}
      {status?.status === 'COMPLETED' && (
        <Alert variant="default" className="mb-6" style={{ background: 'var(--success-soft)', borderColor: 'var(--success)', color: 'var(--success)' }}>
          <CheckCircle2 size={18} className="shrink-0 mt-0.5" />
          <div className="flex items-center justify-between w-full">
            <div>
              <div className="font-semibold text-xs">Replay Finished Successfully</div>
              <div className="text-xs mt-0.5 font-mono">
                Processed all {status.total_count} alerts in {formatDuration(status.wall_clock_elapsed_seconds)} ({status.events_per_second.toFixed(1)} ev/s).
              </div>
            </div>
            <button
              onClick={() => navigate(withRunId('/meta-alerts'))}
              className="flex items-center gap-1 px-3 py-1 text-xs font-medium text-white rounded-[4px] cursor-pointer"
              style={{ background: 'var(--success)' }}
            >
              Investigate MetaAlerts <ArrowRight size={12} />
            </button>
          </div>
        </Alert>
      )}

      {/* Telemetry Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Playback Status" value={status?.status || 'IDLE'} />
        <MetricCard label="Processed Events" value={status ? formatNumber(status.processed_count) : 0} />
        <MetricCard label="Total Dataset Events" value={status ? formatNumber(status.total_count) : 0} />
        <MetricCard label="Throughput" value={status ? `${formatNumber(status.events_per_second)} ev/s` : '0 ev/s'} />
      </div>

      {/* Replay Details Card */}
      {status && status.run_id && (
        <div
          className="p-5 rounded-[7px] border text-xs font-mono space-y-2"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
        >
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Active Run Workspace ID:</span>
            <span className="font-semibold">{status.run_id}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Dataset:</span>
            <span>{status.dataset}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Current Event Timestamp:</span>
            <span>{status.current_event_time || '—'}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Wall-Clock Elapsed Time:</span>
            <span>{formatDuration(status.wall_clock_elapsed_seconds)}</span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--text-tertiary)' }}>Model Version:</span>
            <span>{status.model_version}</span>
          </div>
        </div>
      )}

      {/* Reset Confirmation Dialog */}
      <Dialog
        open={showResetConfirm}
        onClose={() => setShowResetConfirm(false)}
        title="Start New Replay Run?"
      >
        <div className="space-y-4 text-xs">
          <p style={{ color: 'var(--text-secondary)' }}>
            Resetting will prepare a clean, isolated workspace for your next replay run. All data and SQLite evidence from the current run ({status?.run_id?.slice(0, 8)}) will remain preserved on disk for audit investigation.
          </p>
          <div className="flex justify-end gap-2 pt-2 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
            <button
              onClick={() => setShowResetConfirm(false)}
              className="px-3 py-1.5 border rounded-[5px] font-medium cursor-pointer"
              style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
            >
              Cancel
            </button>
            <button
              onClick={handleConfirmReset}
              className="px-3 py-1.5 text-white rounded-[5px] font-medium cursor-pointer"
              style={{ background: 'var(--brand-orange)' }}
            >
              Confirm & Prepare New Run
            </button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}
