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
import { DialogRoot, Dialog, DialogTitle, DialogDescription, DialogClose } from '@cloudflare/kumo/components/dialog';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Button } from '@cloudflare/kumo/components/button';
import { formatNumber, formatDuration } from '@/lib/formatters';
import { Play, Pause, Stop, ArrowClockwise, FastForward, ArrowRight } from '@phosphor-icons/react';

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

  const isAllDatasets = status?.dataset === '__ALL__' || status?.dataset === 'ALL' || status?.dataset_mode === 'all';

  return (
    <>
      <PageHeader
        title="Demonstration Replay Controller"
        description="Deterministic historical workload replay streaming with calibrated pacing, session isolation, and strict evidence logging"
      />

      <div className="px-6 py-4 space-y-4">
        {/* Control Panel Card */}
        <div className="p-4 rounded-lg border border-kumo-hairline bg-kumo-base">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div>
                <label className="block text-[11px] font-semibold mb-1 text-kumo-subtle">
                  Replay Dataset (.jsonl)
                </label>
                <div className="w-[260px]">
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={!isIdle || isLoading || !datasetsData?.items?.length}
                    className="w-full px-3 py-1.5 border border-kumo-hairline rounded-md text-xs font-mono bg-kumo-base text-kumo-default focus:ring-1 focus:ring-kumo-brand outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <option value="__ALL__">All datasets (Sequential)</option>
                    {datasetsData?.items?.map((ds) => (
                      <option key={ds.name} value={ds.name}>
                        {ds.name} ({Math.round(ds.size_bytes / 1024)} KB)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[11px] font-semibold mb-1 text-kumo-subtle">
                  Playback Speed
                </label>
                <div className="w-[200px]">
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(e.target.value as any)}
                    disabled={!isIdle || isLoading}
                    className="w-full px-3 py-1.5 border border-kumo-hairline rounded-md text-xs font-mono bg-kumo-base text-kumo-default focus:ring-1 focus:ring-kumo-brand outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <option value="1">1x (Realtime Clock)</option>
                    <option value="10">10x Throttled</option>
                    <option value="100">100x Fast Pacing</option>
                    <option value="MAX">MAX (Unthrottled Throughput)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2 mt-4 md:mt-0">
              {isIdle && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleAction(() => startReplay(selectedDataset, speed))}
                  disabled={isLoading || !selectedDataset}
                >
                  <Play size={14} weight="fill" /> Start Replay
                </Button>
              )}

              {isRunning && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAction(pauseReplay)}
                  disabled={isLoading}
                >
                  <Pause size={14} weight="fill" /> Pause
                </Button>
              )}

              {isPaused && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleAction(resumeReplay)}
                  disabled={isLoading}
                >
                  <FastForward size={14} weight="fill" /> Resume
                </Button>
              )}

              {(isRunning || isPaused) && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleAction(stopReplay)}
                  disabled={isLoading}
                >
                  <Stop size={14} weight="fill" /> Stop
                </Button>
              )}

              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowResetConfirm(true)}
                disabled={isLoading || status?.status === 'IDLE'}
              >
                <ArrowClockwise size={14} /> Reset New Run
              </Button>
            </div>
          </div>

          {/* Progress Bar */}
          {status && status.total_count > 0 && (
            <div className="mt-4 pt-4 border-t border-kumo-hairline">
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-kumo-subtle">Replay Progress</span>
                <span className="font-semibold text-kumo-default">{progressPercent.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2 rounded-full overflow-hidden bg-kumo-recessed">
                <div
                  className="h-full bg-kumo-brand transition-all duration-300 rounded-full"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Error Alert if replay encountered a malformed line */}
        {status?.status === 'ERROR' && status.last_error && (
          <Banner
            variant="error"
            size="sm"
            title="Replay Execution Failed"
            description={`Dataset: ${String(status.last_error.dataset)} · Line: ${String(status.last_error.line_number)} · Error: ${String(status.last_error.error_message)}`}
          />
        )}

        {/* Replay Completed Banner */}
        {status?.status === 'COMPLETED' && (
          <Banner
            variant="default"
            size="base"
            title="Replay Finished Successfully"
            description={`Processed all ${status.total_count} alerts in ${formatDuration(status.wall_clock_elapsed_seconds)} (${status.events_per_second.toFixed(1)} ev/s).`}
          >
            <Banner.Action onClick={() => navigate(withRunId('/meta-alerts'))}>
              Investigate MetaAlerts <ArrowRight size={14} />
            </Banner.Action>
          </Banner>
        )}

        {/* Telemetry Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="Playback Status" value={status?.status || 'IDLE'} />
          <MetricCard label="Processed Events" value={status ? formatNumber(status.processed_count) : '0'} />
          <MetricCard label="Total Dataset Events" value={status ? formatNumber(status.total_count) : '0'} />
          <MetricCard label="Throughput" value={status ? `${formatNumber(status.events_per_second)} ev/s` : '0 ev/s'} />
        </div>

        {/* Replay Details Card */}
        {status && status.run_id && (
          <div className="p-4 rounded-lg border border-kumo-hairline bg-kumo-base text-xs font-mono space-y-2">
            <div className="flex justify-between">
              <span className="text-kumo-subtle">Active Run Workspace ID:</span>
              <span className="font-semibold text-kumo-default">{status.run_id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-kumo-subtle">Dataset:</span>
              <span className="text-kumo-default">{status.dataset}</span>
            </div>
            {isAllDatasets && (
              <>
                <div className="flex justify-between">
                  <span className="text-kumo-subtle">Current File Name:</span>
                  <span className="text-kumo-default">{(status as any).current_file_name || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-kumo-subtle">File Index / Total:</span>
                  <span className="text-kumo-default">
                    {(status as any).file_index !== undefined ? `${(status as any).file_index} / ${(status as any).file_total}` : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-kumo-subtle">Global Progress:</span>
                  <span className="text-kumo-default">
                    {(status as any).global_progress !== undefined ? `${((status as any).global_progress * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
              </>
            )}
            <div className="flex justify-between">
              <span className="text-kumo-subtle">Current Event Timestamp:</span>
              <span className="text-kumo-default">{status.current_event_time || '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-kumo-subtle">Wall-Clock Elapsed Time:</span>
              <span className="text-kumo-default">{formatDuration(status.wall_clock_elapsed_seconds)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-kumo-subtle">Model Version:</span>
              <span className="text-kumo-default">{status.model_version}</span>
            </div>
          </div>
        )}
      </div>

      {/* Reset Confirmation Dialog */}
      <DialogRoot open={showResetConfirm} onOpenChange={(o) => { if (!o) setShowResetConfirm(false); }}>
        <Dialog>
          <DialogTitle>Start New Replay Run?</DialogTitle>
          <DialogDescription>
            Resetting will prepare a clean, isolated workspace for your next replay run. All data and SQLite evidence from the current run ({status?.run_id?.slice(0, 8)}) will remain preserved on disk for audit investigation.
          </DialogDescription>
          <div className="flex justify-end gap-2 pt-4 mt-4 border-t border-kumo-hairline">
            <DialogClose>
              <Button variant="ghost" size="sm" onClick={() => setShowResetConfirm(false)}>
                Cancel
              </Button>
            </DialogClose>
            <Button
              variant="primary"
              size="sm"
              onClick={handleConfirmReset}
            >
              Confirm & Prepare New Run
            </Button>
          </div>
        </Dialog>
      </DialogRoot>
    </>
  );
}
