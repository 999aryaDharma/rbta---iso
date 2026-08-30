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
import { ReplayPipelineVisualizer, type PipelineStageId } from './ReplayPipelineVisualizer';
import { CurrentMetaAlertCard } from './CurrentMetaAlertCard';
import { PipelineStageDetail } from './PipelineStageDetail';
import { ProcessingTrace } from './ProcessingTrace';
import { DeferredTelegramOutbox } from './DeferredTelegramOutbox';

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
  const [activeStage, setActiveStage] = useState<PipelineStageId>('RBTA');

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
      await queryClient.invalidateQueries({ queryKey: ['telegram-payloads'] });
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

  const telemetry = status?.telemetry || undefined;
  const rawProcessed = telemetry?.raw.processed ?? status?.processed_count ?? 0;
  const metaFinalized = telemetry?.rbta.finalized_meta_alerts ?? 0;
  const latestMeta = telemetry?.latest_meta_alert;
  const decisionCounts = telemetry?.decision_counts;

  return (
    <>
      <PageHeader
        breadcrumbs={['Security Analytics', 'Replay']}
        title="Demonstration Replay Controller"
        description="Deterministic historical workload replay streaming with calibrated pacing, session isolation, and strict evidence logging"
      />

      <div className="px-6 py-8 lg:px-10 space-y-8">
        {/* Control Panel Card */}
        <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-5">
            <div className="flex flex-wrap items-center gap-5">
              <div>
                <label className="block text-[11px] font-semibold mb-1.5 text-kumo-subtle uppercase tracking-wider">
                  Replay Dataset (.jsonl)
                </label>
                <div className="w-[280px]">
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={!isIdle || isLoading || !datasetsData?.items?.length}
                    className="w-full px-3.5 py-2 border border-kumo-hairline rounded-lg text-xs font-mono bg-kumo-recessed/40 text-kumo-strong focus:border-kumo-strong outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
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
                <label className="block text-[11px] font-semibold mb-1.5 text-kumo-subtle uppercase tracking-wider">
                  Playback Speed
                </label>
                <div className="w-[220px]">
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(e.target.value as any)}
                    disabled={!isIdle || isLoading}
                    className="w-full px-3.5 py-2 border border-kumo-hairline rounded-lg text-xs font-mono bg-kumo-recessed/40 text-kumo-strong focus:border-kumo-strong outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
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
            <div className="flex items-center gap-3 mt-2 sm:mt-0">
              {isIdle && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleAction(() => startReplay(selectedDataset, speed))}
                  disabled={isLoading || !selectedDataset}
                >
                  <Play size={14} weight="fill" className="mr-1" /> Start Replay
                </Button>
              )}

              {isRunning && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAction(pauseReplay)}
                  disabled={isLoading}
                >
                  <Pause size={14} weight="fill" className="mr-1" /> Pause
                </Button>
              )}

              {isPaused && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleAction(resumeReplay)}
                  disabled={isLoading}
                >
                  <FastForward size={14} weight="fill" className="mr-1" /> Resume
                </Button>
              )}

              {(isRunning || isPaused) && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleAction(stopReplay)}
                  disabled={isLoading}
                >
                  <Stop size={14} weight="fill" className="mr-1" /> Stop
                </Button>
              )}

              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowResetConfirm(true)}
                disabled={isLoading || status?.status === 'IDLE'}
              >
                <ArrowClockwise size={14} className="mr-1" /> Reset New Run
              </Button>
            </div>
          </div>

          {/* Progress Bar */}
          {status && status.total_count > 0 && (
            <div className="pt-4 border-t border-kumo-hairline">
              <div className="flex justify-between text-xs mb-2 font-mono">
                <span className="text-kumo-subtle">Replay Progress</span>
                <span className="font-semibold text-kumo-strong">{progressPercent.toFixed(1)}% ({formatNumber(status.processed_count)} / {formatNumber(status.total_count)})</span>
              </div>
              <div className="w-full h-2.5 rounded-full overflow-hidden bg-kumo-recessed">
                <div
                  className="h-full bg-kumo-strong transition-all duration-300 rounded-full"
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
            description={`Processed all ${formatNumber(status.total_count)} alerts in ${formatDuration(status.wall_clock_elapsed_seconds)} (${status.events_per_second.toFixed(1)} ev/s).`}
          >
            <Banner.Action onClick={() => navigate(withRunId('/meta-alerts'))}>
              Investigate MetaAlerts <ArrowRight size={14} className="ml-1" />
            </Banner.Action>
          </Banner>
        )}

        {/* Processing Pipeline Visualizer */}
        <ReplayPipelineVisualizer
          status={status}
          telemetry={telemetry}
          activeStage={activeStage}
          onSelectStage={setActiveStage}
        />

        {/* Current Scored MetaAlert Card */}
        <CurrentMetaAlertCard
          latestMeta={latestMeta}
          rawProcessed={rawProcessed}
          metaFinalized={metaFinalized}
          decisionCounts={decisionCounts}
        />

        {/* Selected Pipeline Stage Deep Inspector */}
        <PipelineStageDetail
          activeStage={activeStage}
          telemetry={telemetry}
          status={status}
        />

        {/* Live Processing Trace Ring Buffer */}
        <ProcessingTrace trace={telemetry?.trace} />

        {/* Deferred Telegram Payload Outbox */}
        <DeferredTelegramOutbox />

        {/* Telemetry KPI Metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          <MetricCard label="Playback Status" value={status?.status || 'IDLE'} sub="Replay engine lifecycle" />
          <MetricCard label="Processed Events" value={status ? formatNumber(status.processed_count) : '0'} sub="Events parsed from dataset" />
          <MetricCard label="Total Dataset Events" value={status ? formatNumber(status.total_count) : '0'} sub="Total lines in dataset" />
          <MetricCard label="Throughput" value={status ? `${formatNumber(status.events_per_second)} ev/s` : '0 ev/s'} sub="Streaming velocity" />
        </div>

        {/* Replay Details Card */}
        {status && status.run_id && (
          <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs text-xs font-mono space-y-3">
            <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
              <span className="text-kumo-subtle">Active Run Workspace ID:</span>
              <span className="font-semibold text-kumo-strong">{status.run_id}</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
              <span className="text-kumo-subtle">Dataset Source:</span>
              <span className="text-kumo-default">{status.dataset}</span>
            </div>
            {status.dataset_mode === 'all' && (
              <>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <span className="text-kumo-subtle">Current File:</span>
                  <span className="text-kumo-default">{status.current_dataset || '—'}</span>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
                  <span className="text-kumo-subtle">Dataset Index:</span>
                  <span className="text-kumo-default">
                    {status.current_dataset_index !== undefined && status.dataset_count !== undefined
                      ? `${(status.current_dataset_index || 0) + 1} / ${status.dataset_count}`
                      : '—'}
                  </span>
                </div>
              </>
            )}
            <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
              <span className="text-kumo-subtle">Current Event Timestamp:</span>
              <span className="text-kumo-default">{status.current_event_time || '—'}</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-kumo-hairline/40">
              <span className="text-kumo-subtle">Wall-Clock Elapsed Time:</span>
              <span className="text-kumo-default">{formatDuration(status.wall_clock_elapsed_seconds)}</span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-kumo-subtle">Isolation Forest Model:</span>
              <span className="text-kumo-default">{status.model_version}</span>
            </div>
          </div>
        )}
      </div>

      {/* Reset Confirmation Dialog */}
      <DialogRoot open={showResetConfirm} onOpenChange={(o) => { if (!o) setShowResetConfirm(false); }}>
        <Dialog className="max-w-md w-full p-6 bg-kumo-canvas border border-kumo-hairline shadow-2xl rounded-xl">
          <DialogTitle className="text-base font-bold text-kumo-strong">Start New Replay Run?</DialogTitle>
          <DialogDescription className="text-xs text-kumo-subtle mt-2 leading-relaxed">
            Resetting will prepare a clean, isolated workspace for your next replay run. All data and SQLite evidence from the current run ({status?.run_id?.slice(0, 8)}) will remain preserved on disk for audit investigation.
          </DialogDescription>
          <div className="flex justify-end gap-3 pt-4 mt-4 border-t border-kumo-hairline">
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
