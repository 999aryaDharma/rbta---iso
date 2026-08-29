import React from 'react';
import type { ReplayStatus, PipelineTelemetry } from '@/api/schemas';
import { formatNumber } from '@/lib/formatters';
import {
  Database,
  ArrowsSplit,
  HardDrives,
  ClockAfternoon,
  Stack,
  ChartBarHorizontal,
  TreeStructure,
  Scales,
  PaperPlaneRight,
  CaretRight,
  CheckCircle,
  WarningCircle,
  PauseCircle,
  PlayCircle,
} from '@phosphor-icons/react';

export type PipelineStageId =
  | 'DATASET'
  | 'CANONICAL'
  | 'EVIDENCE'
  | 'RBTA'
  | 'META_ALERT'
  | 'FEATURES'
  | 'ISOLATION_FOREST'
  | 'DECISION'
  | 'OUTPUT_SINK';

interface ReplayPipelineVisualizerProps {
  status: ReplayStatus | undefined;
  telemetry: PipelineTelemetry | undefined;
  activeStage: PipelineStageId;
  onSelectStage: (stage: PipelineStageId) => void;
}

export function ReplayPipelineVisualizer({
  status,
  telemetry,
  activeStage,
  onSelectStage,
}: ReplayPipelineVisualizerProps) {
  const isRunning = status?.status === 'RUNNING';
  const isPaused = status?.status === 'PAUSED';
  const isCompleted = status?.status === 'COMPLETED';
  const isError = status?.status === 'ERROR';

  const rawProcessed = telemetry?.raw.processed ?? status?.processed_count ?? 0;
  const rawTotal = status?.total_count ?? 0;
  const evidenceCount = telemetry?.raw.evidence_count ?? rawProcessed;
  const activeBuckets = telemetry?.rbta.active_buckets ?? 0;
  const finalizedCount = telemetry?.rbta.finalized_meta_alerts ?? 0;
  const activeAgents = telemetry?.rbta.active_agents ?? 0;
  const latestMeta = telemetry?.latest_meta_alert;
  const decisionCounts = telemetry?.decision_counts ?? {};
  const telegramCount = telemetry?.output.telegram_deferred_count ?? 0;

  const stages: {
    id: PipelineStageId;
    title: string;
    sublabel: string;
    icon: React.ComponentType<any>;
    badge: string;
    details: [string, string | number][];
  }[] = [
    {
      id: 'DATASET',
      title: '1. Dataset',
      sublabel: status?.dataset_mode === 'all' ? 'All Datasets (.jsonl)' : (status?.dataset || 'Single .jsonl'),
      icon: Database,
      badge: status?.dataset_mode === 'all' ? `${(status?.current_dataset_index ?? 0) + 1}/${status?.dataset_count ?? 1} files` : '1 file',
      details: [
        ['Processed', `${formatNumber(rawProcessed)} / ${formatNumber(rawTotal)}`],
        ['Throughput', `${status?.events_per_second?.toFixed(1) || '0.0'} ev/s`],
      ],
    },
    {
      id: 'CANONICAL',
      title: '2. Canonicalize',
      sublabel: 'CanonicalRawAlert Schema',
      icon: ArrowsSplit,
      badge: 'Validated',
      details: [
        ['Normalized', formatNumber(rawProcessed)],
        ['Last Group', (telemetry?.raw.last_alert?.rule_group as string) || 'none'],
      ],
    },
    {
      id: 'EVIDENCE',
      title: '3. Raw Evidence',
      sublabel: 'SQLite Durability',
      icon: HardDrives,
      badge: 'WAL Mode',
      details: [
        ['Store Count', formatNumber(evidenceCount)],
        ['Decoder Sync', 'MappingProxy'],
      ],
    },
    {
      id: 'RBTA',
      title: '4. RBTA Window',
      sublabel: 'Adaptive ETW Aggregation',
      icon: ClockAfternoon,
      badge: `${activeBuckets} active`,
      details: [
        ['Active Buckets', activeBuckets],
        ['Active Agents', activeAgents],
      ],
    },
    {
      id: 'META_ALERT',
      title: '5. MetaAlert',
      sublabel: 'Temporal Cluster',
      icon: Stack,
      badge: `#${latestMeta?.meta_id ?? finalizedCount}`,
      details: [
        ['Finalized', formatNumber(finalizedCount)],
        ['Clustered Alert', `${latestMeta?.alert_count ?? 0} alerts`],
      ],
    },
    {
      id: 'FEATURES',
      title: '6. 7 Features',
      sublabel: '7-Dimensional Vector',
      icon: ChartBarHorizontal,
      badge: '7D Vector',
      details: [
        ['Max Severity', `${latestMeta?.max_severity ?? 0}/15`],
        ['MITRE Tactics', `${latestMeta?.mitre_tactics?.length ?? 0} unique`],
      ],
    },
    {
      id: 'ISOLATION_FOREST',
      title: '7. IsoForest',
      sublabel: '200 Tree Ensemble',
      icon: TreeStructure,
      badge: latestMeta?.model_version ?? 'rbta-if-v1',
      details: [
        ['Raw Score', latestMeta?.raw_model_score?.toFixed(6) ?? '—'],
        ['Calibrated', latestMeta?.anomaly_score?.toFixed(6) ?? '—'],
      ],
    },
    {
      id: 'DECISION',
      title: '8. Decision',
      sublabel: 'Tukey IQR Threshold',
      icon: Scales,
      badge: latestMeta?.decision ?? 'NOISE',
      details: [
        ['Score vs Tukey', `${latestMeta?.anomaly_score?.toFixed(4) ?? '0'} / ${latestMeta?.threshold_used?.toFixed(4) ?? '0'}`],
        ['ESCALATE', `${decisionCounts.ESCALATE || 0} metas`],
      ],
    },
    {
      id: 'OUTPUT_SINK',
      title: '9. Output Sink',
      sublabel: 'Deferred Telegram File',
      icon: PaperPlaneRight,
      badge: `${telegramCount} payloads`,
      details: [
        ['ESCALATE Outbox', telegramCount],
        ['Sink Format', 'JSONL .txt'],
      ],
    },
  ];

  return (
    <div className="rounded-lg border border-kumo-hairline bg-kumo-base p-4 space-y-3">
      {/* Header & Status Indicator */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-kumo-hairline">
        <div className="flex items-center gap-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-kumo-subtle">
            Operational Processing Pipeline
          </div>
          <span className="text-[11px] text-kumo-subtle font-mono">
            (Click any stage node to inspect telemetry detail)
          </span>
        </div>

        {/* Runtime State Badge */}
        <div className="flex items-center gap-2 font-mono text-xs">
          {isRunning && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400 border border-blue-200 dark:border-blue-900/40 animate-pulse">
              <PlayCircle size={13} weight="fill" />
              <span>RUNNING</span>
            </span>
          )}
          {isPaused && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400 border border-amber-200 dark:border-amber-900/40">
              <PauseCircle size={13} weight="fill" />
              <span>PAUSED</span>
            </span>
          )}
          {isCompleted && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-green-50 text-green-700 dark:bg-green-950/40 dark:text-green-400 border border-green-200 dark:border-green-900/40">
              <CheckCircle size={13} weight="fill" />
              <span>COMPLETED</span>
            </span>
          )}
          {isError && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400 border border-red-200 dark:border-red-900/40">
              <WarningCircle size={13} weight="fill" />
              <span>ERROR</span>
            </span>
          )}
          {!isRunning && !isPaused && !isCompleted && !isError && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-kumo-recessed text-kumo-subtle border border-kumo-hairline">
              <span>IDLE</span>
            </span>
          )}
        </div>
      </div>

      {/* Pipeline Grid / Flow */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-9 gap-2">
        {stages.map((st, idx) => {
          const isSelected = activeStage === st.id;
          const Icon = st.icon;

          return (
            <button
              key={st.id}
              type="button"
              onClick={() => onSelectStage(st.id)}
              className={`text-left p-2.5 rounded-md border transition-all duration-150 flex flex-col justify-between relative group cursor-pointer ${
                isSelected
                  ? 'bg-kumo-canvas border-kumo-brand ring-1 ring-kumo-brand'
                  : 'bg-kumo-base hover:bg-kumo-canvas border-kumo-hairline hover:border-kumo-subtle'
              }`}
              aria-pressed={isSelected}
            >
              {/* Connector Arrow (hidden on mobile, shown between items on large screens) */}
              {idx < stages.length - 1 && (
                <div className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 z-10 text-kumo-subtle pointer-events-none">
                  <CaretRight size={10} weight="bold" />
                </div>
              )}

              <div>
                <div className="flex items-center justify-between gap-1 mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <Icon size={14} className={isSelected ? 'text-kumo-brand' : 'text-kumo-subtle'} />
                    <span className="text-xs font-semibold text-kumo-default truncate">
                      {st.title}
                    </span>
                  </div>
                </div>

                <div className="text-[10px] text-kumo-subtle truncate mb-2">
                  {st.sublabel}
                </div>
              </div>

              {/* Metric Details */}
              <div className="space-y-1 pt-1.5 border-t border-kumo-hairline font-mono text-[10px]">
                {st.details.map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center text-kumo-subtle">
                    <span>{k}:</span>
                    <span className="font-semibold text-kumo-default truncate max-w-[70px]">{v}</span>
                  </div>
                ))}
              </div>

              {/* Active Stage Indicator dot */}
              {isRunning && (
                <div
                  className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-kumo-brand animate-ping"
                  aria-hidden="true"
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
