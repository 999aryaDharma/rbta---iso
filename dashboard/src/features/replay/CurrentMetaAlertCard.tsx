import type { PipelineLatestMetaAlert } from '@/api/schemas';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatNumber } from '@/lib/formatters';
import { Warning } from '@phosphor-icons/react';

interface CurrentMetaAlertCardProps {
  latestMeta: PipelineLatestMetaAlert | null | undefined;
  rawProcessed: number;
  metaFinalized: number;
  decisionCounts?: Record<string, number>;
}

export function CurrentMetaAlertCard({
  latestMeta,
  rawProcessed,
  metaFinalized,
  decisionCounts = {},
}: CurrentMetaAlertCardProps) {
  const reductionRate = rawProcessed > 0 && metaFinalized > 0
    ? Math.max(0, ((rawProcessed - metaFinalized) / rawProcessed) * 100)
    : 0;

  const score = latestMeta?.anomaly_score ?? 0;
  const threshold = latestMeta?.threshold_used ?? 0;
  const margin = latestMeta?.margin ?? (score - threshold);
  const isEscalate = latestMeta?.action === 'ESCALATE';

  return (
    <div className="rounded-lg border border-kumo-hairline bg-kumo-base p-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-kumo-hairline">
        <div className="flex items-center gap-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-kumo-subtle">
            Current Scored MetaAlert
          </div>
          {latestMeta ? (
            <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-kumo-recessed text-kumo-default border border-kumo-hairline">
              #{latestMeta.meta_id}
            </span>
          ) : (
            <span className="text-xs text-kumo-subtle italic">Awaiting first finalized bucket...</span>
          )}
        </div>

        {/* Global Reduction & Decision KPI counters */}
        <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-kumo-subtle">
            <span>Reduction:</span>
            <span className="font-semibold text-kumo-brand">{reductionRate.toFixed(1)}%</span>
            <span className="text-[11px] text-kumo-subtle">({formatNumber(rawProcessed)} raw → {formatNumber(metaFinalized)} meta)</span>
          </div>

          <div className="flex items-center gap-3 pl-3 border-l border-kumo-hairline">
            <span className="text-red-500 font-semibold flex items-center gap-1">
              <Warning size={13} /> ESCALATE: {decisionCounts.ESCALATE || 0}
            </span>
            <span className="text-kumo-subtle flex items-center gap-1">
              SUPPRESS: {decisionCounts.SUPPRESS || 0}
            </span>
            {decisionCounts.DAILY_DIGEST ? (
              <span className="text-amber-500 flex items-center gap-1">
                DIGEST: {decisionCounts.DAILY_DIGEST}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      {latestMeta ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Target Metadata */}
          <div className="space-y-1">
            <div className="text-[11px] text-kumo-subtle font-medium">Agent & Rule Group</div>
            <div className="font-mono text-xs font-semibold text-kumo-default truncate">
              {latestMeta.agent_name} ({latestMeta.agent_id})
            </div>
            <div className="font-mono text-xs text-kumo-subtle">
              Group: <span className="text-kumo-default font-medium">{latestMeta.rule_group_primary}</span>
            </div>
          </div>

          {/* Member alerts & Severity */}
          <div className="space-y-1">
            <div className="text-[11px] text-kumo-subtle font-medium">Aggregation Summary</div>
            <div className="text-xs text-kumo-default">
              <span className="font-mono font-semibold">{latestMeta.alert_count}</span> raw alerts clustered
            </div>
            <div className="text-xs text-kumo-subtle">
              Max Severity: <span className="font-mono font-semibold text-kumo-default">{latestMeta.max_severity}</span> / 15
            </div>
          </div>

          {/* Anomaly Score vs Threshold */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-kumo-subtle font-medium">Anomaly Score vs Threshold</span>
              <span className={`font-mono font-semibold ${margin >= 0 ? 'text-red-500' : 'text-kumo-subtle'}`}>
                {margin >= 0 ? `+${margin.toFixed(6)}` : margin.toFixed(6)}
              </span>
            </div>
            <div className="h-2 w-full bg-kumo-recessed rounded-full overflow-hidden flex relative">
              <div
                className={`h-full transition-all duration-300 ${isEscalate ? 'bg-red-500' : 'bg-kumo-brand'}`}
                style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-kumo-contrast z-10"
                style={{ left: `${Math.min(100, Math.max(0, threshold * 100))}%` }}
                title={`Threshold: ${threshold.toFixed(6)}`}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-kumo-subtle">
              <span>Score: {score.toFixed(6)}</span>
              <span>Tukey: {threshold.toFixed(6)}</span>
            </div>
          </div>

          {/* Decision & Action */}
          <div className="space-y-1 flex flex-col justify-center">
            <div className="text-[11px] text-kumo-subtle font-medium">Decision & Action</div>
            <div className="flex items-center gap-2">
              {latestMeta.decision && (
                <DecisionBadge decision={latestMeta.decision} action={latestMeta.action || 'SUPPRESS'} />
              )}
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono font-semibold border ${
                  isEscalate
                    ? 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-900/40'
                    : 'bg-kumo-recessed text-kumo-subtle border-kumo-hairline'
                }`}
              >
                {latestMeta.action || 'SUPPRESS'}
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="py-2 text-center text-xs text-kumo-subtle font-mono">
          Pipeline running — waiting for the next temporal aggregation window to close and finalize.
        </div>
      )}
    </div>
  );
}
