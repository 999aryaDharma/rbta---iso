import type { PipelineTraceItem } from '@/api/schemas';
import { TerminalWindow } from '@phosphor-icons/react';

interface ProcessingTraceProps {
  trace: PipelineTraceItem[] | undefined;
}

export function ProcessingTrace({ trace }: ProcessingTraceProps) {
  const items = trace || [];

  const getStageBadgeColor = (stage: string) => {
    switch (stage) {
      case 'RAW':
        return 'bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300 border-neutral-200 dark:border-neutral-700';
      case 'CANONICAL':
        return 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400 border-blue-200 dark:border-blue-900/40';
      case 'RBTA':
        return 'bg-purple-50 text-purple-700 dark:bg-purple-950/40 dark:text-purple-400 border-purple-200 dark:border-purple-900/40';
      case 'FINALIZE':
        return 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-400 border-indigo-200 dark:border-indigo-900/40';
      case 'FEATURES':
        return 'bg-cyan-50 text-cyan-700 dark:bg-cyan-950/40 dark:text-cyan-400 border-cyan-200 dark:border-cyan-900/40';
      case 'SCORE':
        return 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400 border-amber-200 dark:border-amber-900/40';
      case 'DECISION':
        return 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/40';
      case 'OUTPUT':
        return 'bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400 border-red-200 dark:border-red-900/40';
      default:
        return 'bg-kumo-recessed text-kumo-subtle border-kumo-hairline';
    }
  };

  return (
    <div className="rounded-lg border border-kumo-hairline bg-kumo-base p-4 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-kumo-hairline">
        <div className="flex items-center gap-2">
          <TerminalWindow size={14} className="text-kumo-brand" />
          <span className="text-xs font-semibold uppercase tracking-wider text-kumo-subtle">
            Live Runtime Processing Trace
          </span>
        </div>
        <span className="text-[11px] font-mono text-kumo-subtle">
          Showing latest {items.length} pipeline events (ring buffer)
        </span>
      </div>

      <div className="h-44 overflow-y-auto rounded bg-kumo-canvas border border-kumo-hairline p-2 space-y-1 font-mono text-[11px]">
        {items.length > 0 ? (
          items.map((item, idx) => (
            <div key={`${item.timestamp}-${idx}`} className="flex items-start gap-2 py-0.5 leading-relaxed">
              <span className="text-kumo-subtle shrink-0">{item.timestamp}</span>
              <span
                className={`px-1.5 py-0.2 rounded text-[10px] font-semibold uppercase shrink-0 border ${getStageBadgeColor(
                  item.stage
                )}`}
              >
                {item.stage}
              </span>
              <span className="text-kumo-default font-medium truncate">{item.message}</span>
              {item.detail && (
                <span className="text-kumo-subtle truncate max-w-md">({item.detail})</span>
              )}
            </div>
          ))
        ) : (
          <div className="h-full flex items-center justify-center text-xs text-kumo-subtle italic">
            Trace buffer idle. Start or resume replay to observe live stream operations.
          </div>
        )}
      </div>
    </div>
  );
}
