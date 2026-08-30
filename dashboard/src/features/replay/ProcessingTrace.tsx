import type { PipelineTraceItem } from '@/api/schemas';
import { Badge } from '@cloudflare/kumo/components/badge';
import { TerminalWindow } from '@phosphor-icons/react';

interface ProcessingTraceProps {
  trace: PipelineTraceItem[] | undefined;
}

export function ProcessingTrace({ trace }: ProcessingTraceProps) {
  const items = trace || [];

  return (
    <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas p-6 shadow-xs space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-kumo-hairline">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md border border-kumo-hairline bg-kumo-recessed flex items-center justify-center text-kumo-strong">
            <TerminalWindow size={14} />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-kumo-strong">
            Live Runtime Processing Trace
          </span>
        </div>
        <Badge variant="secondary">
          Showing latest {items.length} pipeline events (ring buffer)
        </Badge>
      </div>

      <div className="h-52 overflow-y-auto rounded-lg bg-kumo-recessed/30 border border-kumo-hairline p-3 space-y-1.5 font-mono text-[11px]">
        {items.length > 0 ? (
          items.map((item, idx) => (
            <div key={`${item.timestamp}-${idx}`} className="flex items-start gap-2.5 py-0.5 leading-relaxed">
              <span className="text-kumo-subtle shrink-0 font-mono">{item.timestamp}</span>
              <Badge variant="secondary">
                {item.stage}
              </Badge>
              <span className="text-kumo-strong font-medium truncate">{item.message}</span>
              {item.detail && (
                <span className="text-kumo-subtle truncate max-w-lg">({item.detail})</span>
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
