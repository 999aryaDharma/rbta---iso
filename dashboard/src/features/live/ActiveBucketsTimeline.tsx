import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchBuckets } from '@/api/dashboard';
import { buildTimelineBars, type TimelineBar } from './bucketTimeline';

const TIER_BAR: Record<TimelineBar['tier'], string> = {
  critical: 'bg-red-500',
  high: 'bg-amber-500',
  medium: 'bg-sky-500',
  low: 'bg-kumo-subtle',
};

interface ActiveBucketsTimelineProps {
  bars: TimelineBar[];
  windowMinutes: number;
  onSelect: (metaId: number) => void;
}

export function ActiveBucketsTimeline({ bars, windowMinutes, onSelect }: ActiveBucketsTimelineProps) {
  if (bars.length === 0) {
    return (
      <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs">
        <p role="status" className="text-xs text-kumo-subtle">
          Belum ada bucket aktif dalam {windowMinutes} menit terakhir. Worker berhenti atau belum ada alert masuk.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-4">
      <div className="pb-3 border-b border-kumo-hairline">
        <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong">
          Bucket aktif RBTA: {windowMinutes} menit terakhir (event-time)
        </h3>
        <p className="text-xs text-kumo-subtle mt-1">
          Bar = unit agregasi yang tumbuh saat alert masuk. Bar bukan bukti serangan.
        </p>
      </div>
      <div className="space-y-3">
        {bars.map((b) => {
          const label = `${b.agent_id} · ${b.rule_group_primary} · ${b.alert_count} alert`;
          const body = (
            <>
              <div className="flex items-baseline justify-between gap-2 text-xs mb-1">
                <span className="font-mono font-semibold text-kumo-strong truncate">
                  {b.agent_id} · {b.rule_group_primary}
                </span>
                <span className="font-mono text-kumo-subtle shrink-0">
                  {b.alert_count} alert · sev {b.max_severity}
                </span>
              </div>
              <div className="relative h-3 rounded bg-kumo-recessed/60 overflow-hidden">
                <div
                  className={`absolute top-0 bottom-0 rounded ${TIER_BAR[b.tier]}`}
                  style={{ left: `${b.x0 * 100}%`, width: `${Math.max(1.5, (b.x1 - b.x0) * 100)}%` }}
                />
              </div>
              <p className="text-[11px] text-kumo-subtle mt-1">
                {b.finalized ? `Final → MetaAlert #${b.meta_id}` : 'Mengagregasi… belum ada prediksi'}
              </p>
            </>
          );
          return b.finalized && b.meta_id != null ? (
            <button
              key={b.key}
              type="button"
              aria-label={`${label}, final ke MetaAlert ${b.meta_id}`}
              onClick={() => onSelect(b.meta_id as number)}
              className="block w-full text-left rounded-lg p-2 hover:bg-kumo-recessed/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-kumo-brand"
            >
              {body}
            </button>
          ) : (
            <div key={b.key} aria-label={label} className="p-2">
              {body}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ActiveBucketsTimelineCard({ windowMinutes = 15 }: { windowMinutes?: number }) {
  const navigate = useNavigate();
  const { data: buckets = [] } = usePollingQuery(['buckets', 'live'], () => fetchBuckets(), 5000);
  const now = Date.now();
  const bars = useMemo(
    () => buildTimelineBars(buckets, now, windowMinutes * 60_000),
    [buckets, now, windowMinutes],
  );
  return (
    <ActiveBucketsTimeline
      bars={bars}
      windowMinutes={windowMinutes}
      onSelect={(metaId) => navigate(`/meta-alerts/${metaId}`)}
    />
  );
}
