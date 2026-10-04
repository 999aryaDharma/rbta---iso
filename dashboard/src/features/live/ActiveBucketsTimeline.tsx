import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchBuckets } from '@/api/dashboard';
import { buildTimelineBars, type TimelineBar } from './bucketTimeline';
import { formatNumber, formatSeconds } from '@/lib/formatters';
import { Table } from '@cloudflare/kumo/components/table';
import { Badge } from '@cloudflare/kumo/components/badge';

interface ActiveBucketsTimelineProps {
  bars: TimelineBar[];
  windowMinutes: number;
  onSelect: (metaId: number) => void;
}

function windowSeconds(startIso: string, endIso: string): number | null {
  const s = new Date(startIso).getTime();
  const e = new Date(endIso).getTime();
  if (Number.isNaN(s) || Number.isNaN(e) || e < s) return null;
  return (e - s) / 1000;
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
    <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs overflow-hidden">
      <div className="px-6 py-4 border-b border-kumo-hairline">
        <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong">
          Bucket aktif RBTA: {windowMinutes} menit terakhir (event-time)
        </h3>
        <p className="text-xs text-kumo-subtle mt-1">
          Baris = unit agregasi yang masih tumbuh. Baris bukan bukti serangan; link hanya muncul setelah final.
        </p>
      </div>
      <Table>
        <Table.Header>
          <Table.Row className="bg-kumo-recessed/50 text-[11px] uppercase tracking-wider">
            <Table.Head>Bucket</Table.Head>
            <Table.Head className="text-right">Alert</Table.Head>
            <Table.Head className="text-right">Durasi window</Table.Head>
            <Table.Head className="text-right">Level max</Table.Head>
            <Table.Head className="text-center">Status</Table.Head>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {bars.map((b) => (
            <Table.Row key={b.key} className="hover:bg-kumo-recessed/40 transition-colors text-xs">
              <Table.Cell>
                <span className="font-mono font-semibold text-kumo-strong">
                  {b.agent_id} · {b.rule_group_primary}
                </span>
                {b.agent_name && (
                  <span className="block font-mono text-[11px] text-kumo-subtle">{b.agent_name}</span>
                )}
              </Table.Cell>
              <Table.Cell className="text-right font-mono font-bold text-kumo-strong">
                {formatNumber(b.alert_count)}
              </Table.Cell>
              <Table.Cell className="text-right font-mono text-kumo-subtle">
                {formatSeconds(windowSeconds(b.start_time, b.end_time))}
              </Table.Cell>
              <Table.Cell className="text-right font-mono text-kumo-default">
                {b.max_severity} / 15 ({b.tier})
              </Table.Cell>
              <Table.Cell className="text-center">
                {b.finalized && b.meta_id != null ? (
                  <button
                    type="button"
                    aria-label={`${b.agent_id} ${b.rule_group_primary}, final ke MetaAlert ${b.meta_id}`}
                    onClick={() => onSelect(b.meta_id as number)}
                    className="font-mono font-semibold text-kumo-brand underline underline-offset-2 hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-kumo-brand rounded"
                  >
                    MetaAlert #{b.meta_id}
                  </button>
                ) : (
                  <span className="inline-flex flex-col items-center gap-1">
                    <Badge variant="secondary">Mengagregasi</Badge>
                    {b.meta_id != null && (
                      <span className="font-mono text-[11px] text-kumo-subtle">calon #{b.meta_id}</span>
                    )}
                  </span>
                )}
              </Table.Cell>
            </Table.Row>
          ))}
        </Table.Body>
      </Table>
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
