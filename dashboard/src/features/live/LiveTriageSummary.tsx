import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary } from '@/api/dashboard';
import { fetchLiveStatus } from '@/api/live';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { LIVE_STALE_AFTER_MS } from './LiveStatusPanel';
import { ListChecks } from '@phosphor-icons/react';

export interface TriageSummaryDatum {
  rawAlerts: number | null;
  metaAlerts: number | null;
  arrPercent: number | null;
  escalateOpen: number | null;
  outboxPending: number | null;
  decisionCounts: Record<string, number>;
  decisionSampled: number;
  decisionTotal: number;
  decisionUnavailable: boolean;
  eventLagSec: number | null;
  stale: boolean;
  workerAlive: boolean;
}

interface LiveTriageSummaryPanelProps {
  isLoading: boolean;
  isError: boolean;
  data: TriageSummaryDatum | null;
  errorMessage?: string;
  onRetry: () => void;
}

const DECISION_ORDER = ['CRITICAL', 'SUSPICIOUS', 'NOISE_HIGH', 'NOISE'];

function formatLag(sec: number | null): string {
  if (sec === null || Number.isNaN(sec)) return '-';
  if (sec < 60) return `${sec.toFixed(1)} dtk`;
  return `${(sec / 60).toFixed(1)} mnt`;
}

function formatArr(v: number | null): string {
  if (v === null || Number.isNaN(v)) return '-';
  return `${v}%`;
}

function orderedDecisions(counts: Record<string, number>): Array<[string, number]> {
  const entries = Object.entries(counts);
  return entries.sort(([a], [b]) => {
    const ia = DECISION_ORDER.indexOf(a);
    const ib = DECISION_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

export function LiveTriageSummaryPanel({ isLoading, isError, data, errorMessage, onRetry }: LiveTriageSummaryPanelProps) {
  if (isLoading) {
    return (
      <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs">
        <p role="status" className="text-xs text-kumo-subtle">Memuat beban triase live…</p>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-3">
        <p role="alert" className="text-xs text-red-600">
          Gagal memuat beban triase live{errorMessage ? `: ${errorMessage}` : ''}
        </p>
        <button
          type="button"
          onClick={onRetry}
          className="px-4 py-2 rounded-lg text-sm font-semibold border border-kumo-hairline"
        >
          Coba lagi
        </button>
      </div>
    );
  }

  const decisions = orderedDecisions(data.decisionCounts);
  const decisionTotal = Math.max(1, decisions.reduce((acc, [, v]) => acc + v, 0));

  return (
    <div className="p-6 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-5">
      <div className="flex items-center gap-3 pb-3 border-b border-kumo-hairline">
        <div className="w-8 h-8 rounded-lg border border-kumo-hairline bg-kumo-recessed text-kumo-strong flex items-center justify-center">
          <ListChecks size={18} />
        </div>
        <div>
          <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong">
            Beban triase live
          </h3>
          <p className="text-[11px] text-kumo-subtle mt-0.5">
            ARR = reduksi unit triase, bukan akurasi. Skor anomali = prioritas, bukan vonis serangan.
          </p>
        </div>
        <span className={`ml-auto font-mono text-xs font-bold ${data.workerAlive ? 'text-emerald-500' : 'text-kumo-subtle'}`}>
          {data.workerAlive ? 'Worker hidup' : 'Worker berhenti'}
        </span>
      </div>

      {data.stale && data.workerAlive && (
        <p role="status" className="text-xs text-amber-600">
          Data basi, siklus terakhir lebih dari 2 menit lalu. Periksa koneksi Indexer/API dan log backend.
        </p>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Alert masuk</p>
          <p className="font-mono font-semibold text-kumo-strong text-base">
            {data.rawAlerts ?? '-'}
          </p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">MetaAlert final</p>
          <p className="font-mono font-semibold text-kumo-strong text-base">
            {data.metaAlerts ?? '-'}
          </p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Reduksi live (ARR)</p>
          <p className="font-mono font-semibold text-kumo-strong text-base">
            {formatArr(data.arrPercent)}
          </p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Kesegaran (event-lag)</p>
          <p className="font-mono text-kumo-default text-base">{formatLag(data.eventLagSec)}</p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">ESCALATE terbuka</p>
          <p className="font-mono font-semibold text-kumo-strong text-base">
            {data.escalateOpen ?? '-'}
          </p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Outbox menunggu</p>
          <p className="font-mono text-kumo-default text-base">
            {data.outboxPending ?? '-'}
          </p>
        </div>
      </div>

      <p className="text-[11px] text-kumo-subtle">
        Umur tertua tidak tersedia dari API yang ada. Outbox live hanya mengekspos hitungan
        tanpa stempel waktu per item. Endpoint Telegram payloads milik konteks replay, bukan live.
      </p>

      <div>
        <h4 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong mb-2">
          Distribusi decision (live)
        </h4>
        {data.decisionUnavailable ? (
          <p className="text-xs text-kumo-subtle">
            Distribusi decision tidak tersedia. Daftar MetaAlert live gagal dimuat.
          </p>
        ) : decisions.length === 0 ? (
          <p className="text-xs text-kumo-subtle">
            Belum ada MetaAlert live untuk distribusi decision.
          </p>
        ) : (
          <div className="space-y-2">
            {decisions.map(([name, value]) => (
              <div key={name}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="font-semibold text-kumo-default">{name}</span>
                  <span className="font-mono text-kumo-subtle">{value}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-kumo-recessed">
                  <div
                    className="h-full rounded-full bg-kumo-brand"
                    style={{ width: `${(value / decisionTotal) * 100}%` }}
                  />
                </div>
              </div>
            ))}
            {data.decisionSampled < data.decisionTotal && (
              <p className="text-[11px] text-kumo-subtle">
                Agregat {data.decisionSampled} dari {data.decisionTotal} MetaAlert live (100 terbaru).
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const META_SAMPLE_SIZE = 100;

export function LiveTriageSummaryCard() {
  const summaryQuery = usePollingQuery(['summary', 'live'], () => fetchSummary(), 5000);
  const statusQuery = usePollingQuery(['live-status'], fetchLiveStatus, 5000);
  const metasQuery = usePollingQuery(
    ['meta-alerts', 'live-decision'],
    () => fetchMetaAlerts({ page: 1, page_size: META_SAMPLE_SIZE }),
    5000,
  );

  const isLoading = summaryQuery.isLoading || statusQuery.isLoading;
  const isError = summaryQuery.isError || statusQuery.isError;
  const error = summaryQuery.error ?? statusQuery.error;

  const refetchAll = () => {
    void summaryQuery.refetch();
    void statusQuery.refetch();
    void metasQuery.refetch();
  };

  if (isLoading && !summaryQuery.data && !statusQuery.data) {
    return <LiveTriageSummaryPanel isLoading isError={false} data={null} onRetry={refetchAll} />;
  }

  if (isError || !summaryQuery.data || !statusQuery.data) {
    return (
      <LiveTriageSummaryPanel
        isLoading={false}
        isError
        data={null}
        errorMessage={error instanceof Error ? error.message : undefined}
        onRetry={refetchAll}
      />
    );
  }

  const summary = summaryQuery.data;
  const status = statusQuery.data;

  const decisionCounts: Record<string, number> = {};
  if (metasQuery.data) {
    for (const m of metasQuery.data.items) {
      decisionCounts[m.decision] = (decisionCounts[m.decision] ?? 0) + 1;
    }
  }

  const stale = status.last_cycle_at
    ? Date.now() - new Date(status.last_cycle_at).getTime() > LIVE_STALE_AFTER_MS
    : status.worker_alive;

  const datum: TriageSummaryDatum = {
    rawAlerts: summary.raw_alert_count,
    metaAlerts: summary.meta_alert_count,
    arrPercent: summary.alert_reduction_rate_percent ?? null,
    escalateOpen: summary.escalate_count,
    outboxPending: status.outbox_pending,
    decisionCounts,
    decisionSampled: metasQuery.data ? metasQuery.data.items.length : 0,
    decisionTotal: metasQuery.data ? metasQuery.data.total : 0,
    decisionUnavailable: metasQuery.isError || !metasQuery.data,
    eventLagSec: status.event_lag_sec,
    stale,
    workerAlive: status.worker_alive,
  };

  return (
    <LiveTriageSummaryPanel isLoading={false} isError={false} data={datum} onRetry={refetchAll} />
  );
}
