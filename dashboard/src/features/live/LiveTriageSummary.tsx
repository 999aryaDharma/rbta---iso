import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary } from '@/api/dashboard';
import { fetchLiveStatus } from '@/api/live';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { LIVE_STALE_AFTER_MS } from './LiveStatusPanel';
import { ArrowRight, ListChecks } from '@phosphor-icons/react';

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

const LEVEL_BAR: Record<string, string> = {
  CRITICAL: 'bg-red-500',
  SUSPICIOUS: 'bg-amber-500',
  NOISE_HIGH: 'bg-sky-500',
  NOISE: 'bg-kumo-subtle',
};

function formatLag(sec: number | null): string {
  if (sec === null || Number.isNaN(sec)) return '-';
  if (sec < 60) return `${sec.toFixed(1)} dtk`;
  return `${(sec / 60).toFixed(1)} mnt`;
}

function lagTone(sec: number | null, stale: boolean): string {
  if (sec === null || Number.isNaN(sec)) return 'text-kumo-subtle';
  if (stale || sec >= 300) return 'text-red-600';
  if (sec >= 60) return 'text-amber-600';
  return 'text-emerald-600';
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
  const escalate = data.escalateOpen ?? 0;

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

      <div
        role="img"
        aria-label={`Reduksi: ${data.rawAlerts ?? '-'} alert menjadi ${data.metaAlerts ?? '-'} MetaAlert, ARR ${formatArr(data.arrPercent)}`}
        className="flex items-center gap-3 rounded-lg bg-kumo-recessed/40 px-4 py-3 text-xs"
      >
        <div className="min-w-0">
          <p className="font-mono font-bold text-kumo-strong text-lg leading-none">{data.rawAlerts ?? '-'}</p>
          <p className="text-kumo-subtle mt-1">Alert masuk</p>
        </div>
        <span aria-hidden className="text-kumo-subtle shrink-0"><ArrowRight size={16} /></span>
        <div className="min-w-0">
          <p className="font-mono font-bold text-kumo-strong text-lg leading-none">{data.metaAlerts ?? '-'}</p>
          <p className="text-kumo-subtle mt-1">MetaAlert final</p>
        </div>
        <span aria-hidden className="text-kumo-subtle shrink-0"><ArrowRight size={16} /></span>
        <div className="ml-auto text-right">
          <p className="font-mono font-bold text-kumo-brand text-lg leading-none">{formatArr(data.arrPercent)}</p>
          <p className="text-kumo-subtle mt-1">Reduksi live (ARR)</p>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
        <div>
          <p className="text-xs text-kumo-subtle font-medium">ESCALATE terbuka</p>
          <p className={`font-mono font-bold text-3xl leading-tight ${escalate > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
            {data.escalateOpen ?? '-'}
          </p>
          <p className="text-[11px] text-kumo-subtle">menunggu triase analis</p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Antrean Telegram</p>
          <p className="font-mono font-semibold text-kumo-default text-xl leading-tight">
            {data.outboxPending ?? '-'}
          </p>
        </div>
        <div>
          <p className="text-xs text-kumo-subtle font-medium">Lag event</p>
          <p className={`font-mono font-semibold text-xl leading-tight ${lagTone(data.eventLagSec, data.stale)}`}>
            {formatLag(data.eventLagSec)}
          </p>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-xs uppercase tracking-wider text-kumo-strong mb-2">
          Distribusi level (live)
        </h4>
        {data.decisionUnavailable ? (
          <p className="text-xs text-kumo-subtle">
            Distribusi level tidak tersedia. Daftar MetaAlert live gagal dimuat.
          </p>
        ) : decisions.length === 0 ? (
          <p className="text-xs text-kumo-subtle">
            Belum ada MetaAlert live untuk distribusi level.
          </p>
        ) : (
          <div className="space-y-2">
            <div
              role="img"
              aria-label={decisions.map(([name, value]) => `${name} ${value}`).join(', ')}
              className="flex h-3 overflow-hidden rounded-full bg-kumo-recessed"
            >
              {decisions.map(([name, value]) => (
                <div
                  key={name}
                  title={`${name}: ${value}`}
                  className={`h-full ${LEVEL_BAR[name] ?? 'bg-kumo-brand'}`}
                  style={{ width: `${(value / decisionTotal) * 100}%` }}
                />
              ))}
            </div>
            <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
              {decisions.map(([name, value]) => (
                <li key={name} className="flex items-center gap-1.5">
                  <span aria-hidden className={`inline-block w-2.5 h-2.5 rounded-sm ${LEVEL_BAR[name] ?? 'bg-kumo-brand'}`} />
                  <span className="font-semibold text-kumo-default">{name}</span>
                  <span className="font-mono text-kumo-subtle">{value}</span>
                </li>
              ))}
            </ul>
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
