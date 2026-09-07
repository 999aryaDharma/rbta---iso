import type { LiveEvaluation } from '@/api/schemas';
import { formatNumber } from '@/lib/formatters';

function Stat({ label, value, help }: { label: string; value: string; help: string }) {
  return <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas p-4"><div className="text-[11px] font-semibold uppercase tracking-wider text-kumo-subtle">{label}</div><div className="mt-1 text-2xl font-bold text-kumo-strong">{value}</div><p className="mt-1 text-xs leading-5 text-kumo-subtle">{help}</p></div>;
}

export function LiveEvaluationPanel({ live, eventsPerSecond }: { live?: LiveEvaluation; eventsPerSecond: number }) {
  if (!live) return null;
  const maxBin = Math.max(1, ...live.score_distribution.histogram);
  return (
    <section className="space-y-4" aria-labelledby="live-evaluation-title">
      <div><h2 id="live-evaluation-title" className="text-lg font-bold text-kumo-strong">Evaluasi real-time</h2><p className="text-sm text-kumo-subtle">Metrik deskriptif diperbarui selama replay tanpa melatih ulang model.</p></div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="ARR sementara" value={`${live.live_arr_percent.toFixed(2)}%`} help={live.claim_boundary.arr_interpretation} />
        <Stat label="Unit triase" value={formatNumber(live.triage_units_current)} help={`${formatNumber(live.raw_alerts)} alert mentah → meta-alert aktif + final`} />
        <Stat label="Throughput" value={`${formatNumber(eventsPerSecond)} ev/s`} help="Kecepatan pemrosesan replay pada host demo." />
        <Stat label="Evidence coverage" value={`${live.evidence_coverage_percent.toFixed(2)}%`} help="Proporsi alert replay yang tersimpan sebagai bukti mentah." />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas p-5">
          <h3 className="text-sm font-bold text-kumo-strong">Distribusi skor model</h3>
          <p className="mt-1 text-xs text-kumo-subtle">Skor terkalibrasi dapat berada di luar 0–1; itu bukan error.</p>
          <div className="mt-5 flex h-28 items-end gap-2" aria-label="Histogram skor anomaly">
            {live.score_distribution.histogram.map((value, index) => <div key={live.score_distribution.bins[index]} className="flex flex-1 flex-col items-center gap-2"><span className="text-[10px] font-mono text-kumo-subtle">{value}</span><div className="w-full rounded-t bg-blue-500/80" style={{ height: `${Math.max(3, value / maxBin * 72)}px` }} /><span className="text-[9px] text-kumo-subtle">{live.score_distribution.bins[index]}</span></div>)}
          </div>
          {live.reference_range_exceedance_count > 0 && <p className="mt-4 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">{live.reference_range_exceedance_count} skor di luar rentang referensi 0–1; nilai asli dipertahankan agar auditabel.</p>}
        </div>
        <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas p-5">
          <h3 className="text-sm font-bold text-kumo-strong">Keputusan triase</h3>
          <div className="mt-4 space-y-3">{Object.entries(live.decision_distribution).map(([name, value]) => { const total = Math.max(1, live.finalized_meta_alerts); return <div key={name}><div className="mb-1 flex justify-between text-xs"><span className="font-semibold text-kumo-default">{name}</span><span className="font-mono text-kumo-subtle">{value}</span></div><div className="h-2 overflow-hidden rounded-full bg-kumo-recessed"><div className="h-full rounded-full bg-violet-500" style={{ width: `${value / total * 100}%` }} /></div></div>; })}</div>
          <p className="mt-4 border-t border-kumo-hairline pt-3 text-xs leading-5 text-kumo-subtle">{live.claim_boundary.score_interpretation}</p>
        </div>
      </div>
    </section>
  );
}
