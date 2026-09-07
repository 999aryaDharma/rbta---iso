import type { ReplayDataset } from '@/api/schemas';
import { CheckCircle, ShieldWarning } from '@phosphor-icons/react';

export function ResearchBoundaryCard({ dataset, modelVersion }: { dataset?: ReplayDataset; modelVersion: string }) {
  const checks = [
    ['Dataset', dataset ? `${dataset.classification} · ${dataset.total_events.toLocaleString('id-ID')} alert` : 'belum dipilih'],
    ['Integritas', dataset?.sha256 ? `${dataset.is_valid ? 'valid' : `tidak valid (${dataset.invalid_events})`} · SHA-256 ${dataset.sha256.slice(0, 12)}…` : 'manifest belum tersedia'],
    ['Model beku', modelVersion || 'belum tersedia'],
  ];
  return (
    <section className="rounded-2xl border border-kumo-hairline bg-kumo-canvas p-6 shadow-xs" aria-labelledby="demo-boundary-title">
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <div className="mb-3 flex items-center gap-2 text-amber-700 dark:text-amber-300">
            <ShieldWarning size={22} weight="duotone" />
            <h2 id="demo-boundary-title" className="text-base font-bold text-kumo-strong">Apa yang dibuktikan demo ini?</h2>
          </div>
          <p className="max-w-3xl text-sm leading-6 text-kumo-default">
            Sistem mengelompokkan alert menjadi unit triase, mempertahankan bukti sumber, lalu memberi prioritas dengan skor keanehan. Ini <strong>bukan pendeteksi kebenaran serangan</strong> dan bukan pengukuran akurasi serangan tanpa ground truth.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {['RBTA: reduksi + konteks', 'IF: prioritas keanehan', 'Evidence: dapat ditelusuri'].map((item) => (
              <div key={item} className="rounded-xl bg-kumo-recessed/60 px-3 py-2 text-xs font-semibold text-kumo-strong">{item}</div>
            ))}
          </div>
        </div>
        <dl className="space-y-2 rounded-xl border border-kumo-hairline bg-kumo-recessed/30 p-4">
          {checks.map(([label, value]) => (
            <div key={label} className="flex items-start gap-2 text-xs">
              <CheckCircle className="mt-0.5 shrink-0 text-emerald-600" size={15} weight="fill" />
              <dt className="w-20 shrink-0 font-semibold text-kumo-subtle">{label}</dt>
              <dd className="break-all font-mono text-kumo-strong">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
