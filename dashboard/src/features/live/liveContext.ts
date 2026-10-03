import type { LiveContext } from './LiveReplaySwitcher';

/**
 * Konteks MetaAlert dilihat dari parameter URL `run_id`:
 * tanpa run_id = data live dari service; run_id terisi = run replay
 * terisolasi yang lifecycle-nya dimiliki halaman Replay (/demo).
 */
export function resolveLiveContext(runId: string | null | undefined): LiveContext {
  return runId ? 'replay' : 'live';
}
