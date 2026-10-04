import type { BucketState } from '@/api/schemas';

export type SeverityTier = 'critical' | 'high' | 'medium' | 'low';

/** Petakan level Wazuh 0–15 ke tier warna (murni presentasi, bukan vonis). */
export function severityTier(level: number): SeverityTier {
  if (level >= 12) return 'critical';
  if (level >= 7) return 'high';
  if (level >= 4) return 'medium';
  return 'low';
}

export interface TimelineBar {
  key: string;
  agent_id: string;
  agent_name: string | null;
  rule_group_primary: string;
  meta_id: number | null;
  alert_count: number;
  max_severity: number;
  tier: SeverityTier;
  finalized: boolean;
  /** ISO event-time window bucket (untuk kolom durasi). */
  start_time: string;
  end_time: string;
  /** Fraksi 0..1 dalam jendela geser (terjepit). */
  x0: number;
  x1: number;
}

/**
 * Ubah bucket aktif menjadi bar timeline dalam jendela geser
 * [nowMs - windowMs, nowMs] berbasis event-time Wazuh.
 */
export function buildTimelineBars(
  buckets: BucketState[],
  nowMs: number,
  windowMs: number,
): TimelineBar[] {
  const start = nowMs - windowMs;
  const bars: TimelineBar[] = [];

  for (const b of buckets) {
    const s = new Date(b.start_time).getTime();
    const e = new Date(b.end_time).getTime();
    if (Number.isNaN(s) || Number.isNaN(e)) continue;
    if (e <= start || s >= nowMs) continue;
    const x0 = Math.max(0, (Math.max(s, start) - start) / windowMs);
    const x1 = Math.min(1, (Math.min(e, nowMs) - start) / windowMs);
    if (x1 <= x0) continue;
    bars.push({
      key: `${b.agent_id}|${b.rule_group_primary}`,
      agent_id: b.agent_id,
      agent_name: b.agent_name ?? null,
      rule_group_primary: b.rule_group_primary,
      meta_id: b.meta_id ?? null,
      alert_count: b.alert_count,
      max_severity: b.max_severity,
      tier: severityTier(b.max_severity),
      start_time: b.start_time,
      end_time: b.end_time,
      // Kontrak: hanya flag backend yang menentukan final. meta_id saja
      // berarti reservasi bucket aktif, bukan MetaAlert jadi.
      finalized: b.finalized === true,
      x0,
      x1,
    });
  }

  bars.sort((a, b) =>
    a.agent_id === b.agent_id
      ? a.rule_group_primary.localeCompare(b.rule_group_primary)
      : a.agent_id.localeCompare(b.agent_id),
  );
  return bars;
}
