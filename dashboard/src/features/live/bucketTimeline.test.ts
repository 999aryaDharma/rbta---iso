import { describe, expect, it } from 'vitest';
import { buildTimelineBars, severityTier } from './bucketTimeline';
import type { BucketState } from '@/api/schemas';

const NOW = new Date('2026-10-03T02:00:00Z').getTime();
const MIN = 60_000;

function bucket(partial: Partial<BucketState> = {}): BucketState {
  return {
    meta_id: null,
    agent_id: '005',
    agent_name: 'rbta-arya',
    rule_group_primary: 'rootcheck',
    start_time: new Date(NOW - 10 * MIN).toISOString(),
    end_time: new Date(NOW - 2 * MIN).toISOString(),
    alert_count: 49,
    max_severity: 7,
    ...partial,
  };
}

describe('severityTier', () => {
  it('memetakan level Wazuh ke tier', () => {
    expect(severityTier(12)).toBe('critical');
    expect(severityTier(7)).toBe('high');
    expect(severityTier(4)).toBe('medium');
    expect(severityTier(2)).toBe('low');
  });
});

describe('buildTimelineBars', () => {
  it('kosong bila tidak ada bucket', () => {
    expect(buildTimelineBars([], NOW, 15 * MIN)).toEqual([]);
  });

  it('mengurutkan per agen lalu rule group', () => {
    const bars = buildTimelineBars(
      [
        bucket({ agent_id: '007', rule_group_primary: 'syslog' }),
        bucket({ agent_id: '005', rule_group_primary: 'syslog' }),
        bucket({ agent_id: '005', rule_group_primary: 'rootcheck' }),
      ],
      NOW,
      15 * MIN,
    );
    expect(bars.map((b) => b.key)).toEqual([
      '005|rootcheck',
      '005|syslog',
      '007|syslog',
    ]);
  });

  it('menjepit bar ke jendela geser', () => {
    const bars = buildTimelineBars([bucket()], NOW, 15 * MIN);
    expect(bars).toHaveLength(1);
    expect(bars[0].x0).toBeGreaterThanOrEqual(0);
    expect(bars[0].x1).toBeLessThanOrEqual(1);
    expect(bars[0].x1).toBeGreaterThan(bars[0].x0);
  });

  it('menandai final bila meta_id sudah ada', () => {
    const bars = buildTimelineBars([bucket({ meta_id: 5 })], NOW, 15 * MIN);
    expect(bars[0].finalized).toBe(true);
    expect(bars[0].meta_id).toBe(5);
  });
});
