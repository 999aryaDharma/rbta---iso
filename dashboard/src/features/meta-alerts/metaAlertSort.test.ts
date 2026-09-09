import { describe, expect, it } from 'vitest';
import { nextAlertScoreSort } from './metaAlertSort';

describe('nextAlertScoreSort', () => {
  it('starts with the highest Alert Score and then toggles to the lowest', () => {
    expect(nextAlertScoreSort('end_time', 'desc')).toEqual({ sortBy: 'anomaly_score', sortOrder: 'desc' });
    expect(nextAlertScoreSort('anomaly_score', 'desc')).toEqual({ sortBy: 'anomaly_score', sortOrder: 'asc' });
    expect(nextAlertScoreSort('anomaly_score', 'asc')).toEqual({ sortBy: 'anomaly_score', sortOrder: 'desc' });
  });
});
