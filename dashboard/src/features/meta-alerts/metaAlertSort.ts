export type MetaAlertSortOrder = 'asc' | 'desc';

export function nextAlertScoreSort(
  currentSortBy: string,
  currentSortOrder: MetaAlertSortOrder,
): { sortBy: 'anomaly_score'; sortOrder: MetaAlertSortOrder } {
  return {
    sortBy: 'anomaly_score',
    sortOrder: currentSortBy === 'anomaly_score' && currentSortOrder === 'desc' ? 'asc' : 'desc',
  };
}
