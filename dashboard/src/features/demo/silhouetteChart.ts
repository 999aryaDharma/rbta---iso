export type SilhouetteHistogramPoint = { x: number; count: number };

export function getSilhouetteChartBounds(
  histogram: SilhouetteHistogramPoint[],
  observed?: number | null,
  nullMean?: number | null,
) {
  const references = [observed, nullMean].filter((value): value is number => Number.isFinite(value));
  const values = [...histogram.map((point) => point.x), ...references];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, Math.abs(max) * 0.1, 0.001);
  const horizontalPadding = span * 0.06;
  const maximumCount = Math.max(...histogram.map((point) => point.count), 0);
  const yCeiling = Math.max(5, Math.ceil((maximumCount * 1.1) / 5) * 5);

  return {
    xDomain: [min - horizontalPadding, max + horizontalPadding] as [number, number],
    yDomain: [0, yCeiling] as [number, number],
  };
}
