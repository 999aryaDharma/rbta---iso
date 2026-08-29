import { describe, it, expect } from 'vitest';
import { formatNumber, formatDateTime, formatScore, formatSeconds, formatDuration } from './formatters';

describe('Formatters Utilities', () => {
  it('formats numbers with thousand separators', () => {
    expect(formatNumber(1000)).toBe('1,000');
    expect(formatNumber(1234567)).toBe('1,234,567');
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(null)).toBe('0');
    expect(formatNumber(undefined)).toBe('0');
  });

  it('formats datetime strings into UTC format', () => {
    expect(formatDateTime('2026-08-29T10:00:00Z')).toContain('2026-08-29 10:00:00 UTC');
    expect(formatDateTime(null)).toBe('—');
    expect(formatDateTime(undefined)).toBe('—');
  });

  it('formats anomaly score to fixed precision', () => {
    expect(formatScore(0.852345, 4)).toBe('0.8523');
    expect(formatScore(1.2, 2)).toBe('1.20');
    expect(formatScore(null)).toBe('—');
  });

  it('formats seconds into human readable format', () => {
    expect(formatSeconds(45)).toBe('45.0s');
    expect(formatSeconds(125)).toBe('2m 5s');
    expect(formatSeconds(null)).toBe('—');
  });

  it('formats duration in wall-clock seconds', () => {
    expect(formatDuration(45)).toBe('45.0s');
    expect(formatDuration(125)).toBe('2m 5s');
    expect(formatDuration(3665)).toBe('1h 1m 5s');
    expect(formatDuration(null)).toBe('0s');
  });
});
