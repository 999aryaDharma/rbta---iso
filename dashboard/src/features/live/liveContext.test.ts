import { describe, expect, it } from 'vitest';
import { resolveLiveContext } from './liveContext';

describe('resolveLiveContext', () => {
  it('tanpa run_id berarti konteks live', () => {
    expect(resolveLiveContext(null)).toBe('live');
  });

  it('run_id kosong berarti konteks live', () => {
    expect(resolveLiveContext('')).toBe('live');
  });

  it('run_id terisi berarti konteks replay', () => {
    expect(resolveLiveContext('replay-20261001-abc')).toBe('replay');
  });
});
