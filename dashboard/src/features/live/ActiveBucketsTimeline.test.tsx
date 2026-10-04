import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ActiveBucketsTimeline } from './ActiveBucketsTimeline';
import type { TimelineBar } from './bucketTimeline';

const bars: TimelineBar[] = [
  {
    key: '005|rootcheck',
    agent_id: '005',
    agent_name: 'rbta-arya',
    rule_group_primary: 'rootcheck',
    meta_id: null,
    alert_count: 49,
    max_severity: 7,
    tier: 'high',
    finalized: false,
    start_time: new Date(Date.now() - 10 * 60_000).toISOString(),
    end_time: new Date(Date.now() - 2 * 60_000).toISOString(),
    x0: 0.1,
    x1: 0.9,
  },
];

describe('ActiveBucketsTimeline', () => {
  it('menampilkan state kosong saat belum ada bucket', () => {
    render(<ActiveBucketsTimeline bars={[]} windowMinutes={15} onSelect={() => {}} />);
    expect(screen.getByRole('status')).toHaveTextContent(/belum ada bucket aktif/i);
  });

  it('menandai baris yang masih mengagregasi tanpa link', () => {
    render(<ActiveBucketsTimeline bars={bars} windowMinutes={15} onSelect={() => {}} />);
    expect(screen.getByText(/mengagregasi/i)).toBeInTheDocument();
    expect(screen.getByText(/49/)).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('baris final memanggil onSelect dengan meta_id', () => {
    const onSelect = vi.fn();
    const final = [{ ...bars[0], meta_id: 5, finalized: true }];
    render(<ActiveBucketsTimeline bars={final} windowMinutes={15} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole('button', { name: /metaalert 5/i }));
    expect(onSelect).toHaveBeenCalledWith(5);
  });
});
