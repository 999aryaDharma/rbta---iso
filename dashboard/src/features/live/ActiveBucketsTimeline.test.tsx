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
    x0: 0.1,
    x1: 0.9,
  },
];

describe('ActiveBucketsTimeline', () => {
  it('menampilkan state kosong saat belum ada bucket', () => {
    render(<ActiveBucketsTimeline bars={[]} windowMinutes={15} onSelect={() => {}} />);
    expect(screen.getByRole('status')).toHaveTextContent(/belum ada bucket aktif/i);
  });

  it('menandai bar yang masih mengagregasi', () => {
    render(<ActiveBucketsTimeline bars={bars} windowMinutes={15} onSelect={() => {}} />);
    expect(screen.getByText(/mengagregasi/i)).toBeInTheDocument();
    expect(screen.getByText(/49/)).toBeInTheDocument();
  });

  it('klik bar final memanggil onSelect dengan meta_id', () => {
    const onSelect = vi.fn();
    const final = [{ ...bars[0], meta_id: 5, finalized: true }];
    render(<ActiveBucketsTimeline bars={final} windowMinutes={15} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole('button', { name: /rootcheck/i }));
    expect(onSelect).toHaveBeenCalledWith(5);
  });
});
