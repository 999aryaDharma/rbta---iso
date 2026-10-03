import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { LiveStatusPanel } from './LiveStatusPanel';
import type { LiveStatus } from '@/api/live';

const baseStatus: LiveStatus = {
  worker_alive: true,
  cycles_completed: 12,
  consecutive_failures: 0,
  last_error: null,
  last_cycle_at: new Date().toISOString(),
  live_model_version: 'rbta-if-v1',
  recent_poll_cursor: new Date().toISOString(),
  lag_sec: 4.2,
  buffer_size: null,
  buffer_stats: null,
  outbox_pending: 1,
  dispatcher: null,
  quarantine_total: 0,
  newest_scored_event_time: null,
  newest_ingested_event_time: new Date().toISOString(),
  event_lag_sec: 4.2,
  tls_verify: true,
};

describe('LiveStatusPanel', () => {
  it('menampilkan state loading saat memuat', () => {
    render(<LiveStatusPanel isLoading isError={false} data={null} onRetry={() => {}} />);
    expect(screen.getByRole('status')).toHaveTextContent(/memuat/i);
  });

  it('menampilkan error + tombol coba lagi', () => {
    const onRetry = vi.fn();
    render(<LiveStatusPanel isLoading={false} isError data={null} errorMessage="putus" onRetry={onRetry} />);
    expect(screen.getByRole('alert')).toHaveTextContent(/putus/);
    fireEvent.click(screen.getByRole('button', { name: /coba lagi/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('menampilkan worker hidup, siklus, dan model terpin', () => {
    render(<LiveStatusPanel isLoading={false} isError={false} data={baseStatus} onRetry={() => {}} />);
    expect(screen.getByText(/worker hidup/i)).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('rbta-if-v1')).toBeInTheDocument();
  });

  it('menandai basi bila siklus terakhir sudah tua', () => {
    const stale = { ...baseStatus, last_cycle_at: new Date(Date.now() - 10 * 60_000).toISOString() };
    render(<LiveStatusPanel isLoading={false} isError={false} data={stale} onRetry={() => {}} />);
    expect(screen.getByText(/basi/i)).toBeInTheDocument();
  });

  it('menampilkan worker berhenti saat mode replay', () => {
    const stopped = { ...baseStatus, worker_alive: false, cycles_completed: 0 };
    render(<LiveStatusPanel isLoading={false} isError={false} data={stopped} onRetry={() => {}} />);
    expect(screen.getAllByText(/worker berhenti/i)).toHaveLength(2);
  });
});
