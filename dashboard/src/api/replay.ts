import { apiFetch } from './client';
import { ReplayStatusSchema } from './schemas';
import type { ReplayStatus } from './schemas';

export async function fetchReplayStatus(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/status');
  return ReplayStatusSchema.parse(data);
}

export async function startReplay(dataset: string, speed: number | string = 'MAX'): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/start', {
    method: 'POST',
    body: JSON.stringify({ dataset, speed }),
  });
  return ReplayStatusSchema.parse(data);
}

export async function pauseReplay(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/pause', { method: 'POST' });
  return ReplayStatusSchema.parse(data);
}

export async function resumeReplay(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/resume', { method: 'POST' });
  return ReplayStatusSchema.parse(data);
}

export async function stopReplay(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/stop', { method: 'POST' });
  return ReplayStatusSchema.parse(data);
}

export async function resetReplay(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/reset', { method: 'POST' });
  return ReplayStatusSchema.parse(data);
}
