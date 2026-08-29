import { apiFetch } from './client';
import { ReplayStatusSchema, ReplayDatasetListSchema } from './schemas';
import type { ReplayStatus, ReplayDatasetList } from './schemas';

export async function fetchReplayDatasets(): Promise<ReplayDatasetList> {
  const data = await apiFetch<unknown>('/replay/datasets');
  return ReplayDatasetListSchema.parse(data);
}

export async function fetchReplayStatus(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/status');
  return ReplayStatusSchema.parse(data);
}

export async function startReplay(dataset_name: string, speed_factor: string = 'MAX'): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/start', {
    method: 'POST',
    body: JSON.stringify({ dataset_name, speed_factor }),
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
