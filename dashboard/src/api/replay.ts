import { apiFetch } from './client';
import { ReplayStatusSchema, ReplayDatasetListSchema, DatasetCatalogStatusSchema, TelegramPayloadListSchema, EvaluationStatusSchema } from './schemas';
import type { ReplayStatus, ReplayDatasetList, DatasetCatalogStatus, TelegramPayloadList, EvaluationStatus } from './schemas';
import { getApiKey } from '@/lib/auth';

export async function fetchReplayDatasets(): Promise<ReplayDatasetList> {
  const data = await apiFetch<unknown>('/replay/datasets');
  return ReplayDatasetListSchema.parse(data);
}

export async function fetchReplayStatus(): Promise<ReplayStatus> {
  const data = await apiFetch<unknown>('/replay/status');
  return ReplayStatusSchema.parse(data);
}

export async function fetchDatasetCatalogStatus(): Promise<DatasetCatalogStatus> {
  const data = await apiFetch<unknown>('/replay/datasets/catalog-status');
  return DatasetCatalogStatusSchema.parse(data);
}

export async function refreshDatasetCatalog(): Promise<DatasetCatalogStatus> {
  const data = await apiFetch<unknown>('/replay/datasets/refresh', { method: 'POST' });
  return DatasetCatalogStatusSchema.parse(data);
}

export async function fetchTelegramPayloads(limit: number = 50): Promise<TelegramPayloadList> {
  const data = await apiFetch<unknown>(`/replay/telegram-payloads?limit=${limit}`);
  return TelegramPayloadListSchema.parse(data);
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

export async function fetchEvaluationStatus(): Promise<EvaluationStatus> {
  const data = await apiFetch<unknown>('/replay/evaluation/status');
  return EvaluationStatusSchema.parse(data);
}

export async function startEvaluation(): Promise<EvaluationStatus> {
  const data = await apiFetch<unknown>('/replay/evaluation/start', { method: 'POST' });
  return EvaluationStatusSchema.parse(data);
}

export async function cancelEvaluation(): Promise<EvaluationStatus> {
  const data = await apiFetch<unknown>('/replay/evaluation/cancel', { method: 'POST' });
  return EvaluationStatusSchema.parse(data);
}

export async function downloadEvaluationArtifact(): Promise<void> {
  const headers: HeadersInit = {};
  const apiKey = getApiKey();
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  const response = await fetch('/api/v1/replay/evaluation/artifact', { headers });
  if (!response.ok) throw new Error(`Artifact download failed (${response.status})`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'rbta-iso-evaluation.json';
  anchor.click();
  URL.revokeObjectURL(url);
}
