import { apiFetch } from './client';
import {
  DashboardSummarySchema,
  AgentStateSchema,
  BucketStateSchema,
  TimeseriesSchema,
  SystemInfoSchema,
  IntegrationsSchema,
} from './schemas';
import type {
  DashboardSummary,
  AgentState,
  BucketState,
  TimeseriesData,
  SystemInfo,
  IntegrationsData,
} from './schemas';
import { z } from 'zod';

export async function fetchSummary(runId?: string): Promise<DashboardSummary> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const data = await apiFetch<unknown>(`/dashboard/summary${query}`);
  return DashboardSummarySchema.parse(data);
}

export async function fetchAgents(runId?: string): Promise<AgentState[]> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const data = await apiFetch<unknown>(`/dashboard/agents${query}`);
  return z.array(AgentStateSchema).parse(data);
}

export async function fetchBuckets(runId?: string): Promise<BucketState[]> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const data = await apiFetch<unknown>(`/dashboard/buckets${query}`);
  return z.array(BucketStateSchema).parse(data);
}

export async function fetchTimeseries(windowHours: number = 24, runId?: string): Promise<TimeseriesData> {
  const queryParams = new URLSearchParams();
  queryParams.set('window_hours', String(windowHours));
  if (runId) queryParams.set('run_id', runId);
  const data = await apiFetch<unknown>(`/dashboard/timeseries?${queryParams.toString()}`);
  return TimeseriesSchema.parse(data);
}

export async function fetchSystemInfo(runId?: string): Promise<SystemInfo> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const data = await apiFetch<unknown>(`/dashboard/system${query}`);
  return SystemInfoSchema.parse(data);
}

export async function fetchIntegrations(): Promise<IntegrationsData> {
  const data = await apiFetch<unknown>('/dashboard/integrations');
  return IntegrationsSchema.parse(data);
}
