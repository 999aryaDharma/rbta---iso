import { apiFetch } from './client';
import { DashboardSummarySchema, TimeseriesSchema, SystemInfoSchema, AgentStateSchema, BucketStateSchema } from './schemas';
import type { DashboardSummary, TimeseriesData, SystemInfo, AgentState, BucketState } from './schemas';

export async function fetchSummary(): Promise<DashboardSummary> {
  const data = await apiFetch<unknown>('/dashboard/summary');
  return DashboardSummarySchema.parse(data);
}

export async function fetchTimeseries(): Promise<TimeseriesData> {
  const data = await apiFetch<unknown>('/dashboard/timeseries');
  return TimeseriesSchema.parse(data);
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const data = await apiFetch<unknown>('/dashboard/system');
  return SystemInfoSchema.parse(data);
}

export async function fetchAgents(): Promise<AgentState[]> {
  const data = await apiFetch<unknown[]>('/dashboard/agents');
  return data.map((d) => AgentStateSchema.parse(d));
}

export async function fetchBuckets(): Promise<BucketState[]> {
  const data = await apiFetch<unknown[]>('/dashboard/buckets');
  return data.map((d) => BucketStateSchema.parse(d));
}
