import { apiFetch } from './client';
import { MetaAlertListSchema, MetaAlertSchema, TraceSchema } from './schemas';
import type { MetaAlertList, MetaAlert, Trace } from './schemas';

export interface MetaAlertFilters {
  page?: number;
  page_size?: number;
  decision?: string;
  agent_id?: string;
  rule_group?: string;
  search?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export async function fetchMetaAlerts(filters: MetaAlertFilters = {}): Promise<MetaAlertList> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== '') params.set(k, String(v));
  });
  const query = params.toString();
  const url = query ? `/meta-alerts?${query}` : '/meta-alerts';
  const data = await apiFetch<unknown>(url);
  return MetaAlertListSchema.parse(data);
}

export async function fetchMetaAlert(metaId: number): Promise<MetaAlert> {
  const data = await apiFetch<unknown>(`/meta-alerts/${metaId}`);
  return MetaAlertSchema.parse(data);
}

export async function fetchMetaAlertTrace(metaId: number): Promise<Trace> {
  const data = await apiFetch<unknown>(`/meta-alerts/${metaId}/trace`);
  return TraceSchema.parse(data);
}
