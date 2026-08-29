import { apiFetch } from './client';
import { RawAlertListSchema, RawAlertSchema } from './schemas';
import type { RawAlertList, RawAlert } from './schemas';

export interface RawAlertFilters {
  page?: number;
  page_size?: number;
  search?: string;
  rule_id?: string;
  level_min?: number;
  level_max?: number;
  srcip?: string;
  mitre_tactic?: string;
  from?: string;
  to?: string;
}

export async function fetchMetaAlertRawAlerts(
  metaId: number,
  filters: RawAlertFilters = {},
): Promise<RawAlertList> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== '') params.set(k, String(v));
  });
  const query = params.toString();
  const url = query ? `/meta-alerts/${metaId}/raw-alerts?${query}` : `/meta-alerts/${metaId}/raw-alerts`;
  const data = await apiFetch<unknown>(url);
  return RawAlertListSchema.parse(data);
}

export async function fetchRawAlert(alertId: string): Promise<RawAlert> {
  const data = await apiFetch<unknown>(`/raw-alerts/${encodeURIComponent(alertId)}`);
  return RawAlertSchema.parse(data);
}
