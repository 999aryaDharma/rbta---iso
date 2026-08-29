import { apiFetch } from './client';
import { RawAlertListSchema, RawAlertSchema } from './schemas';
import type { RawAlertList, RawAlert } from './schemas';

export interface RawAlertsQueryParams {
  page?: number;
  page_size?: number;
  search?: string;
  rule_id?: string;
  level_min?: number;
  level_max?: number;
  srcip?: string;
  mitre_tactic?: string;
  run_id?: string;
}

export async function fetchMetaAlertRawAlerts(
  metaId: number,
  params: RawAlertsQueryParams = {}
): Promise<RawAlertList> {
  const queryParams = new URLSearchParams();
  if (params.page) queryParams.set('page', String(params.page));
  if (params.page_size) queryParams.set('page_size', String(params.page_size));
  if (params.search) queryParams.set('search', params.search);
  if (params.rule_id) queryParams.set('rule_id', params.rule_id);
  if (params.level_min !== undefined) queryParams.set('level_min', String(params.level_min));
  if (params.level_max !== undefined) queryParams.set('level_max', String(params.level_max));
  if (params.srcip) queryParams.set('srcip', params.srcip);
  if (params.mitre_tactic) queryParams.set('mitre_tactic', params.mitre_tactic);
  if (params.run_id) queryParams.set('run_id', params.run_id);

  const qs = queryParams.toString();
  const url = `/meta-alerts/${metaId}/raw-alerts${qs ? `?${qs}` : ''}`;
  const data = await apiFetch<unknown>(url);
  return RawAlertListSchema.parse(data);
}

export async function fetchRawAlert(alertId: string, runId?: string): Promise<RawAlert> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const data = await apiFetch<unknown>(`/raw-alerts/${encodeURIComponent(alertId)}${query}`);
  return RawAlertSchema.parse(data);
}
