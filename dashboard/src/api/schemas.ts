import { z } from 'zod';

export const DashboardSummarySchema = z.object({
  raw_alert_count: z.number(),
  meta_alert_count: z.number(),
  alert_reduction_rate_percent: z.number().nullable().optional(),
  escalate_count: z.number(),
  digest_count: z.number(),
  suppress_count: z.number(),
  anomalies_detected: z.number(),
  critical_meta_count: z.number(),
  active_buckets_count: z.number(),
  source_mode: z.string(),
  system_status: z.string(),
});

export type DashboardSummary = z.infer<typeof DashboardSummarySchema>;

export const AgentStateSchema = z.object({
  agent_id: z.string(),
  agent_name: z.string(),
  event_count: z.number(),
  warmup_required: z.number(),
  warmup_progress: z.number(),
  is_warmed_up: z.boolean(),
  baseline_gap_seconds: z.number().nullable().optional(),
  ema_gap_seconds: z.number().nullable().optional(),
  base_delta_t_seconds: z.number(),
  current_delta_t_seconds: z.number(),
  active_bucket_count: z.number(),
  status: z.string(),
});

export type AgentState = z.infer<typeof AgentStateSchema>;

export const BucketStateSchema = z.object({
  meta_id: z.number().nullable().optional(),
  agent_id: z.string(),
  agent_name: z.string().nullable().optional(),
  rule_group_primary: z.string(),
  start_time: z.string(),
  end_time: z.string(),
  alert_count: z.number(),
  max_severity: z.number(),
});

export type BucketState = z.infer<typeof BucketStateSchema>;

export const MetaAlertSchema = z.object({
  meta_id: z.number(),
  agent_id: z.string(),
  agent_name: z.string(),
  rule_group_primary: z.string(),
  start_time: z.string(),
  end_time: z.string(),
  alert_count: z.number(),
  max_severity: z.number(),
  mitre_tactics: z.array(z.string()).default([]),
  seven_features: z.record(z.string(), z.number()),
  raw_model_score: z.number(),
  anomaly_score: z.number(),
  threshold_used: z.number(),
  decision: z.string(),
  action: z.string(),
  escalate: z.boolean(),
  model_version: z.string(),
  feature_schema_version: z.string(),
  score_calibration_version: z.string(),
  source_alert_ids: z.array(z.string()).default([]),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type MetaAlert = z.infer<typeof MetaAlertSchema>;

export const MetaAlertListSchema = z.object({
  items: z.array(MetaAlertSchema),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
});

export type MetaAlertList = z.infer<typeof MetaAlertListSchema>;

export const RawAlertSchema = z.object({
  wazuh_alert_id: z.string(),
  timestamp: z.string(),
  agent_id: z.string(),
  agent_name: z.string(),
  rule_id: z.string(),
  rule_level: z.number(),
  rule_description: z.string().optional().default(''),
  rule_group_primary: z.string(),
  rule_groups_all: z.array(z.string()).default([]),
  mitre_tactics: z.array(z.string()).default([]),
  mitre_techniques: z.array(z.string()).default([]),
  srcip: z.string().nullable().optional(),
  location: z.string().optional().default(''),
  decoder: z.string().optional().default(''),
  full_log: z.string().optional().default(''),
  agent_criticality: z.number(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  original_source_payload: z.record(z.string(), z.unknown()).nullable().optional(),
  source_index: z.string().optional().default(''),
  source_document_id: z.string().optional().default(''),
  source_mode: z.string().optional().default(''),
  ingested_at: z.string().optional().default(''),
});

export type RawAlert = z.infer<typeof RawAlertSchema>;

export const RawAlertListSchema = z.object({
  meta_id: z.number().nullable().optional(),
  source_total: z.number(),
  resolved_total: z.number(),
  filtered_total: z.number(),
  unresolved_alert_ids: z.array(z.string()).default([]),
  items: z.array(RawAlertSchema),
  page: z.number(),
  page_size: z.number(),
});

export type RawAlertList = z.infer<typeof RawAlertListSchema>;

export const ReplayStatusSchema = z.object({
  run_id: z.string().nullable(),
  status: z.enum(['IDLE', 'RUNNING', 'PAUSED', 'STOPPED', 'COMPLETED', 'ERROR']),
  dataset: z.string().nullable(),
  processed_count: z.number(),
  total_count: z.number(),
  progress: z.number(),
  current_event_time: z.string().nullable(),
  wall_clock_elapsed_seconds: z.number(),
  speed: z.union([z.number(), z.string()]),
  events_per_second: z.number(),
  model_version: z.string(),
  dataset_mode: z.string().nullable().optional(),
  dataset_count: z.number().nullable().optional(),
  current_dataset: z.string().nullable().optional(),
  current_dataset_index: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
  last_error: z.record(z.string(), z.unknown()).nullable().optional(),
});

export type ReplayStatus = z.infer<typeof ReplayStatusSchema>;

export const ReplayDatasetSchema = z.object({
  name: z.string(),
  size_bytes: z.number(),
});

export const ReplayDatasetListSchema = z.object({
  items: z.array(ReplayDatasetSchema),
});

export type ReplayDatasetList = z.infer<typeof ReplayDatasetListSchema>;

export const TimeseriesPointSchema = z.object({
  timestamp: z.string(),
  raw_alerts: z.number(),
  meta_alerts: z.number(),
});

export const TimeseriesSchema = z.array(TimeseriesPointSchema);

export type TimeseriesData = z.infer<typeof TimeseriesSchema>;

export const SystemInfoSchema = z.object({
  model_version: z.string(),
  tukey_threshold: z.number(),
  random_state: z.number().nullable().optional(),
  feature_names: z.array(z.string()).default([]),
  base_delta_t_seconds: z.number(),
  adaptive: z.boolean(),
  source_mode: z.string(),
  durable_state_path: z.string(),
  raw_evidence_db_path: z.string().nullable().optional(),
  system_status: z.string(),
});

export type SystemInfo = z.infer<typeof SystemInfoSchema>;

export const TraceSchema = z.object({
  meta_id: z.number(),
  source_alert_ids: z.array(z.string()),
  agent_id: z.string(),
  rule_group_primary: z.string(),
  decision: z.string(),
  action: z.string(),
  model_version: z.string(),
});

export type Trace = z.infer<typeof TraceSchema>;

export const IntegrationItemSchema = z.object({
  name: z.string().optional(),
  status: z.string(),
  detail: z.string().optional(),
});

export const IntegrationsSchema = z.record(z.string(), IntegrationItemSchema);

export type IntegrationsData = z.infer<typeof IntegrationsSchema>;
