import { z } from 'zod';

export const DashboardSummarySchema = z.object({
  raw_alert_count: z.number(),
  meta_alert_count: z.number(),
  alert_reduction_rate: z.number(),
  escalate_count: z.number(),
  suppress_count: z.number(),
  active_agents_count: z.number(),
  active_buckets_count: z.number(),
  outbox_depth: z.number(),
  source_mode: z.string(),
  model_version: z.string(),
  ready: z.boolean(),
  updated_at: z.string(),
});

export type DashboardSummary = z.infer<typeof DashboardSummarySchema>;

export const AgentStateSchema = z.object({
  agent_id: z.string(),
  agent_name: z.string(),
  event_count: z.number(),
  warmup_required: z.number(),
  warmup_progress: z.number(),
  is_warmed_up: z.boolean(),
  baseline_gap_seconds: z.number().nullable(),
  ema_gap_seconds: z.number().nullable(),
  base_delta_t_seconds: z.number(),
  current_delta_t_seconds: z.number(),
  active_bucket_count: z.number(),
  status: z.string(),
});

export type AgentState = z.infer<typeof AgentStateSchema>;

export const BucketStateSchema = z.object({
  meta_id: z.number(),
  agent_id: z.string(),
  agent_name: z.string(),
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
  mitre_tactics: z.array(z.string()),
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
  source_alert_ids: z.array(z.string()),
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
  rule_groups_all: z.array(z.string()).optional().default([]),
  mitre_tactics: z.array(z.string()).optional().default([]),
  mitre_techniques: z.array(z.string()).optional().default([]),
  srcip: z.string().nullable().optional(),
  location: z.string().optional().default(''),
  decoder: z.string().optional().default(''),
  full_log: z.string().optional().default(''),
  agent_criticality: z.number(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  original_source_payload: z.record(z.string(), z.unknown()).nullable().optional(),
  opensearch_index: z.string().optional().default(''),
  opensearch_document_id: z.string().optional().default(''),
  source_mode: z.string().optional().default(''),
  ingested_at: z.string().optional().default(''),
});

export type RawAlert = z.infer<typeof RawAlertSchema>;

export const RawAlertListSchema = z.object({
  meta_id: z.number(),
  total: z.number(),
  resolved_count: z.number(),
  unresolved_alert_ids: z.array(z.string()),
  items: z.array(RawAlertSchema),
  page: z.number(),
  page_size: z.number(),
});

export type RawAlertList = z.infer<typeof RawAlertListSchema>;

export const ReplayStatusSchema = z.object({
  run_id: z.string().nullable(),
  status: z.enum(['IDLE', 'RUNNING', 'PAUSED', 'COMPLETED', 'ERROR']),
  dataset: z.string().nullable(),
  processed_count: z.number(),
  total_count: z.number(),
  current_event_time: z.string().nullable(),
  wall_clock_elapsed_seconds: z.number(),
  speed: z.union([z.number(), z.string()]),
  events_per_second: z.number(),
  error: z.string().nullable().optional(),
});

export type ReplayStatus = z.infer<typeof ReplayStatusSchema>;

export const TimeseriesPointSchema = z.object({
  time: z.string(),
  raw_alerts: z.number(),
  meta_alerts: z.number(),
});

export const TimeseriesSchema = z.object({
  series: z.array(TimeseriesPointSchema),
});

export type TimeseriesData = z.infer<typeof TimeseriesSchema>;

export const SystemInfoSchema = z.object({
  api_status: z.string(),
  runtime_ready: z.boolean(),
  source_mode: z.string(),
  model_version: z.string(),
  feature_schema_version: z.string(),
  score_calibration_version: z.string(),
  threshold: z.number(),
  seen_alerts: z.number(),
  active_buckets: z.number(),
  outbox_depth: z.number(),
  current_run_id: z.string().nullable(),
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
