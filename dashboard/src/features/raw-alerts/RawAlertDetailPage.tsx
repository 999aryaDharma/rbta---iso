import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchRawAlert } from '@/api/rawAlerts';
import { fetchMetaAlertTrace } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/utils';
import { Alert } from '@/components/ui/alert';
import { Copy, Check, ArrowLeft, ChevronLeft, ChevronRight, Shield, Terminal, Globe, Tag, AlertTriangle } from 'lucide-react';

export function RawAlertDetailPage() {
  const { metaId, alertId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');
  const [copied, setCopied] = useState(false);

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const numericMetaId = metaId ? Number(metaId) : null;

  const { data, isError } = useQuery({
    queryKey: ['raw-alert', alertId, runId || 'live'],
    queryFn: () => fetchRawAlert(alertId as string, runId || undefined),
    enabled: Boolean(alertId),
    retry: false,
  });

  const { data: trace } = useQuery({
    queryKey: ['meta-alert-trace', numericMetaId, runId || 'live'],
    queryFn: () => (numericMetaId ? fetchMetaAlertTrace(numericMetaId, runId || undefined) : null),
    enabled: Boolean(numericMetaId),
  });

  const sourceIds = trace?.source_alert_ids || [];
  const currentIndex = alertId ? sourceIds.indexOf(alertId) : -1;
  const prevId = currentIndex > 0 ? sourceIds[currentIndex - 1] : null;
  const nextId = currentIndex >= 0 && currentIndex < sourceIds.length - 1 ? sourceIds[currentIndex + 1] : null;

  // Keyboard navigation shortcuts [ and ]
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) {
        return;
      }
      if (e.key === '[' && prevId && metaId) {
        e.preventDefault();
        navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(prevId)}`));
      } else if (e.key === ']' && nextId && metaId) {
        e.preventDefault();
        navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(nextId)}`));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [prevId, nextId, metaId, runId, navigate]);

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
        <Link to={withRunId('/meta-alerts')} className="hover:underline">MetaAlerts</Link>
        {metaId && (
          <>
            <span>/</span>
            <Link to={withRunId(`/meta-alerts/${metaId}`)} className="hover:underline">#{metaId}</Link>
            <span>/</span>
            <Link to={withRunId(`/meta-alerts/${metaId}/raw-alerts`)} className="hover:underline">Raw Alerts</Link>
          </>
        )}
        <span>/</span>
        <span className="truncate max-w-xs" style={{ color: 'var(--text-primary)' }}>{alertId}</span>
      </div>

      <PageHeader
        title="Raw Alert Forensic Evidence"
        description={`Wazuh Alert ID: ${alertId}`}
        actions={
          <div className="flex items-center gap-2">
            {/* Previous / Next Navigation */}
            {metaId && sourceIds.length > 0 && (
              <div className="flex items-center gap-1 mr-2">
                <button
                  disabled={!prevId}
                  onClick={() => prevId && navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(prevId)}`))}
                  title="Previous Alert ([)"
                  className="flex items-center gap-1 px-2.5 py-1.5 border rounded-[5px] text-xs font-medium disabled:opacity-40 cursor-pointer"
                  style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
                >
                  <ChevronLeft size={14} /> Prev
                </button>

                <span className="text-xs font-mono px-2" style={{ color: 'var(--text-tertiary)' }}>
                  {currentIndex >= 0 ? `${currentIndex + 1} of ${sourceIds.length}` : '—'}
                </span>

                <button
                  disabled={!nextId}
                  onClick={() => nextId && navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(nextId)}`))}
                  title="Next Alert (])"
                  className="flex items-center gap-1 px-2.5 py-1.5 border rounded-[5px] text-xs font-medium disabled:opacity-40 cursor-pointer"
                  style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
                >
                  Next <ChevronRight size={14} />
                </button>
              </div>
            )}

            {data && (
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-secondary)' }}
              >
                {copied ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
                {copied ? 'Copied JSON' : 'Copy Evidence JSON'}
              </button>
            )}

            {metaId && (
              <button
                onClick={() => navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts`))}
                className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium cursor-pointer"
                style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-secondary)' }}
              >
                <ArrowLeft size={14} /> Back to Member List
              </button>
            )}
          </div>
        }
      />

      {/* Unavailable Evidence Banner */}
      {isError && (
        <Alert variant="warning" className="my-4">
          <AlertTriangle size={18} className="shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-xs">Local Evidence Unavailable</div>
            <div className="mt-1 text-xs">
              Alert ID <code className="font-mono font-semibold">{alertId}</code> remains referenced by MetaAlert #{metaId} trace provenance, but its canonical audit record is not present in the local database.
            </div>
          </div>
        </Alert>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Structured Fields */}
          <div className="space-y-4">
            {/* Identity & Ingestion */}
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Tag size={16} style={{ color: 'var(--brand-orange)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Identity & Source</h3>
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Wazuh Alert ID:</dt> <dd className="font-mono font-semibold">{data.wazuh_alert_id}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Timestamp (UTC):</dt> <dd className="font-mono">{formatDateTime(data.timestamp)}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Source Mode:</dt> <dd className="font-mono">{data.source_mode || 'LIVE'}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Agent:</dt> <dd>{data.agent_name} ({data.agent_id}) · Crit: {data.agent_criticality}</dd></div>
                {data.source_index && (
                  <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Source Index:</dt> <dd className="font-mono">{data.source_index}</dd></div>
                )}
                {data.source_document_id && (
                  <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Document ID:</dt> <dd className="font-mono">{data.source_document_id}</dd></div>
                )}
              </dl>
            </div>

            {/* Rule Details */}
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Shield size={16} style={{ color: 'var(--action-blue)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Rule & Detection</h3>
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Rule ID:</dt> <dd className="font-mono font-semibold">{data.rule_id}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Severity Level:</dt> <dd className="font-mono font-semibold">{data.rule_level}/15</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Primary Group:</dt> <dd className="font-mono">{data.rule_group_primary}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Rule Groups:</dt> <dd className="font-mono">{data.rule_groups_all.join(', ') || data.rule_group_primary}</dd></div>
                <div className="pt-1"><dt style={{ color: 'var(--text-tertiary)' }}>Description:</dt> <dd className="mt-1 p-2 rounded-[3px] border text-xs" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-subtle)' }}>{data.rule_description}</dd></div>
              </dl>
            </div>

            {/* Network & MITRE */}
            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <Globe size={16} style={{ color: 'var(--success)' }} />
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Network & MITRE ATT&CK</h3>
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Source IP:</dt> <dd className="font-mono">{data.srcip || '—'}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Location:</dt> <dd className="font-mono">{data.location || '—'}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Decoder:</dt> <dd className="font-mono">{data.decoder || '—'}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>MITRE Tactics:</dt> <dd className="font-mono">{data.mitre_tactics.join(', ') || 'None'}</dd></div>
                <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>MITRE Techniques:</dt> <dd className="font-mono">{data.mitre_techniques.join(', ') || 'None'}</dd></div>
              </dl>
            </div>
          </div>

          {/* Right: Full Log & JSON Profile */}
          <div className="space-y-4">
            {data.full_log && (
              <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
                <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                  <Terminal size={16} style={{ color: 'var(--brand-orange)' }} />
                  <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Full Log Message</h3>
                </div>
                <pre className="text-xs font-mono p-3 rounded-[5px] border overflow-auto max-h-48 whitespace-pre-wrap" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                  {data.full_log}
                </pre>
              </div>
            )}

            <div className="p-5 rounded-[7px] border" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
              <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <h3 className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>Canonical & Evidence JSON</h3>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-[3px] border" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>READ ONLY</span>
              </div>
              <pre className="text-xs font-mono p-3 rounded-[5px] border overflow-auto max-h-[500px]" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
