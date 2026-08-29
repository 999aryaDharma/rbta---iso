import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchRawAlert } from '@/api/rawAlerts';
import { fetchMetaAlertTrace } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/formatters';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Button } from '@cloudflare/kumo/components/button';
import { Copy, Check, ArrowLeft, CaretLeft, CaretRight } from '@phosphor-icons/react';

export function RawAlertDetailPage() {
  const { metaId, alertId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');
  const [copied, setCopied] = useState(false);

  const withRunId = React.useCallback(
    (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path),
    [runId]
  );

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
  }, [prevId, nextId, metaId, withRunId, navigate]);

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <div className="px-6 pt-4 text-xs font-mono text-kumo-subtle flex items-center gap-2">
        <Link to={withRunId('/meta-alerts')} className="hover:underline text-kumo-default">MetaAlerts</Link>
        {metaId && (
          <>
            <span>/</span>
            <Link to={withRunId(`/meta-alerts/${metaId}`)} className="hover:underline text-kumo-default">#{metaId}</Link>
            <span>/</span>
            <Link to={withRunId(`/meta-alerts/${metaId}/raw-alerts`)} className="hover:underline text-kumo-default">Raw Alerts</Link>
          </>
        )}
        <span>/</span>
        <span className="truncate max-w-xs text-kumo-strong font-semibold">{alertId}</span>
      </div>

      <PageHeader
        title="Raw Alert Forensic Evidence"
        description={`Wazuh Alert ID: ${alertId}`}
        actions={
          <div className="flex items-center gap-2">
            {/* Previous / Next Navigation */}
            {metaId && sourceIds.length > 0 && (
              <div className="flex items-center gap-1 mr-2">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!prevId}
                  onClick={() => prevId && navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(prevId)}`))}
                  title="Previous Alert ([)"
                >
                  <CaretLeft size={14} /> Prev
                </Button>

                <span className="text-xs font-mono px-2 text-kumo-subtle">
                  {currentIndex >= 0 ? `${currentIndex + 1} of ${sourceIds.length}` : '—'}
                </span>

                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!nextId}
                  onClick={() => nextId && navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts/${encodeURIComponent(nextId)}`))}
                  title="Next Alert (])"
                >
                  Next <CaretRight size={14} />
                </Button>
              </div>
            )}

            {data && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
              >
                {copied ? <Check size={14} className="text-kumo-success" /> : <Copy size={14} />}
                {copied ? 'Copied JSON' : 'Copy Evidence JSON'}
              </Button>
            )}

            {metaId && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate(withRunId(`/meta-alerts/${metaId}/raw-alerts`))}
              >
                <ArrowLeft size={14} /> Back to Member List
              </Button>
            )}
          </div>
        }
      />

      <div className="px-6 py-4 space-y-4">
        {/* Unavailable Evidence Banner */}
        {isError && (
          <Banner
            variant="alert"
            size="sm"
            title="Local Evidence Unavailable"
            description={`Alert ID ${alertId} remains referenced by MetaAlert #${metaId} trace provenance, but its canonical audit record is not present in the local database.`}
          />
        )}

        {data && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Structured Fields */}
            <div className="space-y-6">
              {/* Identity & Ingestion */}
              <div>
                <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">Identity & Source</h3>
                <dl className="space-y-2 text-xs">
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Wazuh Alert ID:</dt> <dd className="font-mono font-semibold text-kumo-default">{data.wazuh_alert_id}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Timestamp (UTC):</dt> <dd className="font-mono text-kumo-default">{formatDateTime(data.timestamp)}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Source Mode:</dt> <dd className="font-mono text-kumo-default">{data.source_mode || 'LIVE'}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Agent:</dt> <dd className="text-kumo-default">{data.agent_name} ({data.agent_id}) · Crit: {data.agent_criticality}</dd></div>
                  {data.source_index && (
                    <div className="flex justify-between"><dt className="text-kumo-subtle">Source Index:</dt> <dd className="font-mono text-kumo-default">{data.source_index}</dd></div>
                  )}
                  {data.source_document_id && (
                    <div className="flex justify-between"><dt className="text-kumo-subtle">Document ID:</dt> <dd className="font-mono text-kumo-default">{data.source_document_id}</dd></div>
                  )}
                </dl>
              </div>

              {/* Rule Details */}
              <div>
                <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">Rule & Detection</h3>
                <dl className="space-y-2 text-xs">
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Rule ID:</dt> <dd className="font-mono font-semibold text-kumo-default">{data.rule_id}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Severity Level:</dt> <dd className="font-mono font-semibold text-kumo-default">{data.rule_level}/15</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Primary Group:</dt> <dd className="font-mono text-kumo-default">{data.rule_group_primary}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Rule Groups:</dt> <dd className="font-mono text-kumo-default">{data.rule_groups_all.join(', ') || data.rule_group_primary}</dd></div>
                  <div className="pt-1"><dt className="text-kumo-subtle">Description:</dt> <dd className="mt-1 p-2 rounded-md border border-kumo-hairline bg-kumo-recessed text-xs text-kumo-default">{data.rule_description}</dd></div>
                </dl>
              </div>

              {/* Network & MITRE */}
              <div>
                <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">Network & MITRE ATT&CK</h3>
                <dl className="space-y-2 text-xs">
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Source IP:</dt> <dd className="font-mono text-kumo-default">{data.srcip || '—'}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Location:</dt> <dd className="font-mono text-kumo-default">{data.location || '—'}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">Decoder:</dt> <dd className="font-mono text-kumo-default">{data.decoder || '—'}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">MITRE Tactics:</dt> <dd className="font-mono text-kumo-default">{data.mitre_tactics.join(', ') || 'None'}</dd></div>
                  <div className="flex justify-between"><dt className="text-kumo-subtle">MITRE Techniques:</dt> <dd className="font-mono text-kumo-default">{data.mitre_techniques.join(', ') || 'None'}</dd></div>
                </dl>
              </div>
            </div>

            {/* Right: Full Log & JSON Profile */}
            <div className="space-y-6">
              {data.full_log && (
                <div>
                  <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default mb-3 pb-2 border-b border-kumo-hairline">Full Log Message</h3>
                  <pre className="text-xs font-mono p-3 rounded-md border border-kumo-hairline bg-kumo-recessed text-kumo-default overflow-auto max-h-48 whitespace-pre-wrap">
                    {data.full_log}
                  </pre>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-3 pb-2 border-b border-kumo-hairline">
                  <h3 className="font-semibold text-xs uppercase tracking-wider text-kumo-default">Canonical & Evidence JSON</h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-sm border border-kumo-hairline bg-kumo-recessed text-kumo-subtle">READ ONLY</span>
                </div>
                <pre className="text-xs font-mono p-3 rounded-md border border-kumo-hairline bg-kumo-recessed text-kumo-default overflow-auto max-h-[500px]">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
