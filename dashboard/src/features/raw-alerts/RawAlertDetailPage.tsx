import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { fetchRawAlert } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/utils';
import { useState } from 'react';
import { Copy, Check, ArrowLeft, Shield, Terminal, Globe, Tag } from 'lucide-react';

export function RawAlertDetailPage() {
  const { metaId, alertId } = useParams();
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);

  const { data } = useQuery({
    queryKey: ['raw-alert', alertId],
    queryFn: () => fetchRawAlert(alertId as string),
    enabled: Boolean(alertId),
  });

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!data) {
    return <div className="p-6 text-sm" style={{ color: 'var(--text-tertiary)' }}>Loading forensic alert evidence...</div>;
  }

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
        <Link to="/meta-alerts" className="hover:underline">MetaAlerts</Link>
        {metaId && (
          <>
            <span>/</span>
            <Link to={`/meta-alerts/${metaId}`} className="hover:underline">#{metaId}</Link>
            <span>/</span>
            <Link to={`/meta-alerts/${metaId}/raw-alerts`} className="hover:underline">Raw Alerts</Link>
          </>
        )}
        <span>/</span>
        <span className="truncate max-w-xs" style={{ color: 'var(--text-primary)' }}>{data.wazuh_alert_id}</span>
      </div>

      <PageHeader
        title="Raw Alert Forensic Evidence"
        description={`Wazuh Alert ID: ${data.wazuh_alert_id}`}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium bg-white cursor-pointer"
              style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
            >
              {copied ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
              {copied ? 'Copied JSON' : 'Copy Evidence JSON'}
            </button>
            {metaId && (
              <button
                onClick={() => navigate(`/meta-alerts/${metaId}/raw-alerts`)}
                className="flex items-center gap-1.5 px-3 py-1.5 border rounded-[5px] text-xs font-medium bg-white cursor-pointer"
                style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
              >
                <ArrowLeft size={14} /> Back to Member List
              </button>
            )}
          </div>
        }
      />

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
              <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Timestamp:</dt> <dd className="font-mono">{formatDateTime(data.timestamp)}</dd></div>
              <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Source Mode:</dt> <dd className="font-mono">{data.source_mode || 'LIVE'}</dd></div>
              <div className="flex justify-between"><dt style={{ color: 'var(--text-tertiary)' }}>Agent:</dt> <dd>{data.agent_name} ({data.agent_id}) · Crit: {data.agent_criticality}</dd></div>
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
    </div>
  );
}
