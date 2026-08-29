import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlertRawAlerts } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/formatters';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Button } from '@cloudflare/kumo/components/button';
import { Input } from '@cloudflare/kumo/components/input';
import { Table } from '@cloudflare/kumo/components/table';
import { MagnifyingGlass, CaretLeft, CaretRight, ArrowLeft } from '@phosphor-icons/react';

export function RawAlertsPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const search = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');
  const id = Number(metaId);

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data } = useQuery({
    queryKey: ['raw-alerts', id, page, search, runId || 'live'],
    queryFn: () =>
      fetchMetaAlertRawAlerts(id, {
        page,
        page_size: 20,
        search: search || undefined,
        run_id: runId || undefined,
      }),
  });

  const handleSearchChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('search', val);
    else params.delete('search');
    params.set('page', '1');
    setSearchParams(params);
  };

  const totalPages = data ? Math.ceil(data.filtered_total / 20) || 1 : 1;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-xs font-mono text-kumo-subtle">
        <Link to={withRunId('/meta-alerts')} className="hover:underline text-kumo-default">MetaAlerts</Link>
        <span>/</span>
        <Link to={withRunId(`/meta-alerts/${id}`)} className="hover:underline text-kumo-default">#{id}</Link>
        <span>/</span>
        <span className="text-kumo-strong font-semibold">Raw Alerts</span>
      </div>

      <PageHeader
        title={`Member Raw Alerts for MetaAlert #${id}`}
        description="Individual security log events aggregated into this temporal cluster"
        actions={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate(withRunId(`/meta-alerts/${id}`))}
          >
            <ArrowLeft size={14} /> Back to MetaAlert
          </Button>
        }
      />

      {/* Unresolved Evidence Warning Banner */}
      {data && data.unresolved_alert_ids && data.unresolved_alert_ids.length > 0 && (
        <Banner
          variant="alert"
          size="sm"
          title={`Partial Local Evidence: ${data.resolved_total} of ${data.source_total} source alerts resolved.`}
          description={`${data.unresolved_alert_ids.length} source alert(s) remain referenced by the MetaAlert trace but local audit evidence is unavailable: ${data.unresolved_alert_ids.join(', ')}`}
        />
      )}

      {/* Filter Toolbar */}
      <div className="p-4 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <MagnifyingGlass size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-kumo-subtle z-10" />
          <Input
            type="text"
            placeholder="Search Alert ID, Rule ID, Description, IP..."
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-8 text-xs"
          />
        </div>

        <div className="text-xs font-mono text-kumo-subtle">
          {data ? `Showing page ${page} of ${totalPages} (${data.filtered_total} matching / ${data.source_total} total)` : 'Loading...'}
        </div>
      </div>

      {/* Raw Alerts Table */}
      <div className="rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden shadow-xs">
        <Table>
          <Table.Header>
            <Table.Row>
              <Table.Head>Timestamp</Table.Head>
              <Table.Head>Wazuh Alert ID</Table.Head>
              <Table.Head>Rule ID</Table.Head>
              <Table.Head className="text-right">Level</Table.Head>
              <Table.Head>Description</Table.Head>
              <Table.Head>Source IP</Table.Head>
              <Table.Head>MITRE Tactics</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {data?.items.map((a) => (
              <Table.Row
                key={a.wazuh_alert_id}
                className="cursor-pointer hover:bg-kumo-tint"
                onClick={() =>
                  navigate(withRunId(`/meta-alerts/${id}/raw-alerts/${encodeURIComponent(a.wazuh_alert_id)}`))
                }
              >
                <Table.Cell className="text-xs font-mono">{formatDateTime(a.timestamp)}</Table.Cell>
                <Table.Cell className="font-mono text-xs font-semibold text-kumo-link truncate max-w-[140px]">
                  {a.wazuh_alert_id}
                </Table.Cell>
                <Table.Cell className="font-mono text-xs">{a.rule_id}</Table.Cell>
                <Table.Cell className="text-xs text-right font-mono font-semibold">{a.rule_level}</Table.Cell>
                <Table.Cell className="text-xs truncate max-w-[220px]">{a.rule_description}</Table.Cell>
                <Table.Cell className="font-mono text-xs">{a.srcip || '—'}</Table.Cell>
                <Table.Cell className="text-xs">
                  {a.mitre_tactics && a.mitre_tactics.length > 0 ? (
                    <span className="px-1.5 py-0.5 rounded-sm text-[11px] font-mono border border-kumo-hairline bg-kumo-recessed text-kumo-default">
                      {a.mitre_tactics.join(', ')}
                    </span>
                  ) : (
                    <span className="text-kumo-inactive">None</span>
                  )}
                </Table.Cell>
              </Table.Row>
            ))}
            {data && data.items.length === 0 && (
              <Table.Row>
                <Table.Cell colSpan={7} className="p-6 text-center text-xs text-kumo-subtle">
                  No matching raw alerts found.
                </Table.Cell>
              </Table.Row>
            )}
          </Table.Body>
        </Table>

        {/* Pagination Bar */}
        <div className="p-4 flex items-center justify-between border-t border-kumo-hairline bg-kumo-base">
          <Button
            variant="ghost"
            size="sm"
            disabled={page <= 1}
            onClick={() => {
              const params = new URLSearchParams(searchParams);
              params.set('page', String(page - 1));
              setSearchParams(params);
            }}
          >
            <CaretLeft size={14} /> Previous
          </Button>

          <span className="text-xs font-mono text-kumo-subtle">
            Page {page} of {totalPages}
          </span>

          <Button
            variant="ghost"
            size="sm"
            disabled={!data || page >= totalPages}
            onClick={() => {
              const params = new URLSearchParams(searchParams);
              params.set('page', String(page + 1));
              setSearchParams(params);
            }}
          >
            Next <CaretRight size={14} />
          </Button>
        </div>
      </div>
    </div>
  );
}
