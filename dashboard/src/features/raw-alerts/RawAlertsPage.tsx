import { useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { fetchMetaAlertRawAlerts } from '@/api/rawAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime } from '@/lib/formatters';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Button } from '@cloudflare/kumo/components/button';
import { InputGroup } from '@cloudflare/kumo/components/input-group';
import { Table } from '@cloudflare/kumo/components/table';
import { Pagination } from '@cloudflare/kumo/components/pagination';
import { MagnifyingGlass, ArrowLeft } from '@phosphor-icons/react';
import { useEffect, useState } from 'react';

export function RawAlertsPage() {
  const { metaId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const urlSearch = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');
  const id = Number(metaId);
  const [localSearch, setLocalSearch] = useState(urlSearch);

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  // Sync local search from URL
  useEffect(() => { setLocalSearch(urlSearch); }, [urlSearch]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (localSearch !== urlSearch) {
        const params = new URLSearchParams(searchParams);
        if (localSearch) params.set('search', localSearch);
        else params.delete('search');
        params.set('page', '1');
        setSearchParams(params);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [localSearch, urlSearch, searchParams, setSearchParams]);

  const { data, isFetching, isPlaceholderData } = useQuery({
    queryKey: ['raw-alerts', id, page, urlSearch, runId || 'live'],
    queryFn: () =>
      fetchMetaAlertRawAlerts(id, {
        page,
        page_size: 20,
        search: urlSearch || undefined,
        run_id: runId || undefined,
      }),
    placeholderData: keepPreviousData,
    staleTime: 5000,
  });

  // Prefetch next page
  useEffect(() => {
    const totalPages = data ? Math.ceil(data.filtered_total / 20) || 1 : 1;
    if (data && page < totalPages) {
      queryClient.prefetchQuery({
        queryKey: ['raw-alerts', id, page + 1, urlSearch, runId || 'live'],
        queryFn: () =>
          fetchMetaAlertRawAlerts(id, {
            page: page + 1,
            page_size: 20,
            search: urlSearch || undefined,
            run_id: runId || undefined,
          }),
        staleTime: 5000,
      });
    }
  }, [data, page, urlSearch, runId, queryClient, id]);

  return (
    <>
      <div className="px-6 pt-4 text-xs font-mono text-kumo-subtle flex items-center gap-2">
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

      <div className="px-6 py-4 space-y-4">
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
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-kumo-hairline">
          <div className="relative flex-1 max-w-sm">
            <InputGroup>
              <InputGroup.Addon align="start"><MagnifyingGlass size={14} /></InputGroup.Addon>
              <InputGroup.Input
                placeholder="Search Alert ID, Rule ID, Description, IP..."
                value={localSearch}
                onChange={(e) => setLocalSearch(e.target.value)}
              />
            </InputGroup>
          </div>

          <div className="flex items-center gap-2">
            {isFetching && <span className="text-xs text-kumo-subtle">Refreshing...</span>}
            <div className="text-xs font-mono text-kumo-subtle">
              {data ? `Showing ${data.filtered_total} matching / ${data.source_total} total` : 'Loading...'}
            </div>
          </div>
        </div>

        {/* Raw Alerts Table */}
        <div className={`rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden transition-opacity ${isPlaceholderData ? 'opacity-70' : ''}`}>
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

          {/* Pagination */}
          <div className="px-4 py-2 border-t border-kumo-hairline">
            <Pagination
              page={page}
              setPage={(p) => {
                const params = new URLSearchParams(searchParams);
                params.set('page', String(p));
                setSearchParams(params);
              }}
              perPage={20}
              totalCount={data?.filtered_total ?? 0}
            >
              <Pagination.Info />
              <Pagination.Separator />
              <Pagination.Controls />
            </Pagination>
          </div>
        </div>
      </div>
    </>
  );
}
