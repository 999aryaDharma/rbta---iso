import { useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime, formatScore } from '@/lib/formatters';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useCallback, useEffect, useState } from 'react';
import { InputGroup } from '@cloudflare/kumo/components/input-group';
import { Select } from '@cloudflare/kumo/components/select';
import { Table } from '@cloudflare/kumo/components/table';
import { Pagination } from '@cloudflare/kumo/components/pagination';
import { MagnifyingGlass } from '@phosphor-icons/react';

export function MetaAlertsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const decision = searchParams.get('decision') || '';
  const action = searchParams.get('action') || '';
  const urlSearch = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');
  const [localSearch, setLocalSearch] = useState(urlSearch);

  const withRunId = useCallback(
    (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path),
    [runId]
  );

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
    queryKey: ['meta-alerts', page, decision, action, urlSearch, runId || 'live'],
    queryFn: () =>
      fetchMetaAlerts({
        page,
        page_size: 20,
        decision: decision || undefined,
        action: action || undefined,
        search: urlSearch || undefined,
        sort_by: 'end_time',
        sort_order: 'desc',
        run_id: runId || undefined,
      }),
    placeholderData: keepPreviousData,
    staleTime: 5000,
  });

  // Prefetch next page
  useEffect(() => {
    const totalPages = data ? Math.ceil(data.total / 20) || 1 : 1;
    if (data && page < totalPages) {
      queryClient.prefetchQuery({
        queryKey: ['meta-alerts', page + 1, decision, action, urlSearch, runId || 'live'],
        queryFn: () =>
          fetchMetaAlerts({
            page: page + 1,
            page_size: 20,
            decision: decision || undefined,
            action: action || undefined,
            search: urlSearch || undefined,
            sort_by: 'end_time',
            sort_order: 'desc',
            run_id: runId || undefined,
          }),
        staleTime: 5000,
      });
    }
  }, [data, page, decision, action, urlSearch, runId, queryClient]);

  const setFilterParam = useCallback((key: string, val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set(key, val);
    else params.delete(key);
    params.set('page', '1');
    setSearchParams(params);
  }, [searchParams, setSearchParams]);

  return (
    <>
      <PageHeader
        breadcrumbs={['Security Analytics', 'MetaAlerts']}
        title="MetaAlerts Investigation Explorer"
        description="Aggregated security alert clusters scored by Isolation Forest with deterministic Tukey IQR thresholding"
      />

      <div className="px-6 py-8 lg:px-10 space-y-6">
        {/* Filter toolbar card */}
        <div className="p-5 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
            <div className="w-full sm:w-80">
              <InputGroup>
                <InputGroup.Addon align="start"><MagnifyingGlass size={14} className="text-kumo-subtle" /></InputGroup.Addon>
                <InputGroup.Input
                  placeholder="Filter Meta ID, Agent, Rule Group..."
                  value={localSearch}
                  onChange={(e) => setLocalSearch(e.target.value)}
                />
              </InputGroup>
            </div>

            <Select
              size="sm"
              value={action || ''}
              onValueChange={(val) => setFilterParam('action', val as string)}
              placeholder="All Actions"
            >
              <Select.Option value="">All Actions</Select.Option>
              <Select.Option value="ESCALATE">ESCALATE</Select.Option>
              <Select.Option value="DAILY_DIGEST">DAILY_DIGEST</Select.Option>
              <Select.Option value="SUPPRESS">SUPPRESS</Select.Option>
            </Select>

            <Select
              size="sm"
              value={decision || ''}
              onValueChange={(val) => setFilterParam('decision', val as string)}
              placeholder="All Decisions"
            >
              <Select.Option value="">All Decisions</Select.Option>
              <Select.Option value="CRITICAL">CRITICAL</Select.Option>
              <Select.Option value="SUSPICIOUS">SUSPICIOUS</Select.Option>
              <Select.Option value="CONTEXTUAL_ANOMALY">CONTEXTUAL_ANOMALY</Select.Option>
              <Select.Option value="NOISE_HIGH">NOISE_HIGH</Select.Option>
              <Select.Option value="NOISE">NOISE</Select.Option>
            </Select>
          </div>

          <div className="flex items-center gap-2 text-xs text-kumo-subtle">
            {isFetching && <span className="animate-pulse">Refreshing...</span>}
            <span className="font-mono bg-kumo-recessed px-2.5 py-1 rounded border border-kumo-hairline text-kumo-strong font-medium">
              Total: {data?.total ?? 0}
            </span>
          </div>
        </div>

        {/* Table container */}
        <div className={`rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs overflow-hidden transition-opacity ${isPlaceholderData ? 'opacity-70' : ''}`}>
          <Table>
            <Table.Header>
              <Table.Row className="bg-kumo-recessed/50 text-[11px] uppercase tracking-wider">
                <Table.Head>Meta ID</Table.Head>
                <Table.Head>End Time</Table.Head>
                <Table.Head>Agent / Host</Table.Head>
                <Table.Head>Rule Name</Table.Head>
                <Table.Head className="text-right">Alert Count</Table.Head>
                <Table.Head className="text-right">Max Sev</Table.Head>
                <Table.Head className="text-right">Anomaly Score</Table.Head>
                <Table.Head>SOC Decision</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {data?.items.map((m) => (
                <Table.Row
                  key={m.meta_id}
                  className="cursor-pointer hover:bg-kumo-recessed/40 transition-colors text-xs"
                  onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                >
                  <Table.Cell className="font-mono font-semibold text-kumo-strong">#{m.meta_id}</Table.Cell>
                  <Table.Cell className="text-kumo-subtle">{formatDateTime(m.end_time)}</Table.Cell>
                  <Table.Cell className="font-medium text-kumo-default">{m.agent_name} <span className="text-kumo-subtle text-[11px]">({m.agent_id})</span></Table.Cell>
                  <Table.Cell className="font-mono text-kumo-default">{m.rule_group_primary}</Table.Cell>
                  <Table.Cell className="text-right font-mono font-bold text-kumo-strong">{m.alert_count}</Table.Cell>
                  <Table.Cell className="text-right font-mono text-kumo-default">{m.max_severity}/15</Table.Cell>
                  <Table.Cell className="text-right font-mono text-kumo-default">{formatScore(m.anomaly_score)}</Table.Cell>
                  <Table.Cell><DecisionBadge decision={m.decision} action={m.action} /></Table.Cell>
                </Table.Row>
              ))}
              {data && data.items.length === 0 && (
                <Table.Row>
                  <Table.Cell colSpan={8} className="py-12 text-center text-xs text-kumo-subtle font-mono">
                    No MetaAlerts match the specified filter criteria.
                  </Table.Cell>
                </Table.Row>
              )}
            </Table.Body>
          </Table>

          {/* Pagination */}
          <div className="px-6 py-4 border-t border-kumo-hairline bg-kumo-recessed/20">
            <Pagination
              page={page}
              setPage={(p) => {
                const params = new URLSearchParams(searchParams);
                params.set('page', String(p));
                setSearchParams(params);
              }}
              perPage={20}
              totalCount={data?.total ?? 0}
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
