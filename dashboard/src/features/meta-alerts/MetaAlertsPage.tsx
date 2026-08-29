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
        title="MetaAlerts Investigation Table"
        description="Aggregated security alert clusters scored by Isolation Forest with deterministic Tukey IQR thresholding"
      />

      <div className="px-6 py-4 space-y-0">
        {/* Filter toolbar */}
        <div className="flex flex-wrap items-center gap-3 pb-3 border-b border-kumo-hairline">
          <div className="flex-1 min-w-[200px] max-w-sm">
            <InputGroup>
              <InputGroup.Addon align="start"><MagnifyingGlass size={14} /></InputGroup.Addon>
              <InputGroup.Input
                placeholder="Search Meta ID, Agent, Rule Group..."
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

          {isFetching && <span className="text-xs text-kumo-subtle">Refreshing...</span>}
        </div>

        {/* Table */}
        <div className={`rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden transition-opacity ${isPlaceholderData ? 'opacity-70' : ''}`}>
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Meta ID</Table.Head>
                <Table.Head>End Time</Table.Head>
                <Table.Head>Agent</Table.Head>
                <Table.Head>Primary Group</Table.Head>
                <Table.Head className="text-right">Raw Count</Table.Head>
                <Table.Head className="text-right">Max Sev</Table.Head>
                <Table.Head className="text-right">Score</Table.Head>
                <Table.Head>Decision</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {data?.items.map((m) => (
                <Table.Row
                  key={m.meta_id}
                  className="cursor-pointer hover:bg-kumo-tint"
                  onClick={() => navigate(withRunId(`/meta-alerts/${m.meta_id}`))}
                >
                  <Table.Cell className="font-mono text-xs font-semibold">#{m.meta_id}</Table.Cell>
                  <Table.Cell className="text-xs">{formatDateTime(m.end_time)}</Table.Cell>
                  <Table.Cell className="text-xs">{m.agent_name} ({m.agent_id})</Table.Cell>
                  <Table.Cell className="text-xs font-mono">{m.rule_group_primary}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono font-semibold">{m.alert_count}</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{m.max_severity}/15</Table.Cell>
                  <Table.Cell className="text-xs text-right font-mono">{formatScore(m.anomaly_score)}</Table.Cell>
                  <Table.Cell><DecisionBadge decision={m.decision} action={m.action} /></Table.Cell>
                </Table.Row>
              ))}
              {data && data.items.length === 0 && (
                <Table.Row>
                  <Table.Cell colSpan={8} className="p-8 text-center text-xs text-kumo-subtle">
                    No MetaAlerts match the specified filters.
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
