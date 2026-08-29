import { useQuery } from '@tanstack/react-query';
import { fetchMetaAlerts } from '@/api/metaAlerts';
import { PageHeader } from '@/components/shared/PageHeader';
import { DecisionBadge } from '@/components/shared/DecisionBadge';
import { formatDateTime, formatScore } from '@/lib/formatters';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Input } from '@cloudflare/kumo/components/input';
import { Button } from '@cloudflare/kumo/components/button';
import { Table } from '@cloudflare/kumo/components/table';
import { MagnifyingGlass, CaretLeft, CaretRight } from '@phosphor-icons/react';

export function MetaAlertsPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Number(searchParams.get('page') || 1);
  const decision = searchParams.get('decision') || '';
  const action = searchParams.get('action') || '';
  const search = searchParams.get('search') || '';
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const { data } = useQuery({
    queryKey: ['meta-alerts', page, decision, action, search, runId || 'live'],
    queryFn: () =>
      fetchMetaAlerts({
        page,
        page_size: 20,
        decision: decision || undefined,
        action: action || undefined,
        search: search || undefined,
        sort_by: 'end_time',
        sort_order: 'desc',
        run_id: runId || undefined,
      }),
  });

  const handleDecisionChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('decision', val);
    else params.delete('decision');
    params.set('page', '1');
    setSearchParams(params);
  };

  const handleActionChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('action', val);
    else params.delete('action');
    params.set('page', '1');
    setSearchParams(params);
  };

  const handleSearchChange = (val: string) => {
    const params = new URLSearchParams(searchParams);
    if (val) params.set('search', val);
    else params.delete('search');
    params.set('page', '1');
    setSearchParams(params);
  };

  const totalPages = data ? Math.ceil(data.total / 20) || 1 : 1;

  return (
    <div className="space-y-6">
      <PageHeader
        title="MetaAlerts Investigation Table"
        description="Aggregated security alert clusters scored by Isolation Forest with deterministic Tukey IQR thresholding"
      />

      {/* Toolbar / Filters */}
      <div className="p-4 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
          <div className="relative flex-1 max-w-sm">
            <MagnifyingGlass size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-kumo-subtle z-10" />
            <Input
              type="text"
              placeholder="Search Meta ID, Agent, Rule Group..."
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="w-full pl-8 text-xs"
            />
          </div>

          <select
            value={action}
            onChange={(e) => handleActionChange(e.target.value)}
            className="px-3 py-1.5 border border-kumo-hairline rounded-md text-xs font-mono bg-kumo-base text-kumo-default"
          >
            <option value="">All Actions</option>
            <option value="ESCALATE">ESCALATE</option>
            <option value="DAILY_DIGEST">DAILY_DIGEST</option>
            <option value="SUPPRESS">SUPPRESS</option>
          </select>

          <select
            value={decision}
            onChange={(e) => handleDecisionChange(e.target.value)}
            className="px-3 py-1.5 border border-kumo-hairline rounded-md text-xs font-mono bg-kumo-base text-kumo-default"
          >
            <option value="">All Decisions</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="SUSPICIOUS">SUSPICIOUS</option>
            <option value="CONTEXTUAL_ANOMALY">CONTEXTUAL_ANOMALY</option>
            <option value="NOISE_HIGH">NOISE_HIGH</option>
            <option value="NOISE">NOISE</option>
          </select>
        </div>

        <div className="text-xs font-mono text-kumo-subtle">
          {data ? `Showing page ${page} of ${totalPages} (${data.total} total)` : 'Loading...'}
        </div>
      </div>

      {/* MetaAlerts Table */}
      <div className="rounded-lg border border-kumo-hairline bg-kumo-base overflow-hidden shadow-xs">
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
