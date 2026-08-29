import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTelegramPayloads } from '@/api/replay';
import { Table } from '@cloudflare/kumo/components/table';
import { PaperPlaneRight, Copy, Check, ArrowClockwise } from '@phosphor-icons/react';

export function DeferredTelegramOutbox() {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['telegram-payloads'],
    queryFn: () => fetchTelegramPayloads(50),
    refetchInterval: 3000,
  });

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const payloads = data?.items || [];
  const totalCount = data?.total_count || 0;

  return (
    <div className="rounded-lg border border-kumo-hairline bg-kumo-base p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-kumo-hairline">
        <div className="flex items-center gap-2">
          <PaperPlaneRight size={14} className="text-red-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-kumo-subtle">
            Deferred Telegram Payload Outbox
          </span>
          <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400 border border-red-200 dark:border-red-900/40">
            {totalCount} ESCALATE Payloads
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-kumo-subtle">
            File: <code>telegram_escalate_payloads.txt</code>
          </span>
          <button
            type="button"
            onClick={() => refetch()}
            className="p-1 rounded text-kumo-subtle hover:text-kumo-default hover:bg-kumo-recessed"
            title="Refresh outbox"
          >
            <ArrowClockwise size={13} />
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="py-6 text-center text-xs text-kumo-subtle font-mono">
          Loading recorded escalation payloads...
        </div>
      ) : payloads.length > 0 ? (
        <div className="overflow-x-auto rounded border border-kumo-hairline">
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Timestamp</Table.Head>
                <Table.Head>Meta ID</Table.Head>
                <Table.Head>Decision</Table.Head>
                <Table.Head>Group & Endpoint</Table.Head>
                <Table.Head>Score / Threshold</Table.Head>
                <Table.Head>Telegram Message Payload</Table.Head>
                <Table.Head className="text-right">Action</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {payloads.map((p) => {
                const isCopied = copiedId === p.idempotency_key;
                return (
                  <Table.Row key={p.idempotency_key} className="font-mono text-xs">
                    <Table.Cell className="text-kumo-subtle text-[11px] whitespace-nowrap">
                      {p.timestamp.replace('T', ' ').substring(0, 19)}
                    </Table.Cell>
                    <Table.Cell className="font-semibold text-kumo-default">
                      <div>#{p.meta_id}</div>
                      <div className="text-[10px] text-kumo-subtle font-normal">{p.idempotency_key}</div>
                    </Table.Cell>
                    <Table.Cell>
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400 border border-red-200 dark:border-red-900/40">
                        {p.decision}
                      </span>
                    </Table.Cell>
                    <Table.Cell className="text-[11px]">
                      <div className="font-medium text-kumo-default">{p.rule_group_primary}</div>
                      <div className="text-kumo-subtle text-[10px]">{p.agent_name} ({p.agent_id})</div>
                    </Table.Cell>
                    <Table.Cell className="text-[11px]">
                      <span className="text-red-500 font-semibold">{p.anomaly_score.toFixed(4)}</span>
                      <span className="text-kumo-subtle"> &gt; {p.threshold.toFixed(4)}</span>
                    </Table.Cell>
                    <Table.Cell className="text-[11px] text-kumo-subtle max-w-sm truncate" title={p.message}>
                      {p.message}
                    </Table.Cell>
                    <Table.Cell className="text-right">
                      <button
                        type="button"
                        onClick={() => handleCopy(p.idempotency_key, JSON.stringify(p, null, 2))}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded bg-kumo-recessed hover:bg-kumo-canvas text-[11px] text-kumo-subtle hover:text-kumo-default border border-kumo-hairline"
                        title="Copy JSON Payload"
                      >
                        {isCopied ? <Check size={11} className="text-green-500" /> : <Copy size={11} />}
                        <span>{isCopied ? 'Copied' : 'JSON'}</span>
                      </button>
                    </Table.Cell>
                  </Table.Row>
                );
              })}
            </Table.Body>
          </Table>
        </div>
      ) : (
        <div className="py-6 text-center text-xs text-kumo-subtle font-mono italic">
          No ESCALATE payloads recorded in <code className="bg-kumo-recessed px-1 py-0.5 rounded">telegram_escalate_payloads.txt</code>.
          When an anomaly score strictly breaches the Tukey threshold and triggers an ESCALATE action, payload will appear here.
        </div>
      )}
    </div>
  );
}
