import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { fetchBuckets, fetchBucketRawAlerts } from '@/api/dashboard';
import { ApiError } from '@/api/client';
import { PageHeader } from '@/components/shared/PageHeader';
import { formatDateTime, formatNumber } from '@/lib/formatters';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Button } from '@cloudflare/kumo/components/button';
import { Badge } from '@cloudflare/kumo/components/badge';
import { Table } from '@cloudflare/kumo/components/table';
import { Pagination } from '@cloudflare/kumo/components/pagination';
import { ArrowLeft } from '@phosphor-icons/react';

const BUCKET_PAGE_SIZE = 20;

function ruleDescription(description: string, ruleId: string) {
  return description.trim() || `Rule ${ruleId}`;
}

function finalizedHint(message: string): number | null {
  try {
    const parsed = JSON.parse(message) as { detail?: unknown; finalized_meta_id?: unknown };
    return typeof parsed.finalized_meta_id === 'number' ? parsed.finalized_meta_id : null;
  } catch {
    return null;
  }
}

export function BucketDetailPage() {
  const { agentId = '', ruleGroup = '' } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Math.max(1, Number(searchParams.get('page') || 1));
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) =>
    runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path;

  const bucketKey = `${agentId} · ${ruleGroup}`;

  const bucketsQuery = useQuery({
    queryKey: ['buckets', runId || 'live'],
    queryFn: () => fetchBuckets(runId || undefined),
    staleTime: 5000,
  });
  const bucket = bucketsQuery.data?.find(
    (b) => b.agent_id === agentId && b.rule_group_primary === ruleGroup,
  );

  const membersQuery = useQuery({
    queryKey: ['bucket-raw-alerts', agentId, ruleGroup, page, runId || 'live'],
    queryFn: () =>
      fetchBucketRawAlerts(agentId, ruleGroup, {
        page,
        page_size: BUCKET_PAGE_SIZE,
        run_id: runId || undefined,
      }),
    placeholderData: keepPreviousData,
    staleTime: 5000,
    retry: false,
  });

  const goneFinalizedId =
    membersQuery.error instanceof ApiError && membersQuery.error.status === 404
      ? finalizedHint(membersQuery.error.message)
      : null;

  const setPage = (next: number) => {
    const params = new URLSearchParams(searchParams);
    params.set('page', String(next));
    setSearchParams(params);
  };

  return (
    <>
      <PageHeader
        breadcrumbs={['Operasi', 'Live Stream', bucketKey]}
        title={`Bucket ${bucketKey}`}
        description="Raw alert anggota bucket yang masih mengagregasi. Isi berubah saat alert baru masuk sampai bucket final."
        actions={
          <Button variant="ghost" size="sm" onClick={() => navigate(withRunId('/live'))}>
            <ArrowLeft size={14} className="mr-1" /> Kembali ke Live
          </Button>
        }
      />

      <div className="px-6 py-8 lg:px-10 space-y-6">
        {bucket && (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-kumo-hairline bg-kumo-canvas px-6 py-4 text-xs shadow-xs">
            <span className="inline-flex flex-col gap-1">
              <Badge variant="secondary">Mengagregasi</Badge>
              {bucket.meta_id != null && (
                <span className="font-mono text-[11px] text-kumo-subtle">calon #{bucket.meta_id}</span>
              )}
            </span>
            <div>
              <p className="font-mono font-bold text-kumo-strong text-lg leading-none">
                {formatNumber(bucket.alert_count)}
              </p>
              <p className="text-kumo-subtle mt-1">Alert anggota</p>
            </div>
            <div>
              <p className="font-mono text-kumo-default">{formatDateTime(bucket.start_time)}</p>
              <p className="text-kumo-subtle mt-1">Window dibuka</p>
            </div>
            <div>
              <p className="font-mono font-semibold text-kumo-default">
                {bucket.max_severity} / 15
              </p>
              <p className="text-kumo-subtle mt-1">Level max</p>
            </div>
            {bucket.agent_name && (
              <div>
                <p className="font-mono text-kumo-default">{bucket.agent_name}</p>
                <p className="text-kumo-subtle mt-1">Nama agent</p>
              </div>
            )}
          </div>
        )}

        {membersQuery.isError && (
          <Banner
            variant="alert"
            size="sm"
            title={
              goneFinalizedId != null
                ? `Bucket sudah final sebagai MetaAlert #${goneFinalizedId}.`
                : 'Bucket tidak aktif.'
            }
            description={
              goneFinalizedId != null
                ? 'Anggota bucket kini menjadi bukti MetaAlert final; buka halaman detailnya.'
                : 'Bucket sudah final atau belum pernah ada. Kembali ke Live untuk bucket aktif.'
            }
          />
        )}
        {goneFinalizedId != null && (
          <div>
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate(withRunId(`/meta-alerts/${goneFinalizedId}`))}
            >
              Buka MetaAlert #{goneFinalizedId}
            </Button>
          </div>
        )}

        {membersQuery.data && membersQuery.data.unresolved_alert_ids.length > 0 && (
          <Banner
            variant="alert"
            size="sm"
            title={`Partial Local Evidence: ${membersQuery.data.resolved_total} of ${membersQuery.data.source_total} source alerts resolved.`}
            description={`${membersQuery.data.unresolved_alert_ids.length} source alert(s) belum tersedia di bukti lokal: ${membersQuery.data.unresolved_alert_ids.join(', ')}`}
          />
        )}

        <div className="rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs overflow-hidden">
          <div className="px-6 py-4 border-b border-kumo-hairline flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-kumo-strong">
                Anggota bucket ({membersQuery.data ? formatNumber(membersQuery.data.filtered_total) : '…'})
              </h2>
              <p className="text-xs text-kumo-subtle mt-0.5">
                Urutan kedatangan. Baris bukan bukti serangan.
              </p>
            </div>
            {membersQuery.isFetching && (
              <p role="status" className="text-xs text-kumo-subtle font-mono">Refreshing…</p>
            )}
          </div>
          <Table>
            <Table.Header>
              <Table.Row className="bg-kumo-recessed/50 text-[11px] uppercase tracking-wider">
                <Table.Head>Timestamp</Table.Head>
                <Table.Head>Wazuh Alert ID</Table.Head>
                <Table.Head>Detection Signature</Table.Head>
                <Table.Head className="text-right">Level</Table.Head>
                <Table.Head>Rule Group</Table.Head>
                <Table.Head>Source IP</Table.Head>
                <Table.Head>MITRE Tactics</Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {(membersQuery.data?.items ?? []).map((a) => (
                <Table.Row
                  key={a.wazuh_alert_id}
                  onClick={() =>
                    navigate(
                      withRunId(
                        `/live/buckets/${encodeURIComponent(agentId)}/${encodeURIComponent(ruleGroup)}/raw-alerts/${encodeURIComponent(a.wazuh_alert_id)}`,
                      ),
                    )
                  }
                  className="hover:bg-kumo-recessed/40 transition-colors text-xs cursor-pointer"
                >
                  <Table.Cell className="font-mono text-kumo-subtle">{formatDateTime(a.timestamp)}</Table.Cell>
                  <Table.Cell className="font-mono text-kumo-default truncate max-w-[140px]">
                    {a.wazuh_alert_id}
                  </Table.Cell>
                  <Table.Cell>
                    <span className="font-medium text-kumo-strong">
                      {ruleDescription(a.rule_description, a.rule_id)}
                    </span>
                    <span className="block font-mono text-[11px] text-kumo-subtle">Rule ID: {a.rule_id}</span>
                  </Table.Cell>
                  <Table.Cell className="text-right font-mono font-bold text-kumo-strong">
                    {a.rule_level}
                  </Table.Cell>
                  <Table.Cell className="font-mono text-kumo-default">{a.rule_group_primary}</Table.Cell>
                  <Table.Cell className="font-mono text-kumo-subtle">{a.srcip || '-'}</Table.Cell>
                  <Table.Cell>
                    {a.mitre_tactics && a.mitre_tactics.length > 0 ? (
                      <Badge variant="secondary">{a.mitre_tactics.join(', ')}</Badge>
                    ) : (
                      <span className="text-kumo-subtle">None</span>
                    )}
                  </Table.Cell>
                </Table.Row>
              ))}
              {membersQuery.data && membersQuery.data.items.length === 0 && (
                <Table.Row>
                  <Table.Cell colSpan={7} className="py-12 text-center text-xs text-kumo-subtle font-mono">
                    Belum ada anggota bucket yang dapat ditampilkan.
                  </Table.Cell>
                </Table.Row>
              )}
            </Table.Body>
          </Table>

          {membersQuery.data && membersQuery.data.filtered_total > BUCKET_PAGE_SIZE && (
            <div className="px-6 py-4 border-t border-kumo-hairline bg-kumo-recessed/20">
              <Pagination
                page={page}
                setPage={setPage}
                perPage={BUCKET_PAGE_SIZE}
                totalCount={membersQuery.data.filtered_total}
              >
                <Pagination.Info />
                <Pagination.Separator />
                <Pagination.Controls />
              </Pagination>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
