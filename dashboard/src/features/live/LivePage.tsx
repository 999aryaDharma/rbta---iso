import { PageHeader } from '@/components/shared/PageHeader';
import { LiveStatusCard } from './LiveStatusPanel';
import { ActiveBucketsTimelineCard } from './ActiveBucketsTimeline';

export function LivePage() {
  return (
    <>
      <PageHeader
        breadcrumbs={['Operasi', 'Live Stream']}
        title="Live Stream Wazuh"
        description="Arus alert langsung dari Indexer/API — status worker, keterbasian siklus, karantina, dan lag"
      />
      <div className="px-6 py-8 lg:px-10 space-y-8">
        <LiveStatusCard />
        <ActiveBucketsTimelineCard />
      </div>
    </>
  );
}
