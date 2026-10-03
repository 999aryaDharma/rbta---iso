import { PageHeader } from '@/components/shared/PageHeader';
import { LiveTriageSummaryCard } from './LiveTriageSummary';
import { LiveTransportDetails } from './LiveTransportDetails';
import { ActiveBucketsTimelineCard } from './ActiveBucketsTimeline';

export function LivePage() {
  return (
    <>
      <PageHeader
        breadcrumbs={['Operasi', 'Live Stream']}
        title="Live Stream Wazuh"
        description="Beban triase analis dulu: reduksi, antrean ESCALATE, distribusi decision, dan kesegaran data. Detail transport di bawah timeline"
      />
      <div className="px-6 py-8 lg:px-10 space-y-8">
        <LiveTriageSummaryCard />
        <ActiveBucketsTimelineCard />
        <LiveTransportDetails />
      </div>
    </>
  );
}
