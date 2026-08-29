import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/shared/AppShell';
import { OverviewPage } from '@/features/overview/OverviewPage';
import { RBTAPage } from '@/features/rbta/RBTAPage';
import { MetaAlertsPage } from '@/features/meta-alerts/MetaAlertsPage';
import { MetaAlertDetailPage } from '@/features/meta-alerts/MetaAlertDetailPage';
import { RawAlertsPage } from '@/features/raw-alerts/RawAlertsPage';
import { RawAlertDetailPage } from '@/features/raw-alerts/RawAlertDetailPage';
import { ReplayPage } from '@/features/replay/ReplayPage';
import { IntegrationsPage } from '@/features/integrations/IntegrationsPage';
import { SystemPage } from '@/features/system/SystemPage';
import { AuthGate } from '@/components/shared/AuthGate';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 2000,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/dashboard">
        <AuthGate>
          <AppShell>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/rbta" element={<RBTAPage />} />
              <Route path="/meta-alerts" element={<MetaAlertsPage />} />
              <Route path="/meta-alerts/:metaId" element={<MetaAlertDetailPage />} />
              <Route path="/meta-alerts/:metaId/raw-alerts" element={<RawAlertsPage />} />
              <Route path="/meta-alerts/:metaId/raw-alerts/:alertId" element={<RawAlertDetailPage />} />
              <Route path="/replay" element={<ReplayPage />} />
              <Route path="/integrations" element={<IntegrationsPage />} />
              <Route path="/system" element={<SystemPage />} />
            </Routes>
          </AppShell>
        </AuthGate>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
