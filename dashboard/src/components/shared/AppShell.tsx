import { type ReactNode } from 'react';
import { SidebarProvider } from '@cloudflare/kumo/components/sidebar';
import { AppSidebar } from './Sidebar';
import { Topbar } from './Topbar';

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-kumo-canvas text-kumo-default antialiased">
      {/* Full-width Topbar spanning edge to edge */}
      <Topbar />

      {/* Main layout below topbar: Sidebar + Page content */}
      <SidebarProvider defaultOpen>
        <div className="flex flex-1 min-h-[calc(100vh-3.5rem)] overflow-hidden">
          <AppSidebar />
          <main className="flex-1 overflow-y-auto bg-kumo-base/30 min-w-0">
            {children}
          </main>
        </div>
      </SidebarProvider>
    </div>
  );
}
