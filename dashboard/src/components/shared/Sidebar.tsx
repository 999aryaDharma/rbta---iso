import { useCallback, useEffect } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Sidebar, SidebarHeader, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupLabel,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarMenuSub, SidebarMenuSubButton,
  SidebarTrigger, SidebarRail,
} from '@cloudflare/kumo/components/sidebar';
import { Badge } from '@cloudflare/kumo/components/badge';
import {
  House, ChartBar, Cpu, Play, Plugs, GearSix,
  MagnifyingGlass, CaretUpDown, ShieldCheck,
} from '@phosphor-icons/react';

export function AppSidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const withRunId = useCallback(
    (path: string) =>
      runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path,
    [runId]
  );

  useEffect(() => {
    let lastKey = '';
    let lastKeyTime = 0;

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) {
        return;
      }

      const now = Date.now();
      if (lastKey === 'g' && now - lastKeyTime < 1000) {
        const map: Record<string, string> = {
          o: '/overview', m: '/meta-alerts', r: '/rbta', p: '/replay', s: '/system',
        };
        if (map[e.key]) {
          e.preventDefault();
          navigate(withRunId(map[e.key]));
        }
        lastKey = '';
        return;
      }

      if (e.key === 'g') {
        lastKey = 'g';
        lastKeyTime = now;
      } else {
        lastKey = '';
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [withRunId, navigate]);

  const openSearch = () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
  };

  const isPathActive = (path: string) => {
    if (path === '/overview') return location.pathname === '/overview' || location.pathname === '/';
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <Sidebar>
      {/* 1. Header with Company/Platform switcher */}
      <SidebarHeader className="p-3 border-b border-kumo-line flex items-center justify-between">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="size-6 rounded-md bg-kumo-brand/10 border border-kumo-brand/20 text-kumo-brand flex items-center justify-center shrink-0">
            <ShieldCheck size={16} weight="duotone" />
          </div>
          <span className="font-semibold text-sm text-kumo-strong truncate group-data-[state=collapsed]/sidebar:hidden">
            RBTA Platform
          </span>
        </div>
        <CaretUpDown size={14} className="text-kumo-subtle shrink-0 group-data-[state=collapsed]/sidebar:hidden" />
      </SidebarHeader>

      {/* 2. Scrollable Navigation Content */}
      <SidebarContent>
        {/* Quick Search */}
        <div className="px-2 pt-2 pb-1 group-data-[state=collapsed]/sidebar:px-1">
          <button
            type="button"
            onClick={openSearch}
            className="w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg border border-kumo-line bg-kumo-recessed/40 text-kumo-subtle hover:text-kumo-default hover:bg-kumo-recessed text-xs transition-colors cursor-pointer group-data-[state=collapsed]/sidebar:justify-center group-data-[state=collapsed]/sidebar:px-0"
            title="Quick search (Press / or ⌘K)"
          >
            <div className="flex items-center gap-2 truncate">
              <MagnifyingGlass size={15} className="shrink-0" />
              <span className="truncate group-data-[state=collapsed]/sidebar:hidden">Quick search...</span>
            </div>
            <kbd className="hidden sm:inline-flex rounded border border-kumo-line bg-kumo-canvas px-1.5 py-0.5 text-[10px] font-mono text-kumo-subtle group-data-[state=collapsed]/sidebar:hidden">
              ⌘K
            </kbd>
          </button>
        </div>

        {/* Primary Views */}
        <SidebarGroup>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                icon={House}
                active={isPathActive('/overview')}
                tooltip="Home / Overview"
                onClick={() => navigate(withRunId('/overview'))}
              >
                Home
              </SidebarMenuButton>
            </SidebarMenuItem>

            <SidebarMenuItem>
              <SidebarMenuButton
                icon={ChartBar}
                active={isPathActive('/meta-alerts')}
                tooltip="Analytics & Logs"
                onClick={() => navigate(withRunId('/meta-alerts'))}
              >
                Analytics & Logs
              </SidebarMenuButton>
            </SidebarMenuItem>

            <SidebarMenuItem>
              <SidebarMenuButton
                icon={Cpu}
                active={isPathActive('/rbta')}
                tooltip="RBTA Engine"
                onClick={() => navigate(withRunId('/rbta'))}
              >
                RBTA Engine
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>

        {/* Build / Demonstration Section */}
        <SidebarGroup>
          <SidebarGroupLabel>Build</SidebarGroupLabel>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                icon={Play}
                active={isPathActive('/replay')}
                tooltip="Replay Demonstration"
                onClick={() => navigate(withRunId('/replay'))}
              >
                Replay Pipeline
              </SidebarMenuButton>
              <SidebarMenuSub>
                <SidebarMenuSubButton
                  active={isPathActive('/replay')}
                  onClick={() => navigate(withRunId('/replay'))}
                >
                  Visual Flowchart
                </SidebarMenuSubButton>
              </SidebarMenuSub>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>

        {/* Protect & Connect Section */}
        <SidebarGroup>
          <SidebarGroupLabel>Protect & Connect</SidebarGroupLabel>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                icon={Plugs}
                active={isPathActive('/integrations')}
                tooltip="Integrations"
                onClick={() => navigate(withRunId('/integrations'))}
              >
                Integrations
              </SidebarMenuButton>
            </SidebarMenuItem>

            <SidebarMenuItem>
              <SidebarMenuButton
                icon={GearSix}
                active={isPathActive('/system')}
                tooltip="System Health"
                onClick={() => navigate(withRunId('/system'))}
              >
                System Health
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      {/* 3. Footer with Collapse Trigger and Version */}
      <SidebarFooter className="p-3 border-t border-kumo-line flex items-center justify-between">
        <SidebarTrigger />
        <div className="flex items-center gap-1.5 group-data-[state=collapsed]/sidebar:hidden">
          <Badge variant="secondary">v1.0.0</Badge>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
