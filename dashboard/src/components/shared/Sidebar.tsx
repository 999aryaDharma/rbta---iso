import { useCallback, useEffect } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupLabel,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarTrigger,
  SidebarRail,
} from '@cloudflare/kumo/components/sidebar';
import { Badge } from '@cloudflare/kumo/components/badge';
import {
  ChartBar, PaintBucket, Cpu, Play, Plugs, GearSix,
} from '@phosphor-icons/react';

const navGroups = [
  {
    label: 'OVERVIEW',
    items: [
      { to: '/overview', icon: ChartBar, label: 'Overview' },
    ],
  },
  {
    label: 'INVESTIGATE',
    items: [
      { to: '/meta-alerts', icon: PaintBucket, label: 'MetaAlerts' },
      { to: '/rbta', icon: Cpu, label: 'RBTA Engine' },
    ],
  },
  {
    label: 'DEMONSTRATE',
    items: [
      { to: '/replay', icon: Play, label: 'Replay' },
    ],
  },
  {
    label: 'OPERATIONS',
    items: [
      { to: '/integrations', icon: Plugs, label: 'Integrations' },
      { to: '/system', icon: GearSix, label: 'System' },
    ],
  },
];

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

  return (
    <Sidebar>
      <SidebarContent>
        {navGroups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>
              {group.label}
            </SidebarGroupLabel>
            <SidebarMenu>
              {group.items.map((item) => {
                const isActive =
                  location.pathname === item.to ||
                  (item.to !== '/overview' && location.pathname.startsWith(item.to));
                return (
                  <SidebarMenuItem key={item.to}>
                    <SidebarMenuButton
                      icon={item.icon}
                      active={isActive}
                      tooltip={item.label}
                      onClick={() => navigate(withRunId(item.to))}
                    >
                      {item.label}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter className="flex items-center justify-between">
        <SidebarTrigger />
        <div className="flex items-center gap-2 group-data-[state=collapsed]/sidebar:hidden">
          <span className="font-mono text-xs text-kumo-subtle">RBTA Engine</span>
          <Badge variant="secondary">v1.0.0</Badge>
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
