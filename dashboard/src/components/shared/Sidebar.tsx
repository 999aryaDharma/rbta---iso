import { useCallback, useEffect } from 'react';
import { NavLink, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupLabel,
  SidebarHeader, SidebarMenu, SidebarMenuButton, SidebarMenuItem,
  SidebarRail,
} from '@cloudflare/kumo/components/sidebar';
import {
  ChartBar, Shield, Cpu, Play, Plugs, GearSix,
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
      { to: '/meta-alerts', icon: Shield, label: 'MetaAlerts' },
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
    <Sidebar className="border-r border-kumo-hairline">
      <SidebarHeader className="h-14 flex items-center px-4 border-b border-kumo-hairline">
        <span className="text-sm font-semibold text-kumo-default tracking-tight">
          RBTA <span className="text-kumo-subtle font-normal text-xs">Security Analytics</span>
        </span>
      </SidebarHeader>
      <SidebarContent>
        {navGroups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarMenu>
              {group.items.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <NavLink to={withRunId(item.to)}>
                    {({ isActive }) => (
                      <SidebarMenuButton active={isActive} tooltip={item.label}>
                        <item.icon size={16} weight={isActive ? 'fill' : 'regular'} />
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    )}
                  </NavLink>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
}
