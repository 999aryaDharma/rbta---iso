import { useCallback, useEffect } from 'react';
import { NavLink, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupLabel,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem,
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
    <Sidebar className="border-r border-kumo-hairline bg-kumo-canvas select-none shrink-0 w-60">
      <SidebarContent className="py-2">
        {navGroups.map((group) => (
          <SidebarGroup key={group.label} className="px-2 py-1">
            <SidebarGroupLabel className="text-[10px] font-semibold text-kumo-subtle tracking-wider uppercase px-2 py-1">
              {group.label}
            </SidebarGroupLabel>
            <SidebarMenu className="space-y-0.5 mt-0.5">
              {group.items.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <NavLink to={withRunId(item.to)} className="block">
                    {({ isActive }) => (
                      <SidebarMenuButton
                        active={isActive}
                        tooltip={item.label}
                        className={`w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-xs transition-colors ${
                          isActive
                            ? 'bg-kumo-recessed text-kumo-default font-semibold shadow-2xs border-l-2 border-[#F6821F]'
                            : 'text-kumo-default hover:bg-kumo-recessed/70 hover:text-kumo-strong'
                        }`}
                      >
                        <item.icon
                          size={16}
                          weight={isActive ? 'fill' : 'regular'}
                          className={isActive ? 'text-[#F6821F]' : 'text-kumo-subtle'}
                        />
                        <span className="truncate">{item.label}</span>
                      </SidebarMenuButton>
                    )}
                  </NavLink>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>

      {/* Subtle Sidebar Footer */}
      <div className="p-3 border-t border-kumo-hairline bg-kumo-canvas/80 text-[11px] text-kumo-subtle flex items-center justify-between">
        <span className="font-mono">RBTA Engine</span>
        <span className="text-[10px] bg-kumo-recessed px-1.5 py-0.5 rounded border border-kumo-hairline font-mono">
          v1.0.0
        </span>
      </div>

      <SidebarRail />
    </Sidebar>
  );
}
