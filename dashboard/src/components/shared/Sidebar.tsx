import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Cpu, AlertTriangle, PlayCircle, Link2, Settings,
} from 'lucide-react';

const navGroups = [
  {
    label: 'MONITOR',
    items: [
      { to: '/overview', icon: LayoutDashboard, label: 'Overview' },
      { to: '/rbta', icon: Cpu, label: 'RBTA Engine' },
      { to: '/meta-alerts', icon: AlertTriangle, label: 'MetaAlerts' },
    ],
  },
  {
    label: 'DEMONSTRATION',
    items: [
      { to: '/replay', icon: PlayCircle, label: 'Replay' },
    ],
  },
  {
    label: 'OPERATIONS',
    items: [
      { to: '/integrations', icon: Link2, label: 'Integrations' },
      { to: '/system', icon: Settings, label: 'System' },
    ],
  },
];

export function Sidebar() {
  return (
    <nav
      className="w-60 shrink-0 border-r flex flex-col py-4 overflow-auto"
      style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
    >
      {navGroups.map((group) => (
        <div key={group.label} className="mb-4">
          <div
            className="px-4 py-1 text-[11px] font-semibold tracking-wider"
            style={{ color: 'var(--text-disabled)' }}
          >
            {group.label}
          </div>
          {group.items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-4 py-2 text-sm transition-colors relative ${
                  isActive ? 'font-semibold' : 'font-normal'
                }`
              }
              style={({ isActive }) => ({
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: isActive ? 'var(--brand-orange-soft)' : 'transparent',
                borderLeft: isActive ? '3px solid var(--brand-orange)' : '3px solid transparent',
              })}
            >
              <item.icon size={16} />
              {item.label}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}
