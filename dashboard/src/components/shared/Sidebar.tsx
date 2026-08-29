import React from 'react';
import { NavLink, useNavigate, useSearchParams } from 'react-router-dom';
import {
  LayoutDashboard, Shield, Cpu, Play, Network, Settings,
} from 'lucide-react';

const navGroups = [
  {
    label: 'OVERVIEW',
    items: [
      { to: '/overview', icon: LayoutDashboard, label: 'Overview' },
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
      { to: '/integrations', icon: Network, label: 'Integrations' },
      { to: '/system', icon: Settings, label: 'System' },
    ],
  },
];

export function Sidebar() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  React.useEffect(() => {
    let lastKey = '';
    let lastKeyTime = 0;

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) {
        return;
      }

      const now = Date.now();
      if (lastKey === 'g' && now - lastKeyTime < 1000) {
        if (e.key === 'o') {
          e.preventDefault();
          navigate(withRunId('/overview'));
        } else if (e.key === 'm') {
          e.preventDefault();
          navigate(withRunId('/meta-alerts'));
        } else if (e.key === 'r') {
          e.preventDefault();
          navigate(withRunId('/rbta'));
        } else if (e.key === 'p') {
          e.preventDefault();
          navigate(withRunId('/replay'));
        } else if (e.key === 's') {
          e.preventDefault();
          navigate(withRunId('/system'));
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
  }, [runId, navigate]);

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
              to={withRunId(item.to)}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-4 py-2 text-xs transition-colors relative ${
                  isActive ? 'font-semibold' : 'font-medium'
                }`
              }
              style={({ isActive }) => ({
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: isActive ? 'var(--brand-orange-soft)' : 'transparent',
                borderLeft: isActive ? '3px solid var(--brand-orange)' : '3px solid transparent',
              })}
            >
              <item.icon size={15} style={{ color: 'var(--brand-orange)' }} />
              {item.label}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}
