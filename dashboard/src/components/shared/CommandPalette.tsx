import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Dialog } from '@/components/ui/dialog';
import { Search, LayoutDashboard, Shield, Cpu, Play, Network, Settings, ArrowRight } from 'lucide-react';

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = React.useState('');
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) => (runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path);

  const navItems = [
    { label: 'Overview', path: withRunId('/overview'), icon: LayoutDashboard },
    { label: 'MetaAlerts', path: withRunId('/meta-alerts'), icon: Shield },
    { label: 'RBTA Engine', path: withRunId('/rbta'), icon: Cpu },
    { label: 'Demonstration Replay', path: withRunId('/replay'), icon: Play },
    { label: 'Integrations', path: withRunId('/integrations'), icon: Network },
    { label: 'System Configuration', path: withRunId('/system'), icon: Settings },
  ];

  const filteredItems = navItems.filter((i) =>
    i.label.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (path: string) => {
    navigate(path);
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      const numMatch = query.match(/^#?(\d+)$/);
      if (numMatch) {
        navigate(withRunId(`/meta-alerts/${numMatch[1]}`));
        onClose();
      } else if (filteredItems.length > 0) {
        handleSelect(filteredItems[0].path);
      }
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title="Quick Search & Navigation">
      <div className="space-y-4">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
          <input
            autoFocus
            type="text"
            placeholder="Search pages or type Meta ID (e.g. 42)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full pl-8 pr-3 py-2 border rounded-[5px] text-xs bg-[var(--bg-surface)] focus:outline-none focus:ring-1 focus:ring-[var(--action-blue)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
          />
        </div>

        <div className="space-y-1 max-h-60 overflow-auto">
          {filteredItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.path}
                onClick={() => handleSelect(item.path)}
                className="w-full flex items-center justify-between p-2 rounded-[5px] text-xs text-left transition-colors hover:bg-[var(--bg-hover)] cursor-pointer"
                style={{ color: 'var(--text-primary)' }}
              >
                <div className="flex items-center gap-2.5">
                  <Icon size={14} style={{ color: 'var(--brand-orange)' }} />
                  <span>{item.label}</span>
                </div>
                <ArrowRight size={12} style={{ color: 'var(--text-tertiary)' }} />
              </button>
            );
          })}
          {filteredItems.length === 0 && (
            <div className="p-4 text-center text-xs" style={{ color: 'var(--text-tertiary)' }}>
              No matching pages. Press Enter to jump if typing a numeric Meta ID.
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
}
