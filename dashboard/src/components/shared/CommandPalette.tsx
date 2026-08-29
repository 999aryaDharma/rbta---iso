import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DialogRoot, Dialog, DialogClose } from '@cloudflare/kumo/components/dialog';
import { Button } from '@cloudflare/kumo/components/button';
import {
  ChartBar, Shield, Cpu, Play, Plugs, GearSix, ArrowRight, MagnifyingGlass, Hash,
} from '@phosphor-icons/react';

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = React.useState('');
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) =>
    runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path;

  const navItems = [
    { label: 'Overview Dashboard', category: 'Pages', desc: 'Real-time security analytics and live timeseries', path: withRunId('/overview'), icon: ChartBar },
    { label: 'MetaAlerts Explorer', category: 'Pages', desc: 'Investigate clustered alerts and anomaly scores', path: withRunId('/meta-alerts'), icon: Shield },
    { label: 'RBTA Engine State', category: 'Pages', desc: 'Inspect active windows and temporal reduction', path: withRunId('/rbta'), icon: Cpu },
    { label: 'Demonstration Replay', category: 'Pages', desc: 'Step through historical attack validation datasets', path: withRunId('/replay'), icon: Play },
    { label: 'Integrations & Shuffle SOAR', category: 'Operations', desc: 'Manage Shuffle webhook and deferred Telegram sink', path: withRunId('/integrations'), icon: Plugs },
    { label: 'System Configuration', category: 'Operations', desc: 'Governance, model calibration, and environment', path: withRunId('/system'), icon: GearSix },
  ];

  const isNumericId = /^#?(\d+)$/.test(query.trim());
  const numericId = query.trim().replace(/^#/, '');

  const filteredItems = navItems.filter((i) =>
    i.label.toLowerCase().includes(query.toLowerCase()) ||
    i.desc.toLowerCase().includes(query.toLowerCase())
  );

  React.useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleSelect = (path: string) => {
    navigate(path);
    setQuery('');
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % (filteredItems.length + (isNumericId ? 1 : 0)));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const total = filteredItems.length + (isNumericId ? 1 : 0);
      setSelectedIndex((prev) => (prev - 1 + total) % total);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (isNumericId) {
        navigate(withRunId(`/meta-alerts/${numericId}`));
        setQuery('');
        onClose();
      } else if (filteredItems[selectedIndex]) {
        handleSelect(filteredItems[selectedIndex].path);
      }
    }
  };

  return (
    <DialogRoot open={open} onOpenChange={(o) => { if (!o) { setQuery(''); onClose(); } }}>
      <Dialog className="max-w-xl w-full p-0 bg-kumo-canvas border border-kumo-hairline shadow-2xl rounded-xl overflow-hidden">
        {/* Search Header Input */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-kumo-hairline bg-kumo-canvas">
          <MagnifyingGlass size={18} className="text-[#F6821F] shrink-0" />
          <input
            autoFocus
            type="text"
            placeholder="Search RBTA pages or type numeric Meta ID (e.g. 42)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full bg-transparent text-sm text-kumo-default placeholder:text-kumo-subtle focus:outline-hidden font-normal"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="text-xs text-kumo-subtle hover:text-kumo-default px-1 py-0.5 rounded cursor-pointer"
            >
              Clear
            </button>
          )}
        </div>

        {/* Results Body */}
        <div className="p-2 max-h-[340px] overflow-y-auto space-y-1">
          {/* Quick Direct Jump if Numeric ID */}
          {isNumericId && (
            <button
              onClick={() => {
                navigate(withRunId(`/meta-alerts/${numericId}`));
                setQuery('');
                onClose();
              }}
              className="w-full flex items-center justify-between rounded-lg p-2.5 text-sm text-left transition-colors bg-[#F6821F]/10 border border-[#F6821F]/20 text-kumo-default cursor-pointer"
            >
              <div className="flex items-center gap-2.5">
                <div className="w-6 h-6 rounded-md bg-[#F6821F] text-white flex items-center justify-center">
                  <Hash size={14} weight="bold" />
                </div>
                <div>
                  <div className="font-semibold text-xs text-kumo-default">
                    Jump directly to MetaAlert #{numericId}
                  </div>
                  <div className="text-[11px] text-kumo-subtle">
                    Open detailed investigation view for MetaAlert ID {numericId}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1 text-xs text-[#F6821F] font-medium">
                <span>Jump</span>
                <ArrowRight size={13} />
              </div>
            </button>
          )}

          {/* Navigation Items */}
          {filteredItems.map((item, idx) => {
            const Icon = item.icon;
            const isSelected = idx === selectedIndex && !isNumericId;
            return (
              <button
                key={item.path}
                onClick={() => handleSelect(item.path)}
                onMouseEnter={() => setSelectedIndex(idx)}
                className={`w-full flex items-center justify-between rounded-lg p-2.5 text-sm text-left transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-kumo-recessed text-kumo-default border border-kumo-line/60'
                    : 'hover:bg-kumo-recessed/60 text-kumo-default border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-md bg-kumo-recessed border border-kumo-hairline flex items-center justify-center text-[#F6821F] shrink-0">
                    <Icon size={16} />
                  </div>
                  <div>
                    <div className="font-medium text-xs text-kumo-default flex items-center gap-2">
                      <span>{item.label}</span>
                      <span className="text-[10px] uppercase font-semibold text-kumo-subtle tracking-wider px-1.5 py-0.2 rounded bg-kumo-canvas border border-kumo-hairline">
                        {item.category}
                      </span>
                    </div>
                    <div className="text-[11px] text-kumo-subtle mt-0.5">
                      {item.desc}
                    </div>
                  </div>
                </div>
                <ArrowRight
                  size={14}
                  className={isSelected ? 'text-kumo-default' : 'text-kumo-subtle'}
                />
              </button>
            );
          })}

          {filteredItems.length === 0 && !isNumericId && (
            <div className="py-8 text-center text-xs text-kumo-subtle space-y-1">
              <p className="font-medium text-kumo-default">No matching results found</p>
              <p>Type a numeric ID like <span className="font-mono text-kumo-brand">42</span> to jump to a specific MetaAlert</p>
            </div>
          )}
        </div>

        {/* Footer with Hotkey Hints */}
        <div className="px-4 py-2.5 border-t border-kumo-hairline bg-kumo-recessed/40 flex items-center justify-between text-[11px] text-kumo-subtle">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded border border-kumo-hairline bg-kumo-canvas font-mono text-[10px]">↑↓</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded border border-kumo-hairline bg-kumo-canvas font-mono text-[10px]">↵</kbd>
              Select
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded border border-kumo-hairline bg-kumo-canvas font-mono text-[10px]">ESC</kbd>
              Close
            </span>
          </div>
          <DialogClose>
            <Button variant="ghost" size="sm" className="h-6 text-[11px]">
              Dismiss
            </Button>
          </DialogClose>
        </div>
      </Dialog>
    </DialogRoot>
  );
}
