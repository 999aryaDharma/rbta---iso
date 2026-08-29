import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DialogRoot, Dialog, DialogTitle, DialogClose } from '@cloudflare/kumo/components/dialog';
import { Input } from '@cloudflare/kumo/components/input';
import { Button } from '@cloudflare/kumo/components/button';
import { ChartBar, Shield, Cpu, Play, Plugs, GearSix, ArrowRight } from '@phosphor-icons/react';

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = React.useState('');
  const runId = searchParams.get('run_id');

  const withRunId = (path: string) =>
    runId ? `${path}${path.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}` : path;

  const navItems = [
    { label: 'Overview', path: withRunId('/overview'), icon: ChartBar },
    { label: 'MetaAlerts', path: withRunId('/meta-alerts'), icon: Shield },
    { label: 'RBTA Engine', path: withRunId('/rbta'), icon: Cpu },
    { label: 'Demonstration Replay', path: withRunId('/replay'), icon: Play },
    { label: 'Integrations', path: withRunId('/integrations'), icon: Plugs },
    { label: 'System Configuration', path: withRunId('/system'), icon: GearSix },
  ];

  const filteredItems = navItems.filter((i) =>
    i.label.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (path: string) => {
    navigate(path);
    setQuery('');
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      const numMatch = query.match(/^#?(\d+)$/);
      if (numMatch) {
        navigate(withRunId(`/meta-alerts/${numMatch[1]}`));
        setQuery('');
        onClose();
      } else if (filteredItems.length > 0) {
        handleSelect(filteredItems[0].path);
      }
    }
  };

  return (
    <DialogRoot open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <Dialog>
        <DialogTitle>Quick Search & Navigation</DialogTitle>
        <div className="space-y-4 mt-4">
          <Input
            autoFocus
            type="text"
            placeholder="Search pages or type Meta ID (e.g. 42)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <div className="space-y-1 max-h-60 overflow-auto">
            {filteredItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.path}
                  onClick={() => handleSelect(item.path)}
                  className="w-full flex items-center justify-between rounded-lg p-2 text-sm text-left transition-colors hover:bg-kumo-tint text-kumo-default cursor-pointer"
                >
                  <div className="flex items-center gap-2.5">
                    <Icon size={16} className="text-kumo-brand" />
                    <span>{item.label}</span>
                  </div>
                  <ArrowRight size={14} className="text-kumo-subtle" />
                </button>
              );
            })}
            {filteredItems.length === 0 && (
              <div className="p-4 text-center text-sm text-kumo-subtle">
                No matching pages. Press Enter to jump if typing a numeric Meta ID.
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 flex justify-end">
          <DialogClose>
            <Button variant="ghost" size="sm">Close</Button>
          </DialogClose>
        </div>
      </Dialog>
    </DialogRoot>
  );
}
