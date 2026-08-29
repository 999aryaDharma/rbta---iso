import { DialogRoot, Dialog, DialogTitle, DialogClose } from '@cloudflare/kumo/components/dialog';
import { Button } from '@cloudflare/kumo/components/button';
import { Keyboard } from '@phosphor-icons/react';

export function ShortcutsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const shortcutGroups = [
    {
      category: 'GLOBAL NAVIGATION',
      items: [
        { key: 'g o', desc: 'Navigate to Overview Dashboard' },
        { key: 'g m', desc: 'Navigate to MetaAlerts Explorer' },
        { key: 'g r', desc: 'Navigate to RBTA Aggregation Engine' },
        { key: 'g p', desc: 'Navigate to Demonstration Replay' },
        { key: 'g s', desc: 'Navigate to System & Pipeline Health' },
      ],
    },
    {
      category: 'SEARCH & SHORTCUTS',
      items: [
        { key: '/', desc: 'Open Command Palette / Fast Search' },
        { key: '⌘ K', desc: 'Focus Quick Navigation Search' },
        { key: '?', desc: 'Toggle Keyboard Shortcuts Modal' },
      ],
    },
    {
      category: 'INVESTIGATION',
      items: [
        { key: '[', desc: 'Previous member alert in MetaAlert' },
        { key: ']', desc: 'Next member alert in MetaAlert' },
      ],
    },
  ];

  return (
    <DialogRoot open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <Dialog className="max-w-md w-full p-6 bg-kumo-canvas border border-kumo-hairline shadow-2xl rounded-xl">
        <div className="flex items-center gap-2.5 pb-3 border-b border-kumo-hairline">
          <div className="w-8 h-8 rounded-lg bg-[#F6821F]/10 border border-[#F6821F]/20 flex items-center justify-center text-[#F6821F]">
            <Keyboard size={18} weight="duotone" />
          </div>
          <div>
            <DialogTitle className="text-base font-semibold text-kumo-default leading-none">
              Keyboard Shortcuts
            </DialogTitle>
            <p className="text-xs text-kumo-subtle mt-1">
              Press keys sequentially to trigger actions rapidly
            </p>
          </div>
        </div>

        <div className="space-y-4 my-4 max-h-[60vh] overflow-y-auto pr-1">
          {shortcutGroups.map((group) => (
            <div key={group.category} className="space-y-1.5">
              <span className="text-[10px] font-semibold text-kumo-subtle uppercase tracking-wider block px-1">
                {group.category}
              </span>
              <div className="rounded-lg border border-kumo-hairline bg-kumo-recessed/40 divide-y divide-kumo-hairline/60">
                {group.items.map((s) => (
                  <div
                    key={s.key}
                    className="flex items-center justify-between px-3 py-2 text-xs"
                  >
                    <span className="text-kumo-default">{s.desc}</span>
                    <kbd className="px-2 py-0.5 rounded border border-kumo-line font-mono text-[11px] font-medium bg-kumo-canvas shadow-2xs text-kumo-strong">
                      {s.key}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="pt-3 border-t border-kumo-hairline flex items-center justify-between">
          <span className="text-[11px] text-kumo-subtle font-mono">
            RBTA v1.0 • Hotkeys Active
          </span>
          <DialogClose>
            <Button variant="secondary" size="sm">
              Done
            </Button>
          </DialogClose>
        </div>
      </Dialog>
    </DialogRoot>
  );
}
