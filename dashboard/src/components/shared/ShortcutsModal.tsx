import { DialogRoot, Dialog, DialogTitle, DialogClose } from '@cloudflare/kumo/components/dialog';
import { Button } from '@cloudflare/kumo/components/button';

export function ShortcutsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const shortcuts = [
    { key: 'g o', desc: 'Navigate to Overview' },
    { key: 'g m', desc: 'Navigate to MetaAlerts' },
    { key: 'g r', desc: 'Navigate to RBTA Engine' },
    { key: 'g p', desc: 'Navigate to Demonstration Replay' },
    { key: 'g s', desc: 'Navigate to System' },
    { key: '/', desc: 'Open Command Palette / Quick Search' },
    { key: '?', desc: 'Open Keyboard Shortcuts Help' },
    { key: '[', desc: 'Previous member raw alert' },
    { key: ']', desc: 'Next member raw alert' },
  ];

  return (
    <DialogRoot open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <Dialog>
        <DialogTitle>Keyboard Shortcuts</DialogTitle>
        <div className="space-y-3 mt-4">
          <p className="text-xs text-kumo-subtle">
            Press keys sequentially (e.g., <kbd className="px-1.5 py-0.5 border border-kumo-hairline rounded bg-kumo-recessed font-mono text-[11px]">g</kbd> then <kbd className="px-1.5 py-0.5 border border-kumo-hairline rounded bg-kumo-recessed font-mono text-[11px]">o</kbd>)
          </p>
          <div className="divide-y divide-kumo-hairline">
            {shortcuts.map((s) => (
              <div key={s.key} className="flex items-center justify-between py-2 text-xs">
                <span className="text-kumo-default">{s.desc}</span>
                <kbd className="px-2 py-0.5 rounded-sm border border-kumo-hairline font-mono text-[11px] font-semibold bg-kumo-recessed text-kumo-default">
                  {s.key}
                </kbd>
              </div>
            ))}
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
