import { Dialog } from '@/components/ui/dialog';

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
    <Dialog open={open} onClose={onClose} title="Keyboard Shortcuts">
      <div className="space-y-3">
        <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
          Press keys sequentially (e.g., <kbd className="px-1.5 py-0.5 border rounded bg-[var(--bg-subtle)] font-mono text-[11px]">g</kbd> then <kbd className="px-1.5 py-0.5 border rounded bg-[var(--bg-subtle)] font-mono text-[11px]">o</kbd>)
        </p>
        <div className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between py-2 text-xs">
              <span style={{ color: 'var(--text-secondary)' }}>{s.desc}</span>
              <kbd className="px-2 py-0.5 rounded-[4px] border font-mono text-[11px] font-semibold" style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)' }}>
                {s.key}
              </kbd>
            </div>
          ))}
        </div>
      </div>
    </Dialog>
  );
}
