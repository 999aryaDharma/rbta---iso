import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Activity, Search, HelpCircle, Sun, Moon, Laptop } from 'lucide-react';
import { usePollingQuery } from '@/hooks/usePolling';
import { fetchSummary } from '@/api/dashboard';
import { useTheme } from '@/context/ThemeContext';
import { ShortcutsModal } from './ShortcutsModal';
import { CommandPalette } from './CommandPalette';

export function Topbar() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('run_id');
  const { data } = usePollingQuery(['summary-topbar', runId || 'live'], () => fetchSummary(runId || undefined), 3000);
  const { theme, setTheme } = useTheme();

  const [showShortcuts, setShowShortcuts] = React.useState(false);
  const [showPalette, setShowPalette] = React.useState(false);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) {
        return;
      }
      if (e.key === '/') {
        e.preventDefault();
        setShowPalette(true);
      } else if (e.key === '?') {
        e.preventDefault();
        setShowShortcuts(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dark');
    else if (theme === 'dark') setTheme('system');
    else setTheme('light');
  };

  return (
    <>
      <header
        className="h-14 flex items-center justify-between px-4 border-b shrink-0"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity size={18} style={{ color: 'var(--brand-orange)' }} />
            <span className="font-semibold text-sm tracking-tight" style={{ color: 'var(--text-primary)' }}>
              RBTA <span className="font-normal text-xs" style={{ color: 'var(--text-tertiary)' }}>Security Analytics</span>
            </span>
          </div>

          <div className="flex items-center gap-2 ml-4 text-xs">
            {runId ? (
              <span
                className="px-2 py-0.5 rounded-[4px] font-mono text-[11px] border font-medium"
                style={{ background: 'var(--brand-orange-soft)', borderColor: 'var(--brand-orange)', color: 'var(--brand-orange)' }}
              >
                REPLAY: {runId.slice(0, 8)}
              </span>
            ) : (
              <span
                className="px-2 py-0.5 rounded-[4px] font-mono text-[11px] border font-medium"
                style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
              >
                LIVE RUNTIME
              </span>
            )}

            <span className="flex items-center gap-1 font-mono text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: data?.system_status === 'READY' ? 'var(--success)' : 'var(--danger)' }}
              />
              {data?.system_status === 'READY' ? 'READY' : 'NOT READY'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Quick Search Button */}
          <button
            onClick={() => setShowPalette(true)}
            className="flex items-center gap-2 px-2.5 py-1 rounded-[5px] border text-xs cursor-pointer hover:bg-[var(--bg-subtle)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)', background: 'var(--bg-surface)' }}
          >
            <Search size={13} />
            <span className="text-[11px]">Quick Search</span>
            <kbd className="px-1.5 py-0.5 rounded border text-[10px] font-mono" style={{ borderColor: 'var(--border-default)', background: 'var(--bg-subtle)' }}>
              /
            </kbd>
          </button>

          {/* Theme Switcher */}
          <button
            onClick={cycleTheme}
            title={`Theme: ${theme} (click to toggle)`}
            className="p-1.5 rounded-[5px] border text-xs cursor-pointer hover:bg-[var(--bg-subtle)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)', background: 'var(--bg-surface)' }}
          >
            {theme === 'light' && <Sun size={14} />}
            {theme === 'dark' && <Moon size={14} />}
            {theme === 'system' && <Laptop size={14} />}
          </button>

          {/* Keyboard Shortcuts Help */}
          <button
            onClick={() => setShowShortcuts(true)}
            title="Keyboard Shortcuts (?)"
            className="p-1.5 rounded-[5px] border text-xs cursor-pointer hover:bg-[var(--bg-subtle)]"
            style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)', background: 'var(--bg-surface)' }}
          >
            <HelpCircle size={14} />
          </button>
        </div>
      </header>

      <ShortcutsModal open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <CommandPalette open={showPalette} onClose={() => setShowPalette(false)} />
    </>
  );
}
