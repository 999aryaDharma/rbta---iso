import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '@cloudflare/kumo/components/button';
import { Badge } from '@cloudflare/kumo/components/badge';
import { MagnifyingGlass, Question, Sun, Moon, Desktop } from '@phosphor-icons/react';
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
      <header className="h-14 flex items-center justify-between px-4 border-b border-kumo-hairline bg-kumo-base shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs">
            {runId ? (
              <Badge variant="warning">REPLAY: {runId.slice(0, 8)}</Badge>
            ) : (
              <Badge variant="secondary">LIVE RUNTIME</Badge>
            )}

            <span className="flex items-center gap-1 font-mono text-[11px] text-kumo-subtle">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  data?.system_status === 'READY' ? 'bg-kumo-success' : 'bg-kumo-danger'
                }`}
              />
              {data?.system_status === 'READY' ? 'READY' : 'NOT READY'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <Button variant="ghost" size="sm" onClick={() => setShowPalette(true)}>
            <MagnifyingGlass size={14} />
            <span className="text-xs">Search</span>
            <kbd className="ml-1 rounded border border-kumo-line bg-kumo-recessed px-1.5 py-0.5 text-[10px] font-mono">/</kbd>
          </Button>

          <Button variant="ghost" size="sm" onClick={cycleTheme} aria-label={`Theme: ${theme}`}>
            {theme === 'light' && <Sun size={16} />}
            {theme === 'dark' && <Moon size={16} />}
            {theme === 'system' && <Desktop size={16} />}
          </Button>

          <Button variant="ghost" size="sm" onClick={() => setShowShortcuts(true)} aria-label="Keyboard shortcuts">
            <Question size={16} />
          </Button>
        </div>
      </header>

      <ShortcutsModal open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <CommandPalette open={showPalette} onClose={() => setShowPalette(false)} />
    </>
  );
}
