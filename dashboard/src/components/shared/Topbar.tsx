import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '@cloudflare/kumo/components/button';
import { Badge } from '@cloudflare/kumo/components/badge';
import {
  MagnifyingGlass, Question, Sun, Moon, Desktop, User,
} from '@phosphor-icons/react';
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
      if (e.key === '/' || (e.key === 'k' && (e.metaKey || e.ctrlKey))) {
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
      <header className="w-full h-14 min-h-[3.5rem] flex items-center justify-between px-6 lg:px-8 border-b border-kumo-line bg-kumo-canvas shrink-0 z-20 sticky top-0">
        {/* Left: Context Breadcrumb & Runtime Status Badges */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2 text-xs font-medium text-kumo-subtle">
            <span>Security Operations</span>
            <span>/</span>
            <span className="text-kumo-strong font-semibold">Production SOC</span>
          </div>

          <span className="h-4 w-px bg-kumo-line hidden sm:inline-block shrink-0" />

          {/* Runtime Mode & Health Status */}
          <div className="flex items-center gap-2 text-xs">
            {runId ? (
              <Badge variant="warning">REPLAY: {runId.slice(0, 8)}</Badge>
            ) : (
              <Badge variant="success">LIVE RUNTIME</Badge>
            )}

            <span className="hidden lg:inline-flex">
              <Badge variant={data?.system_status === 'READY' ? 'secondary' : 'error'}>
                {data?.system_status === 'READY' ? 'SYSTEM READY' : 'DEGRADED'}
              </Badge>
            </span>
          </div>
        </div>

        {/* Right: Quick Search, Theme Cycler, Shortcuts Help, User Info */}
        <div className="flex items-center gap-2">
          {/* Quick Search Button */}
          <button
            type="button"
            onClick={() => setShowPalette(true)}
            className="flex items-center justify-between gap-3 px-3 py-1.5 rounded-lg border border-kumo-line bg-kumo-recessed/40 hover:bg-kumo-recessed text-kumo-subtle hover:text-kumo-default text-xs transition-colors cursor-pointer w-36 sm:w-52"
            title="Search RBTA (Press / or ⌘K)"
          >
            <div className="flex items-center gap-2 truncate">
              <MagnifyingGlass size={14} className="shrink-0 text-kumo-subtle" />
              <span className="truncate">Search RBTA...</span>
            </div>
            <kbd className="rounded border border-kumo-line bg-kumo-canvas px-1.5 py-0.5 text-[10px] font-mono text-kumo-subtle">
              /
            </kbd>
          </button>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={cycleTheme}
            aria-label={`Theme: ${theme}`}
            className="text-kumo-subtle hover:text-kumo-default"
          >
            {theme === 'light' && <Sun size={16} />}
            {theme === 'dark' && <Moon size={16} />}
            {theme === 'system' && <Desktop size={16} />}
          </Button>

          {/* Keyboard Shortcuts Help */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowShortcuts(true)}
            aria-label="Keyboard shortcuts"
            className="text-kumo-subtle hover:text-kumo-default"
          >
            <Question size={16} />
          </Button>

          {/* User Profile Pill */}
          <div className="hidden sm:flex items-center gap-2.5 pl-3 ml-1 border-l border-kumo-line">
            <div className="size-7 rounded-full bg-kumo-recessed border border-kumo-line flex items-center justify-center text-kumo-strong">
              <User size={14} />
            </div>
            <div className="hidden xl:flex flex-col text-left">
              <span className="text-[12px] font-medium text-kumo-strong leading-none">SOC Analyst</span>
              <span className="text-[10px] font-mono text-kumo-subtle mt-0.5">UID 10001</span>
            </div>
          </div>
        </div>
      </header>

      <ShortcutsModal open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <CommandPalette open={showPalette} onClose={() => setShowPalette(false)} />
    </>
  );
}
