import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '@cloudflare/kumo/components/button';
import { Badge } from '@cloudflare/kumo/components/badge';
import {
  MagnifyingGlass, Question, Sun, Moon, Desktop, CaretDown, User, ShieldCheck,
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
      <header className="w-full h-14 min-h-[3.5rem] flex items-center justify-between px-4 lg:px-8 border-b border-kumo-hairline bg-kumo-canvas shrink-0 z-30 sticky top-0">
        {/* Left: Security Platform Logo, Workspace Context & Runtime Status */}
        <div className="flex items-center gap-4 min-w-0">
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-md border border-kumo-hairline bg-kumo-recessed flex items-center justify-center text-kumo-strong">
              <ShieldCheck size={18} weight="duotone" />
            </div>
            <span className="font-semibold text-sm tracking-tight text-kumo-strong hidden sm:inline-block">
              RBTA <span className="font-normal text-kumo-subtle">Security Analytics</span>
            </span>
          </div>

          <span className="h-4 w-px bg-kumo-hairline hidden sm:inline-block shrink-0" />

          {/* Account / Workspace Switcher Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-md text-xs font-medium bg-kumo-recessed/60 border border-kumo-hairline text-kumo-default hover:bg-kumo-recessed cursor-default transition-colors">
            <span className="w-1.5 h-1.5 rounded-full bg-kumo-strong" />
            <span className="truncate max-w-[150px]">Production SOC</span>
            <CaretDown size={11} className="text-kumo-subtle" />
          </div>

          {/* Runtime Mode & Health Status */}
          <div className="flex items-center gap-2.5 text-xs">
            {runId ? (
              <Badge variant="warning">REPLAY: {runId.slice(0, 8)}</Badge>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-kumo-recessed text-kumo-strong border border-kumo-hairline">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                LIVE RUNTIME
              </span>
            )}

            <span className="hidden lg:flex items-center gap-1.5 font-mono text-[11px] text-kumo-subtle bg-kumo-recessed/40 px-2.5 py-0.5 rounded border border-kumo-hairline">
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full ${
                  data?.system_status === 'READY' ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
              {data?.system_status === 'READY' ? 'SYSTEM READY' : 'DEGRADED'}
            </span>
          </div>
        </div>

        {/* Right: Global Search, Theme Cycler, Shortcuts Help, User Info */}
        <div className="flex items-center gap-2.5">
          {/* Quick Search Button */}
          <button
            onClick={() => setShowPalette(true)}
            className="flex items-center justify-between gap-3 px-3.5 py-1.5 rounded-md border border-kumo-hairline bg-kumo-recessed/40 hover:bg-kumo-recessed text-kumo-subtle hover:text-kumo-default text-xs transition-colors cursor-pointer w-40 sm:w-60"
            title="Search RBTA (Press / or ⌘K)"
          >
            <div className="flex items-center gap-2 truncate">
              <MagnifyingGlass size={14} className="shrink-0 text-kumo-subtle" />
              <span className="truncate">Search RBTA...</span>
            </div>
            <div className="flex items-center gap-0.5 shrink-0">
              <kbd className="rounded border border-kumo-hairline bg-kumo-canvas px-1.5 py-0.5 text-[10px] font-mono shadow-2xs text-kumo-subtle">
                /
              </kbd>
            </div>
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
          <div className="hidden sm:flex items-center gap-2.5 pl-3 ml-1 border-l border-kumo-hairline">
            <div className="w-7 h-7 rounded-full bg-kumo-recessed border border-kumo-hairline flex items-center justify-center text-kumo-strong">
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
