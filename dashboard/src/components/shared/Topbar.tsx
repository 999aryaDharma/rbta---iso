import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '@cloudflare/kumo/components/button';
import { Badge } from '@cloudflare/kumo/components/badge';
import {
  MagnifyingGlass, Question, Sun, Moon, Desktop, CaretDown, User,
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
      <header className="w-full h-14 min-h-[3.5rem] flex items-center justify-between px-4 lg:px-6 border-b border-kumo-hairline bg-kumo-canvas shrink-0 z-30 sticky top-0">
        {/* Left: Cloudflare Flame Logo, Account Context & Runtime Status */}
        <div className="flex items-center gap-3 md:gap-4 min-w-0">
          <div className="flex items-center gap-2 shrink-0">
            {/* Cloudflare Cloud Logo Icon */}
            <svg
              viewBox="0 0 24 24"
              className="w-6 h-6 text-[#F6821F] shrink-0 fill-current"
              aria-hidden="true"
            >
              <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z" />
            </svg>
            <span className="font-semibold text-sm tracking-tight text-kumo-default hidden sm:inline-block">
              Cloudflare <span className="font-normal text-kumo-subtle">| RBTA</span>
            </span>
          </div>

          <span className="h-4 w-px bg-kumo-line hidden sm:inline-block shrink-0" />

          {/* Account / Workspace Switcher Pill */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-kumo-recessed/70 border border-kumo-hairline text-kumo-default hover:bg-kumo-recessed cursor-default transition-colors">
            <span className="w-1.5 h-1.5 rounded-full bg-[#F6821F]" />
            <span className="truncate max-w-[140px]">RBTA SOC Analytics</span>
            <CaretDown size={11} className="text-kumo-subtle" />
          </div>

          {/* Runtime Mode & Health Status */}
          <div className="flex items-center gap-2 text-xs">
            {runId ? (
              <Badge variant="warning">REPLAY: {runId.slice(0, 8)}</Badge>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE RUNTIME
              </span>
            )}

            <span className="hidden lg:flex items-center gap-1.5 font-mono text-[11px] text-kumo-subtle bg-kumo-recessed/50 px-2 py-0.5 rounded border border-kumo-hairline">
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full ${
                  data?.system_status === 'READY' ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
              {data?.system_status === 'READY' ? 'READY' : 'NOT READY'}
            </span>
          </div>
        </div>

        {/* Center / Right: Global Search, Theme Cycler, Shortcuts Help, User Info */}
        <div className="flex items-center gap-2">
          {/* Quick Search Button */}
          <button
            onClick={() => setShowPalette(true)}
            className="flex items-center justify-between gap-3 px-3 py-1.5 rounded-md border border-kumo-line bg-kumo-recessed/60 hover:bg-kumo-recessed text-kumo-subtle hover:text-kumo-default text-xs transition-colors cursor-pointer w-36 sm:w-56"
            title="Search RBTA (Press / or ⌘K)"
          >
            <div className="flex items-center gap-1.5 truncate">
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
          <div className="hidden sm:flex items-center gap-2 pl-2 ml-1 border-l border-kumo-hairline">
            <div className="w-7 h-7 rounded-full bg-[#F6821F]/10 border border-[#F6821F]/20 flex items-center justify-center text-[#F6821F]">
              <User size={14} weight="bold" />
            </div>
            <div className="hidden xl:flex flex-col text-left">
              <span className="text-[12px] font-medium text-kumo-default leading-none">SOC Analyst</span>
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
