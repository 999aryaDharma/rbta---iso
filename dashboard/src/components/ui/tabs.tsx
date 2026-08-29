import * as React from 'react';

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (val: string) => void;
}
const TabsContext = React.createContext<TabsContextValue>({ activeTab: '', setActiveTab: () => {} });

export function Tabs({ defaultValue, value, onValueChange, children, className = '' }: { defaultValue?: string; value?: string; onValueChange?: (val: string) => void; children: React.ReactNode; className?: string }) {
  const [tab, setTab] = React.useState(defaultValue || '');
  const currentTab = value !== undefined ? value : tab;
  const setTabVal = onValueChange || setTab;

  return (
    <TabsContext.Provider value={{ activeTab: currentTab, setActiveTab: setTabVal }}>
      <div className={`space-y-4 ${className}`}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ children, className = '', style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={`inline-flex items-center gap-1 p-1 rounded-[6px] border ${className}`}
      style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', ...style }}
    >
      {children}
    </div>
  );
}

export function TabsTrigger({ value, children, className = '' }: { value: string; children: React.ReactNode; className?: string }) {
  const { activeTab, setActiveTab } = React.useContext(TabsContext);
  const isActive = activeTab === value;

  return (
    <button
      type="button"
      onClick={() => setActiveTab(value)}
      className={`px-3 py-1 rounded-[4px] text-xs font-medium transition-all cursor-pointer ${className}`}
      style={{
        background: isActive ? 'var(--bg-surface)' : 'transparent',
        color: isActive ? 'var(--brand-orange)' : 'var(--text-secondary)',
        boxShadow: isActive ? '0 1px 2px rgba(0,0,0,0.05)' : 'none',
        fontWeight: isActive ? 600 : 500,
      }}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children, className = '' }: { value: string; children: React.ReactNode; className?: string }) {
  const { activeTab } = React.useContext(TabsContext);
  if (activeTab !== value) return null;
  return <div className={`mt-2 ${className}`}>{children}</div>;
}
