import { type ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, sub, icon }: MetricCardProps) {
  return (
    <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-canvas hover:border-kumo-line/80 transition-all shadow-xs flex flex-col justify-between">
      <div className="flex items-center justify-between text-[11px] font-semibold text-kumo-subtle uppercase tracking-wider">
        <span className="truncate">{label}</span>
        {icon && <span className="text-kumo-subtle shrink-0 ml-2">{icon}</span>}
      </div>
      <div className="text-2xl lg:text-3xl font-semibold font-mono text-kumo-default tracking-tight mt-2">
        {value}
      </div>
      {sub ? (
        <div className="text-xs text-kumo-subtle mt-1.5 truncate">
          {sub}
        </div>
      ) : (
        <div className="h-4 mt-1.5" />
      )}
    </div>
  );
}
