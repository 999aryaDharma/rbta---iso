import { type ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, sub, icon }: MetricCardProps) {
  return (
    <div className="p-5 rounded-lg border border-kumo-hairline bg-kumo-base shadow-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-kumo-subtle">
          {label}
        </span>
        {icon}
      </div>
      <div className="text-2xl font-semibold text-kumo-default tracking-tight">
        {value}
      </div>
      {sub && (
        <div className="text-xs mt-1 text-kumo-subtle">
          {sub}
        </div>
      )}
    </div>
  );
}
