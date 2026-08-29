import { type ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, sub, icon }: MetricCardProps) {
  return (
    <div className="p-3 rounded-lg border border-kumo-hairline bg-kumo-base">
      <div className="text-xs text-kumo-subtle mb-1">
        {label}
        {icon && <span className="ml-2 inline-block">{icon}</span>}
      </div>
      <div className="text-lg font-semibold font-mono text-kumo-default">{value}</div>
      {sub && <div className="text-xs text-kumo-subtle mt-0.5">{sub}</div>}
    </div>
  );
}
