import * as React from 'react';

export function Skeleton({ className = '', style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={`animate-pulse rounded-[4px] ${className}`}
      style={{ background: 'var(--bg-subtle)', ...style }}
    />
  );
}
