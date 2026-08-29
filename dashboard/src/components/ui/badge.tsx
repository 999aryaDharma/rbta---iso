import * as React from 'react';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'outline';
}

export function Badge({ className = '', variant = 'default', style, ...props }: BadgeProps) {
  const base = 'inline-flex items-center px-2 py-0.5 rounded-[4px] text-[11px] font-mono font-medium border';
  let variantStyles = {};
  if (variant === 'default') {
    variantStyles = { background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--text-secondary)' };
  } else if (variant === 'success') {
    variantStyles = { background: 'var(--success-soft)', borderColor: 'var(--success)', color: 'var(--success)' };
  } else if (variant === 'warning') {
    variantStyles = { background: 'var(--warning-soft)', borderColor: 'var(--warning)', color: 'var(--warning)' };
  } else if (variant === 'danger') {
    variantStyles = { background: 'var(--danger-soft)', borderColor: 'var(--danger)', color: 'var(--danger)' };
  } else if (variant === 'outline') {
    variantStyles = { background: 'transparent', borderColor: 'var(--border-default)', color: 'var(--text-primary)' };
  }

  return (
    <span
      className={`${base} ${className}`}
      style={{ ...variantStyles, ...style }}
      {...props}
    />
  );
}
