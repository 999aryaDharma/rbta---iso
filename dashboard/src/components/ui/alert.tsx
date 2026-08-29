import * as React from 'react';

export function Alert({ variant = 'default', children, className = '', style }: { variant?: 'default' | 'warning' | 'danger'; children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  let defaultStyle = { background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--text-primary)' };
  if (variant === 'warning') {
    defaultStyle = { background: 'var(--warning-soft)', borderColor: 'var(--warning)', color: 'var(--warning)' };
  } else if (variant === 'danger') {
    defaultStyle = { background: 'var(--danger-soft)', borderColor: 'var(--danger)', color: 'var(--danger)' };
  }

  return (
    <div className={`p-3 rounded-[5px] border text-xs flex items-start gap-2.5 ${className}`} style={{ ...defaultStyle, ...style }}>
      {children}
    </div>
  );
}
