import * as React from 'react';

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', style, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={`px-3 py-1.5 border rounded-[5px] text-xs font-mono transition-colors focus:outline-none focus:ring-1 focus:ring-[var(--action-blue)] ${className}`}
        style={{
          background: 'var(--bg-surface)',
          borderColor: 'var(--border-default)',
          color: 'var(--text-primary)',
          ...style,
        }}
        {...props}
      />
    );
  }
);
Select.displayName = 'Select';
