import * as React from 'react';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', style, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`w-full px-3 py-1.5 border rounded-[5px] text-xs transition-colors focus:outline-none focus:ring-1 focus:ring-[var(--action-blue)] ${className}`}
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
Input.displayName = 'Input';
