import * as React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'default', size = 'sm', ...props }, ref) => {
    const base = 'inline-flex items-center justify-center rounded-[5px] font-medium transition-colors cursor-pointer disabled:opacity-50 disabled:pointer-events-none';
    const sizeClasses = {
      sm: 'px-3 py-1.5 text-xs',
      md: 'px-4 py-2 text-sm',
      lg: 'px-5 py-2.5 text-base',
    }[size];

    let variantStyles = {};
    if (variant === 'default') {
      variantStyles = { background: 'var(--action-blue)', color: 'white' };
    } else if (variant === 'outline') {
      variantStyles = { borderColor: 'var(--border-default)', border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' };
    } else if (variant === 'secondary') {
      variantStyles = { background: 'var(--bg-subtle)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' };
    } else if (variant === 'danger') {
      variantStyles = { background: 'var(--danger)', color: 'white' };
    } else if (variant === 'ghost') {
      variantStyles = { background: 'transparent', color: 'var(--text-secondary)' };
    }

    return (
      <button
        ref={ref}
        className={`${base} ${sizeClasses} ${className}`}
        style={variantStyles}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';
