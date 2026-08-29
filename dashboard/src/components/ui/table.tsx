import * as React from 'react';

export function Table({ className = '', ...props }: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="w-full overflow-auto">
      <table className={`w-full text-xs text-left ${className}`} {...props} />
    </div>
  );
}

export function TableHeader({ className = '', style, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return (
    <thead
      className={`border-b ${className}`}
      style={{ background: 'var(--bg-subtle)', borderColor: 'var(--border-default)', color: 'var(--text-tertiary)', ...style }}
      {...props}
    />
  );
}

export function TableBody({ className = '', ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={`divide-y ${className}`} {...props} />;
}

export function TableRow({ className = '', style, ...props }: React.HTMLAttributes<HTMLTableRowElement>) {
  return (
    <tr
      className={`border-b transition-colors hover:bg-[var(--bg-hover)] ${className}`}
      style={{ borderColor: 'var(--border-subtle)', ...style }}
      {...props}
    />
  );
}

export function TableHead({ className = '', style, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={`px-4 py-2.5 font-medium text-xs ${className}`}
      style={{ color: 'var(--text-tertiary)', ...style }}
      {...props}
    />
  );
}

export function TableCell({ className = '', style, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td
      className={`px-4 py-2.5 text-xs ${className}`}
      style={{ color: 'var(--text-primary)', ...style }}
      {...props}
    />
  );
}
