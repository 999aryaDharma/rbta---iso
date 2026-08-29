import * as React from 'react';

export function Tooltip({ content, children }: { content: React.ReactNode; children: React.ReactNode }) {
  const [show, setShow] = React.useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 px-2.5 py-1 rounded-[4px] text-[11px] font-mono z-50 whitespace-nowrap border shadow-sm"
          style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)', color: 'var(--text-primary)' }}
        >
          {content}
        </div>
      )}
    </div>
  );
}
