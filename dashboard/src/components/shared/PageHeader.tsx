import { type ReactNode } from 'react';

export function PageHeader({ title, description, actions }: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="border-b border-kumo-hairline bg-kumo-canvas">
      <div className="flex items-start justify-between px-6 py-5">
        <div>
          <h1 className="text-lg font-semibold text-kumo-default">{title}</h1>
          {description && (
            <p className="text-sm mt-0.5 text-kumo-subtle">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
