interface DecisionBadgeProps {
  decision: string;
  action: string;
}

export function DecisionBadge({ action }: DecisionBadgeProps) {
  const isEscalate = action === 'ESCALATE';
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-[3px]"
      style={{
        background: isEscalate ? 'var(--danger-soft)' : 'var(--bg-muted)',
        color: isEscalate ? 'var(--danger)' : 'var(--text-secondary)',
      }}
    >
      {isEscalate && '!'}
      {action}
    </span>
  );
}
