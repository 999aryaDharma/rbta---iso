import { Badge } from '@cloudflare/kumo/components/badge';

interface DecisionBadgeProps {
  decision?: string;
  action: string;
}

export function DecisionBadge({ decision, action }: DecisionBadgeProps) {
  const actionVariant = action === 'ESCALATE' ? 'error' : action === 'DAILY_DIGEST' ? 'info' : 'secondary';
  return (
    <span className="inline-flex flex-wrap items-center gap-1" aria-label={`Decision ${decision || 'unknown'}, action ${action || 'SUPPRESS'}`}>
      {decision && <Badge variant="secondary" className="font-mono text-[11px] font-medium">Decision: {decision}</Badge>}
      <Badge variant={actionVariant} className="font-mono text-[11px] font-semibold">Action: {action || 'SUPPRESS'}</Badge>
    </span>
  );
}
