import { Badge } from '@cloudflare/kumo/components/badge';

interface DecisionBadgeProps {
  decision?: string;
  action: string;
}

export function DecisionBadge({ action }: DecisionBadgeProps) {
  if (action === 'ESCALATE') {
    return (
      <Badge variant="error" className="font-mono text-[11px] font-semibold tracking-wide">
        ESCALATE
      </Badge>
    );
  }

  if (action === 'DAILY_DIGEST') {
    return (
      <Badge variant="info" className="font-mono text-[11px] font-medium">
        DAILY_DIGEST
      </Badge>
    );
  }

  return (
    <Badge variant="secondary" className="font-mono text-[11px] text-kumo-subtle font-normal">
      {action || 'SUPPRESS'}
    </Badge>
  );
}
