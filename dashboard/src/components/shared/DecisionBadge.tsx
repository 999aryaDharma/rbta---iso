import { Badge } from '@cloudflare/kumo/components/badge';

interface DecisionBadgeProps {
  decision?: string;
  action: string;
}

export function DecisionBadge({ action }: DecisionBadgeProps) {
  if (action === 'ESCALATE') {
    return (
      <Badge variant="error" className="font-semibold tracking-wide">
        ESCALATE
      </Badge>
    );
  }

  if (action === 'DAILY_DIGEST') {
    return (
      <Badge variant="info" className="font-medium">
        DAILY_DIGEST
      </Badge>
    );
  }

  return (
    <Badge variant="secondary" className="font-normal text-kumo-subtle">
      {action || 'SUPPRESS'}
    </Badge>
  );
}
