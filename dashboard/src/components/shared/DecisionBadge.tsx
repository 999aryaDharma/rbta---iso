import { Badge } from '@cloudflare/kumo/components/badge';

interface DecisionBadgeProps {
  decision?: string;
  action: string;
}

export function DecisionBadge({ action }: DecisionBadgeProps) {
  const isEscalate = action === 'ESCALATE';
  return (
    <Badge variant={isEscalate ? 'error' : 'secondary'}>
      {action}
    </Badge>
  );
}
