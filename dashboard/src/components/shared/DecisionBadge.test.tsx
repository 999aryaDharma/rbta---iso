import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DecisionBadge } from './DecisionBadge';

describe('DecisionBadge Component', () => {
  it('renders ESCALATE action badge', () => {
    render(<DecisionBadge decision="CRITICAL" action="ESCALATE" />);
    const badge = screen.getByText('Action: ESCALATE');
    expect(badge).toBeDefined();
    expect(screen.getByText('Decision: CRITICAL')).toBeDefined();
  });

  it('renders SUPPRESS action badge', () => {
    render(<DecisionBadge action="SUPPRESS" />);
    const badge = screen.getByText('SUPPRESS');
    expect(badge).toBeDefined();
  });
});
