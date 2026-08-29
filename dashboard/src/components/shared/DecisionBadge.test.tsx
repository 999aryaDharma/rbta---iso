import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DecisionBadge } from './DecisionBadge';

describe('DecisionBadge Component', () => {
  it('renders ESCALATE action badge', () => {
    render(<DecisionBadge action="ESCALATE" />);
    const badge = screen.getByText('ESCALATE');
    expect(badge).toBeDefined();
  });

  it('renders SUPPRESS action badge', () => {
    render(<DecisionBadge action="SUPPRESS" />);
    const badge = screen.getByText('SUPPRESS');
    expect(badge).toBeDefined();
  });
});
