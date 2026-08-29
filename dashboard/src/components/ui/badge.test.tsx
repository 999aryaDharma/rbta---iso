import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge } from './badge';

describe('Badge Component', () => {
  it('renders badge label and applies font class', () => {
    render(<Badge variant="danger">ESCALATE</Badge>);
    const badge = screen.getByText('ESCALATE');
    expect(badge).toBeDefined();
    expect(badge.className).toContain('font-mono');
  });
});
