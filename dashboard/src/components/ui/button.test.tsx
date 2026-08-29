import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Button } from './button';

describe('Button Component', () => {
  it('renders children and default variant correctly', () => {
    render(<Button>Investigate</Button>);
    const btn = screen.getByRole('button', { name: /investigate/i });
    expect(btn).toBeDefined();
    expect(btn.textContent).toBe('Investigate');
  });

  it('applies disabled attributes', () => {
    render(<Button disabled>Disabled Action</Button>);
    const btn = screen.getByRole('button', { name: /disabled action/i }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});
