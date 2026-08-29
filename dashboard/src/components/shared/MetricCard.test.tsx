import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricCard } from './MetricCard';

describe('MetricCard Component', () => {
  it('renders label, value, and subtitle', () => {
    render(<MetricCard label="Raw Events" value="12,500" sub="Last 24 hours" />);
    expect(screen.getByText('Raw Events')).toBeDefined();
    expect(screen.getByText('12,500')).toBeDefined();
    expect(screen.getByText('Last 24 hours')).toBeDefined();
  });
});
