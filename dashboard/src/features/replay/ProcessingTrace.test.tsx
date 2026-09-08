import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ProcessingTrace } from './ProcessingTrace';

describe('ProcessingTrace', () => {
  it('shows only the newest 10 events in a compact scroll area', () => {
    const trace = Array.from({ length: 12 }, (_, index) => ({
      timestamp: `10:00:${String(index).padStart(2, '0')}`,
      stage: 'RBTA',
      message: `Kejadian ${index + 1}`,
    }));
    const { container } = render(<ProcessingTrace trace={trace} />);

    expect(screen.queryByText('Kejadian 1')).not.toBeInTheDocument();
    expect(screen.queryByText('Kejadian 2')).not.toBeInTheDocument();
    expect(screen.getByText('Kejadian 3')).toBeInTheDocument();
    expect(screen.getByText('Kejadian 12')).toBeInTheDocument();
    expect(container.querySelector('.overflow-y-auto')).toBeInTheDocument();
  });
});
