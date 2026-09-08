import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ProcessingTrace } from './ProcessingTrace';

describe('ProcessingTrace', () => {
  it('uses the page scroll instead of creating its own vertical scroll area', () => {
    const { container } = render(<ProcessingTrace trace={[{ timestamp: '10:00:00', stage: 'RBTA', message: 'Bucket diperbarui' }]} />);

    expect(screen.getByText('Bucket diperbarui')).toBeInTheDocument();
    expect(container.querySelector('.overflow-y-auto')).not.toBeInTheDocument();
  });
});
