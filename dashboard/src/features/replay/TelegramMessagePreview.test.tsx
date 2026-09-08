import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TelegramMessagePreview } from './TelegramMessagePreview';

describe('TelegramMessagePreview', () => {
  it('renders the supported Telegram HTML tags instead of showing them as text', () => {
    render(<TelegramMessagePreview message={'<b>Decision:</b> CRITICAL\n<code>rbta-if-v1</code>\n<i>Bukan bukti serangan.</i>'} />);

    expect(screen.getByText('Decision:').tagName).toBe('B');
    expect(screen.getByText('rbta-if-v1').tagName).toBe('CODE');
    expect(screen.getByText('Bukan bukti serangan.').tagName).toBe('I');
    expect(screen.queryByText(/<b>Decision:<\/b>/)).not.toBeInTheDocument();
  });

  it('keeps unsupported markup as plain text', () => {
    render(<TelegramMessagePreview message={'<script>alert(1)</script>'} />);

    expect(screen.getByText('<script>alert(1)</script>')).toBeInTheDocument();
  });
});
