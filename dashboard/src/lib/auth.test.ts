import { describe, it, expect, beforeEach } from 'vitest';
import { getApiKey, setApiKey, clearApiKey, isAuthenticated } from './auth';

describe('Auth Storage Utilities', () => {
  beforeEach(() => {
    clearApiKey();
  });

  it('stores and retrieves session API key', () => {
    expect(isAuthenticated()).toBe(false);
    expect(getApiKey()).toBeNull();

    setApiKey('test-secret-key-456');
    expect(isAuthenticated()).toBe(true);
    expect(getApiKey()).toBe('test-secret-key-456');

    clearApiKey();
    expect(isAuthenticated()).toBe(false);
    expect(getApiKey()).toBeNull();
  });
});
