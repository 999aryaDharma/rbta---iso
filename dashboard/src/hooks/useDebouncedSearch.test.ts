import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDebouncedSearch } from './useDebouncedSearch';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';

function wrapper({ children }: { children: React.ReactNode }) {
  return React.createElement(MemoryRouter, { initialEntries: ['/meta-alerts?search=initial'] }, children);
}

describe('useDebouncedSearch hook', () => {
  it('initializes from URL search parameter', () => {
    const { result } = renderHook(() => useDebouncedSearch('search', 200), { wrapper });
    expect(result.current.value).toBe('initial');
  });

  it('updates local value immediately when onChange is called', () => {
    const { result } = renderHook(() => useDebouncedSearch('search', 200), { wrapper });
    act(() => {
      result.current.onChange('new-search-term');
    });
    expect(result.current.value).toBe('new-search-term');
  });
});
