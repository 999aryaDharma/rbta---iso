import { apiFetch } from './client';

export async function checkAuth(): Promise<boolean> {
  try {
    const data = await apiFetch<{ authenticated: boolean }>('/auth/check');
    return Boolean(data && data.authenticated);
  } catch {
    return false;
  }
}
