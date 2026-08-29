import { useQuery, type UseQueryOptions } from '@tanstack/react-query';

export function usePollingQuery<T>(
  key: string[],
  fetcher: () => Promise<T>,
  intervalMs: number,
  options?: Partial<UseQueryOptions<T>>,
) {
  return useQuery<T>({
    queryKey: key,
    queryFn: fetcher,
    refetchInterval: intervalMs,
    ...options,
  });
}
