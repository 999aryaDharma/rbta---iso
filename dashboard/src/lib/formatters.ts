export function formatNumber(num: number | null | undefined): string {
  if (num === null || num === undefined) return '0';
  return num.toLocaleString();
}

export function formatDateTime(isoString: string | null | undefined): string {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return String(isoString);
    const formatter = new Intl.DateTimeFormat('sv-SE', {
      timeZone: 'Asia/Makassar',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
    return `${formatter.format(d)} WITA`;
  } catch {
    return String(isoString);
  }
}

export function formatScore(score: number | null | undefined, precision: number = 4): string {
  if (score === null || score === undefined) return '—';
  return score.toFixed(precision);
}

export function formatSeconds(secs: number | null | undefined): string {
  if (secs === null || secs === undefined) return '—';
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const m = Math.floor(secs / 60);
  const rem = secs % 60;
  return `${m}m ${rem.toFixed(0)}s`;
}

export function formatDuration(secs: number | null | undefined): string {
  if (secs === null || secs === undefined) return '0s';
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const mins = Math.floor(secs / 60);
  const remSecs = Math.floor(secs % 60);
  if (mins < 60) return `${mins}m ${remSecs}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${remMins}m ${remSecs}s`;
}
