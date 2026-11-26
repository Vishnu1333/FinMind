/** Formatting helpers – the product currency is Malaysian Ringgit (MYR). */

export const CURRENCY = 'MYR';

const myrFormatter = new Intl.NumberFormat('en-MY', {
  style: 'currency',
  currency: 'MYR',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const myrCompactFormatter = new Intl.NumberFormat('en-MY', {
  style: 'currency',
  currency: 'MYR',
  notation: 'compact',
  maximumFractionDigits: 1,
});

export function formatMYR(amount: number | string | null | undefined, compact = false): string {
  if (amount === null || amount === undefined) return '—';
  const value = typeof amount === 'string' ? parseFloat(amount) : amount;
  if (Number.isNaN(value)) return '—';
  return compact ? myrCompactFormatter.format(value) : myrFormatter.format(value);
}

export function formatSignedMYR(amount: number | string | null | undefined): string {
  if (amount === null || amount === undefined) return '—';
  const value = typeof amount === 'string' ? parseFloat(amount) : amount;
  if (Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${myrFormatter.format(value)}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-MY', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-MY', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function titleCase(text: string): string {
  return text
    .split(/[\s_]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}
