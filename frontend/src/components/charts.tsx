'use client';

export const CHART_COLORS = {
  income: '#10B981',
  expenses: '#EF4444',
  net: '#4F46E5',
  forecast: '#7C3AED',
  primary: '#4F46E5',
  palette: [
    '#4F46E5',
    '#7C3AED',
    '#0EA5E9',
    '#10B981',
    '#F59E0B',
    '#EF4444',
    '#EC4899',
    '#14B8A6',
    '#8B5CF6',
    '#64748B',
  ],
};

export const axisProps = {
  stroke: 'transparent',
  tick: { fill: '#8794AE', fontSize: 11 },
  tickLine: false,
  axisLine: false,
};

export const gridProps = {
  stroke: '#EBEEF3',
  strokeDasharray: '4 4',
  vertical: false,
};

interface TooltipEntry {
  name?: string | number;
  value?: string | number;
  color?: string;
}

/** Consistent white-card tooltip for every chart. */
export function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
  formatter?: (value: number) => string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-xl bg-white border border-ink-950/[0.08] shadow-[var(--shadow-pop)] px-3.5 py-2.5">
      {label !== undefined && (
        <p className="text-[11px] font-semibold text-ink-400 uppercase tracking-wide mb-1.5">
          {label}
        </p>
      )}
      <div className="space-y-1">
        {payload.map((entry: TooltipEntry, i: number) => (
          <div key={i} className="flex items-center justify-between gap-4 text-[12.5px]">
            <span className="flex items-center gap-1.5 text-ink-600">
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: (entry.color as string) || '#4F46E5' }}
              />
              {entry.name}
            </span>
            <span className="tabular font-semibold text-ink-950">
              {formatter
                ? formatter(Number(entry.value))
                : Number(entry.value).toLocaleString('en-MY')}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

