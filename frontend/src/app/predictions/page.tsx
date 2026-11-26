'use client';

import { useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Cpu } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR, formatSignedMYR } from '@/lib/format';
import { Card, CardHeader, Badge, Tabs, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';
import { CHART_COLORS, ChartTooltip, axisProps, gridProps } from '@/components/charts';

interface ForecastData {
  currency: string;
  forecast: { day: number; date: string; predicted_cashflow: number; cumulative: number }[];
  model: {
    selected: string;
    selection_method: string;
    trained_on_days: number;
    holdout_days: number;
    holdout_mae_scaled: number;
    holdout_rmse_scaled: number;
    candidates: Record<string, { holdout_mae_scaled: number; holdout_rmse_scaled: number }>;
    series_mean: number;
    series_std: number;
  };
}

interface DailyData {
  days: number;
  series: { date: string; income: string; expenses: string; net: string }[];
}

const HORIZONS = [
  { id: '14', label: '14 days' },
  { id: '30', label: '30 days' },
  { id: '60', label: '60 days' },
  { id: '90', label: '90 days' },
];

const CANDIDATE_LABELS: Record<string, string> = {
  lstm: 'LSTM',
  gru: 'GRU',
  ar14: 'AR(14)',
};

export default function PredictionsPage() {
  const [days, setDays] = useState('30');
  const forecast = useApi<ForecastData>(`/predictions/forecast?days=${days}`);
  const history = useApi<DailyData>(`/analytics/cash-flow/daily?days=30`);

  if (forecast.loading) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Predictions"
          subtitle="Cash-flow forecasts from a model trained on your business's real history."
        />
        <div className="flex items-center gap-3 text-[13px] text-ink-500">
          <Cpu className="w-4 h-4" />
          Training forecast models on your transaction history…
        </div>
        <Card className="p-6">
          <Skeleton className="h-72" />
        </Card>
      </div>
    );
  }

  if (forecast.error || !forecast.data) {
    return (
      <Card>
        <EmptyState
          icon={<Cpu className="w-6 h-6" />}
          title="Forecast not available yet"
          description={
            forecast.error
              ? forecast.error
              : 'Forecasts need about two months of transaction history to train on. Keep transacting and check back.'
          }
        />
      </Card>
    );
  }

  const model = forecast.data.model;
  const projectedNet = forecast.data.forecast.reduce((s, p) => s + p.predicted_cashflow, 0);
  const positiveDays = forecast.data.forecast.filter((p) => p.predicted_cashflow > 0).length;
  const negativeDays = forecast.data.forecast.length - positiveDays;

  // Combine: last 30 days of actuals + forecast horizon.
  const actualSeries = (history.data?.series ?? []).map((p) => ({
    label: new Date(p.date).toLocaleDateString('en-MY', { month: 'short', day: 'numeric' }),
    Actual: parseFloat(p.net),
    Predicted: null as number | null,
  }));
  const forecastSeries = forecast.data.forecast.map((p) => ({
    label: new Date(p.date).toLocaleDateString('en-MY', { month: 'short', day: 'numeric' }),
    Actual: null as number | null,
    Predicted: p.predicted_cashflow,
  }));
  const combined = [...actualSeries, ...forecastSeries];

  const candidateEntries = Object.entries(model.candidates).sort(
    (a, b) => a[1].holdout_mae_scaled - b[1].holdout_mae_scaled,
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Predictions"
        subtitle="Cash-flow forecasts from a model trained on your business's real history."
        actions={
          <Tabs
            tabs={HORIZONS.map((h) => ({ id: h.id, label: h.label }))}
            active={days}
            onChange={setDays}
          />
        }
      />

      {/* Headline stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 animate-fade-up">
        <Card className="p-5">
          <p className="text-[12.5px] font-medium text-ink-500 uppercase tracking-wide">
            Projected net · {days} days
          </p>
          <p
            className={cn(
              'tabular text-[26px] font-semibold mt-2',
              projectedNet >= 0 ? 'text-positive-600' : 'text-negative-600',
            )}
          >
            {formatSignedMYR(projectedNet)}
          </p>
        </Card>
        <Card className="p-5">
          <p className="text-[12.5px] font-medium text-ink-500 uppercase tracking-wide">
            Positive days ahead
          </p>
          <p className="tabular text-[26px] font-semibold text-ink-950 mt-2">
            {positiveDays}
            <span className="text-[15px] font-medium text-ink-400"> / {positiveDays + negativeDays}</span>
          </p>
        </Card>
        <Card className="p-5">
          <p className="text-[12.5px] font-medium text-ink-500 uppercase tracking-wide">Selected model</p>
          <div className="mt-2.5">
            <Badge tone="primary">{model.selected}</Badge>
          </div>
          <p className="text-[12px] text-ink-500 mt-2">Chosen by holdout accuracy</p>
        </Card>
      </div>

      {/* Chart */}
      <Card className="animate-fade-up stagger-1">
        <CardHeader
          title={`Forecast — next ${days} days`}
          subtitle="Solid line: actual daily net · Dashed line: model prediction"
          action={<Badge tone="ink">Model prediction</Badge>}
        />
        {history.error && actualSeries.length === 0 ? (
          <ErrorState
            message={apiErrorMessage(history.error, 'recent cash flow')}
            onRetry={history.refetch}
          />
        ) : (
          <div className="px-3 pb-5">
            <div className="h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={combined} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.net} stopOpacity={0.16} />
                      <stop offset="100%" stopColor={CHART_COLORS.net} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={44} />
                  <YAxis {...axisProps} tickFormatter={(v) => formatMYR(v, true)} width={72} />
                  <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                  <ReferenceLine y={0} stroke="#B4BECE" strokeDasharray="2 4" />
                  <Area
                    type="monotone"
                    dataKey="Actual"
                    name="Actual net"
                    stroke={CHART_COLORS.net}
                    strokeWidth={2}
                    fill="url(#actualFill)"
                    connectNulls={false}
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="Predicted"
                    name="Predicted net"
                    stroke={CHART_COLORS.forecast}
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    fill="none"
                    connectNulls={false}
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </Card>

      {/* Model provenance */}
      <Card className="animate-fade-up stagger-2">
        <CardHeader
          title="Model provenance"
          subtitle="Exactly how this forecast was produced"
        />
        <div className="px-6 pb-6 grid grid-cols-1 md:grid-cols-2 gap-x-10 gap-y-4">
          <div className="space-y-3 text-[13px]">
            <Row label="Training data" value={`${model.trained_on_days} days of your transactions`} />
            <Row label="Selection method" value="Walk-forward holdout error (MAE)" />
            <Row label="Holdout window" value={`${model.holdout_days} days`} />
            <Row
              label="Holdout MAE (scaled)"
              value={model.holdout_mae_scaled.toFixed(3)}
            />
            <Row
              label="Holdout RMSE (scaled)"
              value={model.holdout_rmse_scaled.toFixed(3)}
            />
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-400 mb-2.5">
              Candidate models evaluated
            </p>
            <div className="space-y-2">
              {candidateEntries.map(([key, m]) => (
                <div
                  key={key}
                  className={cn(
                    'flex items-center justify-between rounded-xl border px-3.5 py-2.5 text-[13px]',
                    CANDIDATE_LABELS[key]?.toLowerCase() === model.selected.toLowerCase() || key === model.selected.toLowerCase()
                      ? 'border-primary-600/25 bg-primary-50/60'
                      : 'border-ink-950/[0.07] bg-white',
                  )}
                >
                  <span className="font-medium text-ink-900">
                    {CANDIDATE_LABELS[key] ?? key}
                    {(CANDIDATE_LABELS[key] ?? key) === model.selected && (
                      <Badge tone="primary" className="ml-2">
                        Selected
                      </Badge>
                    )}
                  </span>
                  <span className="tabular text-ink-500">MAE {m.holdout_mae_scaled.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-ink-950/[0.05] pb-2.5">
      <span className="text-ink-500">{label}</span>
      <span className="tabular font-medium text-ink-900 text-right">{value}</span>
    </div>
  );
}
