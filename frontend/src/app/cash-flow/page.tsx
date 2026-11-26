'use client';

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { TrendingUp, CalendarClock, LineChart } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR, formatSignedMYR } from '@/lib/format';
import { Card, CardHeader, Stat, Skeleton } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';
import { CHART_COLORS, ChartTooltip, axisProps, gridProps } from '@/components/charts';

interface CashFlowData {
  period_start: string;
  period_end: string;
  income: string;
  expenses: string;
  net: string;
}

interface RunwayData {
  months: string | null;
  insufficient_data: boolean;
  message: string | null;
}

interface DailyData {
  days: number;
  series: { date: string; income: string; expenses: string; net: string }[];
}

interface TrendData {
  weeks: number;
  series: { week: string; income: string; expenses: string; net: string }[];
}

export default function CashFlowPage() {
  const cashFlow = useApi<CashFlowData>('/analytics/cash-flow');
  const runway = useApi<RunwayData>('/analytics/runway');
  const daily = useApi<DailyData>('/analytics/cash-flow/daily?days=90');
  const trend = useApi<TrendData>('/analytics/trends?weeks=12');

  const loading = cashFlow.loading || daily.loading || trend.loading;
  if (loading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Cash Flow" subtitle="How money moves through your business." />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Card className="p-6">
          <Skeleton className="h-72" />
        </Card>
      </div>
    );
  }

  if (cashFlow.error || !cashFlow.data) {
    return (
      <ErrorState
        message={apiErrorMessage(cashFlow.error, 'cash flow data')}
        onRetry={cashFlow.refetch}
      />
    );
  }

  const income = parseFloat(cashFlow.data.income);
  const expenses = parseFloat(cashFlow.data.expenses);
  const net = parseFloat(cashFlow.data.net);
  const runwayMonths =
    runway.data && !runway.data.insufficient_data && runway.data.months
      ? parseFloat(runway.data.months)
      : null;

  const dailySeries = (daily.data?.series ?? []).map((p) => ({
    label: new Date(p.date).toLocaleDateString('en-MY', { month: 'short', day: 'numeric' }),
    net: parseFloat(p.net),
  }));

  const trendSeries = (trend.data?.series ?? []).map((p) => ({
    label: `W/C ${new Date(p.week).toLocaleDateString('en-MY', { day: 'numeric', month: 'short' })}`,
    Income: parseFloat(p.income),
    Expenses: parseFloat(p.expenses),
  }));

  return (
    <div className="space-y-6">
      <PageHeader
        title="Cash Flow"
        subtitle="Income, expenses and net movement — daily and weekly."
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 animate-fade-up">
        <Stat
          label="Income · all time"
          value={formatMYR(income)}
          tone="positive"
          icon={<TrendingUp className="w-4 h-4" />}
        />
        <Stat
          label="Expenses · all time"
          value={formatMYR(expenses)}
          tone="negative"
          icon={<TrendingDownIcon />}
        />
        <Stat
          label="Net cash flow"
          value={formatSignedMYR(net)}
          tone={net >= 0 ? 'positive' : 'negative'}
          icon={<LineChart className="w-4 h-4" />}
        />
      </div>

      {/* Runway */}
      <Card className="p-6 animate-fade-up stagger-1">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <span className="w-11 h-11 rounded-xl bg-primary-50 text-primary-600 flex items-center justify-center shrink-0">
            <CalendarClock className="w-5 h-5" />
          </span>
          <div className="flex-1">
            <h3 className="text-[15px] font-semibold text-ink-950">Cash runway</h3>
            {runwayMonths !== null ? (
              <>
                <p className="tabular text-[30px] font-semibold tracking-tight text-ink-950 mt-1">
                  {runwayMonths.toFixed(1)}
                  <span className="text-[15px] font-medium text-ink-500"> months</span>
                </p>
                <p className="text-[12.5px] text-ink-500 mt-0.5">
                  Estimated time before cash depletion at your current average burn rate.
                </p>
              </>
            ) : (
              <p className="text-[13px] text-ink-500 mt-1">
                {runway.data?.message ??
                  'Not enough expense history to compute runway yet.'}
              </p>
            )}
          </div>
        </div>
      </Card>

      {/* Daily net chart */}
      <Card className="animate-fade-up stagger-2">
        <CardHeader
          title="Daily net cash flow — last 90 days"
          subtitle="Money in minus money out, per day"
        />
        <div className="px-3 pb-5">
          {daily.error ? (
            <ErrorState message={apiErrorMessage(daily.error, 'daily cash flow')} onRetry={daily.refetch} />
          ) : dailySeries.length === 0 ? (
            <EmptyState
              icon={<LineChart className="w-6 h-6" />}
              title="No cash movement yet"
              description="Daily cash flow appears as your accounts record activity."
            />
          ) : (
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dailySeries} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="dailyNet" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.net} stopOpacity={0.2} />
                      <stop offset="100%" stopColor={CHART_COLORS.net} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={40} />
                  <YAxis {...axisProps} tickFormatter={(v) => formatMYR(v, true)} width={72} />
                  <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                  <ReferenceLine y={0} stroke="#B4BECE" strokeDasharray="2 4" />
                  <Area
                    type="monotone"
                    dataKey="net"
                    name="Net"
                    stroke={CHART_COLORS.net}
                    strokeWidth={2}
                    fill="url(#dailyNet)"
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </Card>

      {/* Weekly trend */}
      <Card className="animate-fade-up stagger-3">
        <CardHeader
          title="Weekly income vs expenses"
          subtitle="Last 12 weeks — spot the pattern behind your cash balance"
        />
        <div className="px-3 pb-5">
          {trend.error ? (
            <ErrorState message={apiErrorMessage(trend.error, 'weekly trends')} onRetry={trend.refetch} />
          ) : trendSeries.length === 0 ? (
            <EmptyState
              icon={<TrendingUp className="w-6 h-6" />}
              title="Not enough history yet"
              description="Weekly trends appear after your first week of activity."
            />
          ) : (
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trendSeries} margin={{ top: 8, right: 12, left: 0, bottom: 0 }} barGap={2}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={24} />
                  <YAxis {...axisProps} tickFormatter={(v) => formatMYR(v, true)} width={72} />
                  <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                  <Bar dataKey="Income" fill={CHART_COLORS.income} radius={[4, 4, 0, 0]} maxBarSize={26} />
                  <Bar dataKey="Expenses" fill={CHART_COLORS.expenses} radius={[4, 4, 0, 0]} maxBarSize={26} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

function TrendingDownIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
      <path strokeLinecap="round" strokeLinejoin="round" d="m17 7 5 5-5 5M22 12H8m-5 5 5 5-5 5" transform="rotate(180 12 12)" />
    </svg>
  );
}
