'use client';

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { TrendingUp, TrendingDown, Store } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR } from '@/lib/format';
import { Card, CardHeader, Skeleton } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';
import { CHART_COLORS, ChartTooltip, axisProps, gridProps } from '@/components/charts';

interface SpendingData {
  currency: string;
  breakdown: { category_id: number; category_name: string; amount: string; percentage: string }[];
  total: string;
}

interface TrendData {
  weeks: number;
  series: { week: string; income: string; expenses: string; net: string }[];
}

interface MerchantsData {
  currency: string;
  merchants: { merchant: string; total: string; count: number }[];
}

export default function AnalyticsPage() {
  const spending = useApi<SpendingData>('/analytics/spending');
  const trend = useApi<TrendData>('/analytics/trends?weeks=12');
  const merchants = useApi<MerchantsData>('/analytics/merchants?limit=8');

  const loading = spending.loading || trend.loading || merchants.loading;
  if (loading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Analytics" subtitle="Understand where your money goes." />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {[0, 1].map((i) => (
            <Card key={i} className="p-6">
              <Skeleton className="h-64" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (spending.error || !spending.data) {
    return (
      <ErrorState
        message={apiErrorMessage(spending.error, 'spending analytics')}
        onRetry={spending.refetch}
      />
    );
  }

  const breakdown = spending.data.breakdown;
  const totalSpend = parseFloat(spending.data.total);
  const pieData = breakdown.slice(0, 8).map((b) => ({
    name: b.category_name,
    value: parseFloat(b.amount),
  }));

  const trendSeries = (trend.data?.series ?? []).map((p) => ({
    label: `W/C ${new Date(p.week).toLocaleDateString('en-MY', { day: 'numeric', month: 'short' })}`,
    Income: parseFloat(p.income),
    Expenses: parseFloat(p.expenses),
  }));

  const merchantSeries = (merchants.data?.merchants ?? []).map((m) => ({
    label: m.merchant,
    Spend: parseFloat(m.total),
  }));

  const maxCategoryPct = breakdown.length
    ? Math.max(...breakdown.map((b) => parseFloat(b.percentage)))
    : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analytics"
        subtitle="Category breakdown, merchant analysis and weekly trends."
      />

      {/* Spending by category */}
      <Card className="animate-fade-up">
        <CardHeader
          title="Spending by category"
          subtitle="All-time expenses, grouped by category"
        />
        {breakdown.length === 0 ? (
          <EmptyState
            icon={<TrendingDown className="w-6 h-6" />}
            title="No spending recorded"
            description="Category analysis appears once you have expense transactions."
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 px-6 pb-6">
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={105}
                    paddingAngle={2.5}
                    dataKey="value"
                    stroke="none"
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS.palette[i % CHART_COLORS.palette.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-3 lg:pl-2">
              {breakdown.slice(0, 6).map((b, i) => {
                const pct = parseFloat(b.percentage);
                return (
                  <div key={b.category_id}>
                    <div className="flex items-center justify-between text-[13px] mb-1">
                      <span className="flex items-center gap-2 text-ink-700">
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ background: CHART_COLORS.palette[i % CHART_COLORS.palette.length] }}
                        />
                        {b.category_name}
                      </span>
                      <span className="tabular font-semibold text-ink-950">
                        {formatMYR(b.amount)}{' '}
                        <span className="text-ink-400 font-normal">· {pct.toFixed(1)}%</span>
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-ink-100 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${maxCategoryPct ? (pct / maxCategoryPct) * 100 : 0}%`,
                          background: CHART_COLORS.palette[i % CHART_COLORS.palette.length],
                        }}
                      />
                    </div>
                  </div>
                );
              })}
              <p className="pt-2 border-t border-ink-950/[0.06] text-[12.5px] text-ink-500">
                Total tracked spending: <span className="tabular font-semibold text-ink-800">{formatMYR(totalSpend)}</span>
              </p>
            </div>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Top merchants */}
        <Card className="animate-fade-up stagger-1">
          <CardHeader title="Top merchants" subtitle="Where your business spends the most" />
          {merchants.error ? (
            <ErrorState
              message={apiErrorMessage(merchants.error, 'merchant analytics')}
              onRetry={merchants.refetch}
            />
          ) : merchantSeries.length === 0 ? (
            <EmptyState
              icon={<Store className="w-6 h-6" />}
              title="No merchant spending yet"
              description="Top merchants appear as your business records expenses."
            />
          ) : (
            <div className="px-3 pb-5">
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={merchantSeries}
                    layout="vertical"
                    margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
                  >
                    <CartesianGrid {...gridProps} horizontal={true} vertical={false} />
                    <XAxis type="number" {...axisProps} tickFormatter={(v) => formatMYR(v, true)} />
                    <YAxis
                      type="category"
                      dataKey="label"
                      {...axisProps}
                      width={130}
                      tick={{ fill: '#5C6B8A', fontSize: 11.5 }}
                    />
                    <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} cursor={{ fill: 'rgba(16,23,40,0.03)' }} />
                    <Bar dataKey="Spend" fill={CHART_COLORS.primary} radius={[0, 5, 5, 0]} maxBarSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </Card>

        {/* Weekly trend */}
        <Card className="animate-fade-up stagger-2">
          <CardHeader title="Weekly trend" subtitle="Income vs expenses, last 12 weeks" />
          {trend.error ? (
            <ErrorState
              message={apiErrorMessage(trend.error, 'weekly trends')}
              onRetry={trend.refetch}
            />
          ) : trendSeries.length === 0 ? (
            <EmptyState
              icon={<TrendingUp className="w-6 h-6" />}
              title="Not enough history yet"
              description="Trends appear after your first weeks of activity."
            />
          ) : (
            <div className="px-3 pb-5">
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={trendSeries} margin={{ top: 4, right: 12, left: 0, bottom: 0 }} barGap={2}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={28} />
                    <YAxis {...axisProps} tickFormatter={(v) => formatMYR(v, true)} width={72} />
                    <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} cursor={{ fill: 'rgba(16,23,40,0.03)' }} />
                    <Bar dataKey="Income" fill={CHART_COLORS.income} radius={[4, 4, 0, 0]} maxBarSize={20} />
                    <Bar dataKey="Expenses" fill={CHART_COLORS.expenses} radius={[4, 4, 0, 0]} maxBarSize={20} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
