'use client';

import Link from 'next/link';
import {
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  ArrowLeftRight,
  Sparkles,
  ShieldAlert,
  ChevronRight,
  CalendarClock,
} from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR, formatSignedMYR, formatDate } from '@/lib/format';
import { Card, CardHeader, Stat, Badge, Skeleton, Button } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';
import { CHART_COLORS, ChartTooltip, axisProps, gridProps } from '@/components/charts';

interface SummaryData {
  currency: string;
  accounts: { account_id: number; name: string; balance: string }[];
  total_balance: string;
  total_income: string;
  total_expenses: string;
  net_cash_flow: string;
  transaction_count?: number;
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

interface SpendingData {
  currency: string;
  breakdown: { category_id: number; category_name: string; amount: string; percentage: string }[];
  total: string;
}

interface ForecastData {
  currency: string;
  forecast: { day: number; date: string; predicted_cashflow: number; cumulative: number }[];
  model: {
    selected: string;
    trained_on_days: number;
    holdout_mae_scaled: number;
    candidates: Record<string, { holdout_mae_scaled: number }>;
  };
}

interface Transaction {
  id: number;
  amount: number;
  merchant: string | null;
  description: string | null;
  timestamp: string;
  type: string;
  category_name: string | null;
}

interface AnomalyData {
  available: boolean;
  anomalies: {
    transaction_id: number;
    date: string;
    amount: number;
    merchant: string | null;
    category: string | null;
    reasons: string[];
  }[];
}

export default function DashboardPage() {
  const summary = useApi<SummaryData>('/analytics/summary?days=30');
  const runway = useApi<RunwayData>('/analytics/runway');
  const daily = useApi<DailyData>('/analytics/cash-flow/daily?days=30');
  const spending = useApi<SpendingData>('/analytics/spending');
  const forecast = useApi<ForecastData>('/predictions/forecast?days=30');
  const transactions = useApi<Transaction[]>('/transactions?limit=5');
  const anomalies = useApi<AnomalyData>('/predictions/anomalies?top_n=3');

  const refetchAll = () => {
    summary.refetch();
    runway.refetch();
    daily.refetch();
    spending.refetch();
    forecast.refetch();
    transactions.refetch();
    anomalies.refetch();
  };

  const initialLoading = summary.loading && !summary.data;

  if (initialLoading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Dashboard" subtitle="Financial overview" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Card className="p-6">
          <Skeleton className="h-72" />
        </Card>
      </div>
    );
  }

  if (summary.error || !summary.data) {
    return (
      <div className="space-y-6">
        <PageHeader title="Dashboard" subtitle="Financial overview" />
        <ErrorState
          message={apiErrorMessage(summary.error, 'your financial overview')}
          onRetry={refetchAll}
        />
      </div>
    );
  }

  const hasAccounts = summary.data.accounts && summary.data.accounts.length > 0;
  const hasTransactions =
    (transactions.data && transactions.data.length > 0) ||
    (summary.data.transaction_count !== undefined && summary.data.transaction_count > 0);
  const hasActivity =
    parseFloat(summary.data.total_balance) !== 0 ||
    parseFloat(summary.data.total_income) !== 0 ||
    parseFloat(summary.data.total_expenses) !== 0;

  const isWorkspaceEmpty = !hasAccounts && !hasTransactions && !hasActivity;

  if (isWorkspaceEmpty) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Financial overview"
          subtitle="Your financial workspace is ready."
        />
        <Card className="p-12 text-center">
          <div className="max-w-md mx-auto space-y-4">
            <div className="w-12 h-12 rounded-2xl brand-gradient text-white flex items-center justify-center mx-auto shadow-md">
              <Sparkles className="w-6 h-6" />
            </div>
            <h3 className="text-[18px] font-semibold text-ink-950">
              Your financial workspace is ready.
            </h3>
            <p className="text-[13.5px] text-ink-500 leading-relaxed">
              Connect an account or add transactions to see your financial overview.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
              <Link href="/accounts">
                <Button variant="primary" size="sm">
                  <Wallet className="w-4 h-4 mr-1.5" />
                  View Accounts
                </Button>
              </Link>
              <Link href="/transactions">
                <Button variant="secondary" size="sm">
                  <ArrowLeftRight className="w-4 h-4 mr-1.5" />
                  View Transactions
                </Button>
              </Link>
              <Link href="/assistant">
                <Button variant="secondary" size="sm">
                  <Sparkles className="w-4 h-4 mr-1.5" />
                  Ask FinMind AI
                </Button>
              </Link>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  const totalIncome = parseFloat(summary.data.total_income);
  const totalExpenses = parseFloat(summary.data.total_expenses);
  const netFlow = parseFloat(summary.data.net_cash_flow);
  const runwayMonths =
    runway.data && !runway.data.insufficient_data && runway.data.months
      ? parseFloat(runway.data.months)
      : null;

  const runwayTone =
    runwayMonths === null
      ? 'neutral'
      : runwayMonths < 3
      ? 'negative'
      : runwayMonths < 6
      ? 'caution'
      : 'positive';

  const dailySeries = (daily.data?.series ?? []).map((p) => ({
    ...p,
    netNum: parseFloat(p.net),
    label: new Date(p.date).toLocaleDateString('en-MY', { month: 'short', day: 'numeric' }),
  }));

  const spendingData = (spending.data?.breakdown ?? []).slice(0, 6).map((b) => ({
    name: b.category_name,
    value: parseFloat(b.amount),
  }));
  const spendingTotal = spending.data ? parseFloat(spending.data.total) : 0;

  const forecastProjected = forecast.data
    ? forecast.data.forecast.reduce((sum, p) => sum + p.predicted_cashflow, 0)
    : null;
  const modelSelected = forecast.data?.model?.selected;

  const topAnomalies = anomalies.data?.available ? (anomalies.data.anomalies ?? []).slice(0, 2) : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Financial overview"
        subtitle="Your business at a glance — balances, cash flow and what’s ahead."
      />

      {/* KPI row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4 animate-fade-up">
        <Stat
          label="Total balance"
          value={formatMYR(summary.data.total_balance)}
          sub={`${summary.data.accounts.length} account${summary.data.accounts.length === 1 ? '' : 's'} linked`}
          icon={<Wallet className="w-4 h-4" />}
        />
        <Stat
          label="Income · 30 days"
          value={formatMYR(totalIncome)}
          tone="positive"
          icon={<TrendingUp className="w-4 h-4" />}
        />
        <Stat
          label="Expenses · 30 days"
          value={formatMYR(totalExpenses)}
          tone="negative"
          icon={<TrendingDown className="w-4 h-4" />}
        />
        <Stat
          label="Net cash flow · 30 days"
          value={formatSignedMYR(netFlow)}
          tone={netFlow >= 0 ? 'positive' : 'negative'}
          sub={
            runwayMonths !== null ? (
              <span>
                Runway{' '}
                <span
                  className={
                    runwayTone === 'negative'
                      ? 'text-negative-600 font-semibold'
                      : runwayTone === 'caution'
                      ? 'text-caution-600 font-semibold'
                      : 'font-semibold text-ink-700'
                  }
                >
                  {runwayMonths.toFixed(1)} months
                </span>{' '}
                at current burn
              </span>
            ) : (
              'Runway needs more expense history'
            )
          }
          icon={<ArrowLeftRight className="w-4 h-4" />}
        />
      </div>

      {/* Upcoming risk */}
      {runwayMonths !== null && runwayMonths < 6 && (
        <Card className="p-4 flex items-start gap-3 border-caution-600/15 bg-caution-50/50 animate-fade-up stagger-1">
          <CalendarClock className="w-5 h-5 text-caution-600 mt-0.5 shrink-0" />
          <div className="text-[13px] text-ink-700">
            <span className="font-semibold text-caution-600">Cash runway attention:</span>{' '}
            at the current burn rate your balance covers about {runwayMonths.toFixed(1)} months
            {runwayMonths < 3 ? ' — consider reviewing recurring expenses soon.' : '.'}{' '}
            <Link href="/cash-flow" className="font-medium text-primary-600 hover:text-primary-700">
              Review cash flow
            </Link>
          </div>
        </Card>
      )}

      {/* Cash flow + spending */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        <Card className="xl:col-span-3 animate-fade-up stagger-1">
          <CardHeader
            title="Cash flow — last 30 days"
            subtitle="Daily net movement of money across your accounts"
            action={
              <Link href="/cash-flow">
                <Button variant="ghost" size="sm">
                  Details <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </Link>
            }
          />
          <div className="px-3 pb-4">
            {daily.loading ? (
              <div className="h-[260px] flex items-center justify-center p-4">
                <Skeleton className="h-full w-full" />
              </div>
            ) : daily.error ? (
              <ErrorState message={apiErrorMessage(daily.error, 'cash flow')} onRetry={daily.refetch} />
            ) : dailySeries.length === 0 ? (
              <EmptyState
                icon={<TrendingUp className="w-6 h-6" />}
                title="No cash-flow history yet."
                description="Transactions will appear here as soon as your accounts have activity."
              />
            ) : (
              <div className="h-[260px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={dailySeries} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="netFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={CHART_COLORS.net} stopOpacity={0.18} />
                        <stop offset="100%" stopColor={CHART_COLORS.net} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={32} />
                    <YAxis {...axisProps} tickFormatter={(v) => formatMYR(v, true)} width={72} />
                    <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                    <Area
                      type="monotone"
                      dataKey="netNum"
                      name="Net"
                      stroke={CHART_COLORS.net}
                      strokeWidth={2}
                      fill="url(#netFill)"
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </Card>

        <Card className="xl:col-span-2 animate-fade-up stagger-2">
          <CardHeader
            title="Spending by category"
            subtitle="Where your money went"
            action={
              <Link href="/analytics">
                <Button variant="ghost" size="sm">
                  Details <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </Link>
            }
          />
          {spending.loading ? (
            <div className="p-6">
              <Skeleton className="h-[180px]" />
            </div>
          ) : spending.error ? (
            <ErrorState message={apiErrorMessage(spending.error, 'spending')} onRetry={spending.refetch} />
          ) : spendingData.length === 0 ? (
            <EmptyState
              icon={<TrendingDown className="w-6 h-6" />}
              title="No spending recorded"
              description="Category insights appear once you have expense transactions."
            />
          ) : (
            <div className="px-6 pb-6">
              <div className="flex items-center gap-6">
                <div className="h-[180px] w-[180px] shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={spendingData}
                        cx="50%"
                        cy="50%"
                        innerRadius={58}
                        outerRadius={86}
                        paddingAngle={3}
                        dataKey="value"
                        stroke="none"
                      >
                        {spendingData.map((_, i) => (
                          <Cell key={i} fill={CHART_COLORS.palette[i % CHART_COLORS.palette.length]} />
                        ))}
                      </Pie>
                      <Tooltip content={<ChartTooltip formatter={(v) => formatMYR(v)} />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="min-w-0 flex-1 space-y-2">
                  {spendingData.slice(0, 4).map((item, i) => (
                    <div key={item.name} className="flex items-center justify-between gap-2 text-[13px]">
                      <span className="flex items-center gap-2 text-ink-600 min-w-0">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ background: CHART_COLORS.palette[i % CHART_COLORS.palette.length] }}
                        />
                        <span className="truncate">{item.name}</span>
                      </span>
                      <span className="tabular font-semibold text-ink-950 shrink-0">
                        {formatMYR(item.value, true)}
                      </span>
                    </div>
                  ))}
                  <p className="pt-2 border-t border-ink-950/[0.06] text-[12px] text-ink-500">
                    Total tracked spending: <span className="tabular font-semibold text-ink-800">{formatMYR(spendingTotal)}</span>
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Predictive intelligence + AI insights */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        <Card className="xl:col-span-3 animate-fade-up stagger-3">
          <CardHeader
            title="Predictive intelligence"
            subtitle="Next 30-day cash-flow projection"
            action={
              <Link href="/predictions">
                <Button variant="ghost" size="sm">
                  Details <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </Link>
            }
          />
          <div className="px-6 pb-6">
            {forecast.loading ? (
              <Skeleton className="h-28" />
            ) : forecast.error || !forecast.data || (forecast.data.forecast ?? []).length === 0 ? (
              <p className="text-[13px] text-ink-500 py-6">
                Forecasting needs a little more transaction history. Check back once your account
                has about two months of activity.
              </p>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[12.5px] font-medium text-ink-500 uppercase tracking-wide">
                      Projected net · next 30 days
                    </p>
                    <p
                      className={`tabular text-[26px] font-semibold mt-1 ${
                        (forecastProjected ?? 0) >= 0 ? 'text-positive-600' : 'text-negative-600'
                      }`}
                    >
                      {formatSignedMYR(forecastProjected ?? 0)}
                    </p>
                  </div>
                  <Badge tone="primary">{modelSelected} model</Badge>
                </div>
                <p className="text-[12.5px] text-ink-500 leading-relaxed">
                  Trained on {forecast.data?.model?.trained_on_days ?? 0} days of this business&rsquo;s
                  transactions. Selected among {Object.keys(forecast.data?.model?.candidates ?? {}).length}{' '}
                  candidate models by holdout accuracy.
                </p>
              </div>
            )}
          </div>
        </Card>

        <Card className="xl:col-span-2 animate-fade-up stagger-4">
          <CardHeader
            title="AI insights"
            subtitle="From your real financial data"
            action={
              <Link href="/anomalies">
                <Button variant="ghost" size="sm">
                  All <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </Link>
            }
          />
          <div className="px-6 pb-6 space-y-3">
            {anomalies.loading ? (
              <Skeleton className="h-20" />
            ) : topAnomalies.length > 0 ? (
              topAnomalies.map((a) => (
                <div
                  key={a.transaction_id}
                  className="flex items-start gap-3 rounded-xl bg-caution-50/60 border border-caution-600/10 p-3"
                >
                  <ShieldAlert className="w-4 h-4 text-caution-600 mt-0.5 shrink-0" />
                  <div className="text-[12.5px] text-ink-700 min-w-0">
                    <span className="font-semibold">
                      {formatMYR(a.amount)} to {a.merchant || 'an unknown merchant'}
                    </span>{' '}
                    — {a.reasons[0] ?? 'flagged as unusual'}.
                  </div>
                </div>
              ))
            ) : (
              <div className="flex items-start gap-3 rounded-xl bg-positive-50/60 border border-positive-600/10 p-3">
                <Sparkles className="w-4 h-4 text-positive-600 mt-0.5 shrink-0" />
                <p className="text-[12.5px] text-ink-700">
                  No unusual transactions detected in your recent activity.
                </p>
              </div>
            )}
            <Link
              href="/assistant"
              className="flex items-center justify-between rounded-xl border border-primary-600/15 bg-primary-50/50 p-3 hover:bg-primary-50 transition-colors"
            >
              <span className="text-[12.5px] text-ink-700">
                <Sparkles className="w-4 h-4 text-primary-600 inline mr-1.5 -mt-0.5" />
                Ask the assistant anything about these numbers
              </span>
              <ChevronRight className="w-4 h-4 text-primary-600" />
            </Link>
          </div>
        </Card>
      </div>

      {/* Recent transactions */}
      <Card className="animate-fade-up stagger-5">
        <CardHeader
          title="Recent transactions"
          subtitle="Latest activity across your accounts"
          action={
            <Link href="/transactions">
              <Button variant="ghost" size="sm">
                View all <ChevronRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          }
        />
        {transactions.loading ? (
          <div className="p-6 space-y-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-10" />
            ))}
          </div>
        ) : transactions.error ? (
          <ErrorState
            message={apiErrorMessage(transactions.error, 'transactions')}
            onRetry={transactions.refetch}
          />
        ) : (transactions.data ?? []).length === 0 ? (
          <EmptyState
            icon={<ArrowLeftRight className="w-6 h-6" />}
            title="No transactions yet"
            description="Your account activity will appear here."
          />
        ) : (
          <div className="divide-y divide-ink-950/[0.05]">
            {(transactions.data ?? []).map((t) => (
              <div key={t.id} className="flex items-center justify-between gap-4 px-6 py-3.5">
                <div className="min-w-0">
                  <p className="text-[13.5px] font-medium text-ink-950 truncate">
                    {t.merchant || t.description || 'Transaction'}
                  </p>
                  <p className="text-[12px] text-ink-500">
                    {formatDate(t.timestamp)}
                    {t.category_name ? ` · ${t.category_name}` : ''}
                  </p>
                </div>
                <span
                  className={`tabular text-[14px] font-semibold shrink-0 ${
                    t.type === 'credit' ? 'text-positive-600' : 'text-ink-950'
                  }`}
                >
                  {t.type === 'credit' ? '+' : '−'}
                  {formatMYR(t.amount)}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
