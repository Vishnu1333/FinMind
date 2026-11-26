'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ShieldAlert, Search, Cpu } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { formatMYR, formatDate } from '@/lib/format';
import { Card, CardHeader, Badge, Button, Input, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface Anomaly {
  transaction_id: number;
  date: string;
  type: string;
  amount: number;
  merchant: string | null;
  category: string | null;
  anomaly_score: number;
  reasons: string[];
  evidence: {
    business_avg_spend: number;
    business_spend_std: number;
    amount_z_score: number | null;
    hour: number | null;
    duplicate: boolean;
  };
}

interface AnomalyData {
  available: boolean;
  reason: string | null;
  anomalies: Anomaly[];
  model: {
    type: string;
    n_estimators: number;
    contamination: number;
    trained_on: number;
    features: string[];
    evidence_rules: string[];
  } | null;
}

function reasonTone(reason: string): 'negative' | 'caution' {
  return reason.includes('duplicate') || reason.includes('standard deviations') ? 'negative' : 'caution';
}

export default function AnomaliesPage() {
  const anomalies = useApi<AnomalyData>('/predictions/anomalies?top_n=25');
  const [checkAmount, setCheckAmount] = useState('');
  const [checkLoading, setCheckLoading] = useState(false);
  const [checkResult, setCheckResult] = useState<
    { available: boolean; is_anomaly: boolean | null; score: number | null; reason: string | null } | null
  >(null);
  const [checkError, setCheckError] = useState('');

  const handleCheck = async (e: React.FormEvent) => {
    e.preventDefault();
    const amount = parseFloat(checkAmount);
    if (Number.isNaN(amount)) return;
    setCheckLoading(true);
    setCheckError('');
    setCheckResult(null);
    try {
      const result = await api.post<{
        available: boolean;
        is_anomaly: boolean | null;
        score: number | null;
        reason: string | null;
      }>('/predictions/anomaly', { amount });
      setCheckResult(result);
    } catch (err) {
      setCheckError(
        apiErrorMessage(err, 'anomaly check'),
      );
    } finally {
      setCheckLoading(false);
    }
  };

  if (anomalies.loading) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Anomalies"
          subtitle="Unusual activity detected by an isolation-forest model trained on your own history."
        />
        <div className="space-y-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    );
  }

  if (anomalies.error) {
    return (
      <ErrorState
        message={apiErrorMessage(anomalies.error, 'anomaly detection')}
        onRetry={anomalies.refetch}
      />
    );
  }

  const data = anomalies.data;
  if (!data?.available) {
    return (
      <Card>
        <EmptyState
          icon={<Cpu className="w-6 h-6" />}
          title="Anomaly detection not available yet"
          description={data?.reason ?? 'More transaction history is needed before anomalies can be detected.'}
        />
      </Card>
    );
  }

  const flags = data.anomalies;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Anomalies"
        subtitle="Unusual activity detected by an isolation-forest model trained on your own history."
        actions={<Badge tone="neutral">Model trained on {data.model?.trained_on} transactions</Badge>}
      />

      {flags.length === 0 ? (
        <Card>
          <EmptyState
            icon={<ShieldAlert className="w-6 h-6" />}
            title="Nothing unusual detected"
            description="No transactions were flagged in the current analysis. This page updates as new activity is recorded."
          />
        </Card>
      ) : (
        <div className="space-y-3 animate-fade-up">
          {flags.map((a, i) => (
            <Card key={a.transaction_id} hover className={cn(i < 5 && `animate-fade-up stagger-${Math.min(i + 1, 5)}`)}>
              <div className="p-5 flex flex-col sm:flex-row gap-4">
                <div className="sm:w-40 shrink-0">
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-400">
                    {formatDate(a.date)}
                  </p>
                  <p
                    className={cn(
                      'tabular text-[21px] font-semibold mt-1',
                      a.type === 'debit' ? 'text-ink-950' : 'text-positive-600',
                    )}
                  >
                    {a.type === 'debit' ? '−' : '+'}
                    {formatMYR(a.amount)}
                  </p>
                  <p className="text-[12px] text-ink-500 mt-0.5">{a.category ?? 'Uncategorised'}</p>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[14px] font-medium text-ink-950">
                    {a.merchant ?? 'Unknown merchant'}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {a.reasons.map((reason) => (
                      <Badge key={reason} tone={reasonTone(reason)}>
                        {reason}
                      </Badge>
                    ))}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-ink-500">
                    <span>
                      Typical spend:{' '}
                      <span className="tabular font-medium text-ink-700">
                        {formatMYR(a.evidence.business_avg_spend)}
                      </span>
                    </span>
                    {a.evidence.amount_z_score !== null && (
                      <span>
                        Deviation:{' '}
                        <span className="tabular font-medium text-ink-700">
                          {a.evidence.amount_z_score.toFixed(1)}σ
                        </span>
                      </span>
                    )}
                    {a.evidence.hour !== null && (
                      <span>
                        Time:{' '}
                        <span className="tabular font-medium text-ink-700">
                          {String(a.evidence.hour).padStart(2, '0')}:00
                        </span>
                      </span>
                    )}
                    <span className="tabular">Model score {a.anomaly_score}</span>
                  </div>
                </div>
                <Link
                  href={`/transactions?search=${encodeURIComponent(a.merchant ?? '')}`}
                  className="self-center shrink-0"
                >
                  <Button variant="ghost" size="sm">
                    <Search className="w-3.5 h-3.5" />
                    Find in transactions
                  </Button>
                </Link>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Single transaction check */}
      <Card className="animate-fade-up stagger-3">
        <CardHeader
          title="Check a transaction amount"
          subtitle="Score any hypothetical amount against your business's spending distribution"
        />
        <form onSubmit={handleCheck} className="px-6 pb-6 flex flex-col sm:flex-row gap-3">
          <Input
            type="number"
            step="0.01"
            min="0"
            required
            placeholder="Amount, e.g. 2500"
            value={checkAmount}
            onChange={(e) => setCheckAmount(e.target.value)}
            className="sm:w-64"
          />
          <Button type="submit" loading={checkLoading} className="shrink-0">
            <Cpu className="w-4 h-4" />
            Analyse
          </Button>
        </form>
        {checkError && (
          <p className="px-6 pb-5 text-[13px] text-negative-600">{checkError}</p>
        )}
        {checkResult && !checkResult.available && (
          <p className="px-6 pb-5 text-[13px] text-ink-500">{checkResult.reason}</p>
        )}
        {checkResult?.available && (
          <div className="px-6 pb-6">
            <div
              className={cn(
                'rounded-xl border p-4 animate-fade-in',
                checkResult.is_anomaly
                  ? 'border-negative-600/20 bg-negative-50/60'
                  : 'border-positive-600/20 bg-positive-50/60',
              )}
            >
              <p
                className={cn(
                  'text-[14px] font-semibold',
                  checkResult.is_anomaly ? 'text-negative-600' : 'text-positive-600',
                )}
              >
                {checkResult.is_anomaly
                  ? 'This amount stands out from your normal spending pattern.'
                  : 'This amount looks consistent with your normal spending pattern.'}
              </p>
              <p className="text-[12.5px] text-ink-500 mt-1 tabular">
                Model score: {checkResult.score} (lower means more unusual)
              </p>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
