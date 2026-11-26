'use client';

import { useState } from 'react';
import { Wallet, Landmark, ChevronDown } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR, formatDate } from '@/lib/format';
import { Card, Badge, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface Account {
  id: number;
  name: string;
  balance: number;
  created_at: string;
  business_id: number;
}

interface Transaction {
  id: number;
  amount: number;
  merchant: string | null;
  timestamp: string;
  type: string;
  category_name: string | null;
}

function AccountCard({ account }: { account: Account }) {
  const [open, setOpen] = useState(false);
  const recent = useApi<Transaction[]>(open ? `/transactions?account_id=${account.id}&limit=5` : null);

  return (
    <Card hover className="overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="focus-ring w-full text-left p-6"
        aria-expanded={open}
      >
        <div className="flex items-start justify-between gap-3">
          <span className="w-11 h-11 rounded-xl bg-ink-950 text-white flex items-center justify-center">
            {account.name.toLowerCase().includes('saving') ? (
              <Landmark className="w-5 h-5" />
            ) : (
              <Wallet className="w-5 h-5" />
            )}
          </span>
          <Badge tone="neutral">Account #{account.id}</Badge>
        </div>
        <p className="mt-5 text-[15px] font-semibold text-ink-950">{account.name}</p>
        <p className="tabular text-[28px] font-semibold tracking-tight text-ink-950 mt-1.5">
          {formatMYR(account.balance)}
        </p>
        <p className="text-[12px] text-ink-500 mt-1">Open since {formatDate(account.created_at)}</p>
        <span className="mt-4 inline-flex items-center gap-1 text-[12.5px] font-medium text-primary-600">
          {open ? 'Hide recent activity' : 'View recent activity'}
          <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} />
        </span>
      </button>
      {open && (
        <div className="border-t border-ink-950/[0.06] bg-ink-50/40 animate-fade-in">
          {recent.loading ? (
            <div className="p-5 space-y-2">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-10" />
              ))}
            </div>
          ) : recent.error ? (
            <ErrorState message={apiErrorMessage(recent.error, 'account activity')} onRetry={recent.refetch} />
          ) : (recent.data ?? []).length === 0 ? (
            <p className="p-5 text-[13px] text-ink-500 text-center">No activity recorded yet.</p>
          ) : (
            <div className="divide-y divide-ink-950/[0.05]">
              {(recent.data ?? []).map((t) => (
                <div key={t.id} className="flex items-center justify-between gap-3 px-5 py-3">
                  <div className="min-w-0">
                    <p className="text-[13px] font-medium text-ink-900 truncate">
                      {t.merchant || 'Transaction'}
                    </p>
                    <p className="text-[11.5px] text-ink-500">{formatDate(t.timestamp)}</p>
                  </div>
                  <span
                    className={`tabular text-[13px] font-semibold ${
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
        </div>
      )}
    </Card>
  );
}

export default function AccountsPage() {
  const accounts = useApi<Account[]>('/accounts');

  return (
    <div className="space-y-6">
      <PageHeader
        title="Accounts"
        subtitle="Your business bank accounts and their current balances."
      />
      {accounts.loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {[0, 1].map((i) => (
            <Skeleton key={i} className="h-56" />
          ))}
        </div>
      ) : accounts.error ? (
        <ErrorState
          message={apiErrorMessage(accounts.error, 'accounts')}
          onRetry={accounts.refetch}
        />
      ) : (accounts.data ?? []).length === 0 ? (
        <Card>
          <EmptyState
            icon={<Wallet className="w-6 h-6" />}
            title="No accounts yet"
            description="Your business accounts will appear here once they are opened."
          />
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-fade-up">
          {(accounts.data ?? []).map((account) => (
            <AccountCard key={account.id} account={account} />
          ))}
        </div>
      )}
    </div>
  );
}
