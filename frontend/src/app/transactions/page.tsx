'use client';

import { useMemo, useState } from 'react';
import { ArrowLeftRight, Search, ChevronLeft, ChevronRight } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { apiErrorMessage } from '@/lib/api';
import { formatMYR, formatDate } from '@/lib/format';
import { Card, Badge, Button, Input, Select, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface Transaction {
  id: number;
  account_id: number;
  amount: number;
  description: string | null;
  merchant: string | null;
  timestamp: string;
  type: string;
  category_id: number | null;
  category_name: string | null;
}

interface Category {
  id: number;
  name: string;
}

const PAGE_SIZE = 25;

export default function TransactionsPage() {
  const [search, setSearch] = useState('');
  const [type, setType] = useState('');
  const [category, setCategory] = useState('');
  const [page, setPage] = useState(0);

  const categories = useApi<Category[]>('/categories');

  const query = useMemo(() => {
    const params = new URLSearchParams();
    params.set('limit', String(PAGE_SIZE));
    params.set('skip', String(page * PAGE_SIZE));
    if (search.trim()) params.set('search', search.trim());
    if (type) params.set('type', type);
    if (category) params.set('category_id', category);
    return `/transactions?${params.toString()}`;
  }, [search, type, category, page]);

  const transactions = useApi<Transaction[]>(query, [query]);

  const filtersActive = !!search.trim() || !!type || !!category;
  const rows = transactions.data ?? [];
  const hasPrev = page > 0;
  const hasNext = rows.length === PAGE_SIZE;

  const resetFilters = () => {
    setSearch('');
    setType('');
    setCategory('');
    setPage(0);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Transactions"
        subtitle="Search, filter and review every movement of money."
      />

      {/* Filters */}
      <Card className="p-4 animate-fade-up">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-400" />
            <Input
              placeholder="Search merchant or description…"
              className="pl-10"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
            />
          </div>
          <Select
            value={type}
            onChange={(e) => {
              setType(e.target.value);
              setPage(0);
            }}
            className="sm:w-40"
          >
            <option value="">All types</option>
            <option value="credit">Money in</option>
            <option value="debit">Money out</option>
          </Select>
          <Select
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
              setPage(0);
            }}
            className="sm:w-44"
          >
            <option value="">All categories</option>
            {(categories.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </Select>
          {filtersActive && (
            <Button variant="ghost" size="sm" onClick={resetFilters} className="shrink-0 self-center">
              Clear
            </Button>
          )}
        </div>
      </Card>

      {/* Table */}
      <Card className="overflow-hidden animate-fade-up stagger-1">
        {transactions.loading ? (
          <div className="p-5 space-y-3">
            {[0, 1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        ) : transactions.error ? (
          <ErrorState
            message={apiErrorMessage(transactions.error, 'transactions')}
            onRetry={transactions.refetch}
          />
        ) : rows.length === 0 ? (
          <EmptyState
            icon={<ArrowLeftRight className="w-6 h-6" />}
            title={filtersActive ? 'No matching transactions' : 'No transactions yet'}
            description={
              filtersActive
                ? 'Nothing matches the current filters. Try widening your search.'
                : 'Your account activity will appear here.'
            }
            action={
              filtersActive ? (
                <Button variant="secondary" size="sm" onClick={resetFilters}>
                  Clear filters
                </Button>
              ) : undefined
            }
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left min-w-[640px]">
                <thead>
                  <tr className="border-b border-ink-950/[0.07] bg-ink-50/50">
                    <th className="px-6 py-3 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                      Date
                    </th>
                    <th className="px-6 py-3 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                      Merchant
                    </th>
                    <th className="px-6 py-3 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                      Category
                    </th>
                    <th className="px-6 py-3 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                      Type
                    </th>
                    <th className="px-6 py-3 text-[11px] font-semibold uppercase tracking-wide text-ink-500 text-right">
                      Amount
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-950/[0.05]">
                  {rows.map((t) => (
                    <tr key={t.id} className="hover:bg-ink-50/40 transition-colors">
                      <td className="px-6 py-3.5 text-[13px] text-ink-500 whitespace-nowrap">
                        {formatDate(t.timestamp)}
                      </td>
                      <td className="px-6 py-3.5">
                        <span className="text-[13.5px] font-medium text-ink-950">
                          {t.merchant || t.description || 'Transaction'}
                        </span>
                      </td>
                      <td className="px-6 py-3.5">
                        {t.category_name ? (
                          <Badge tone="neutral">{t.category_name}</Badge>
                        ) : (
                          <span className="text-[12.5px] text-ink-300">—</span>
                        )}
                      </td>
                      <td className="px-6 py-3.5">
                        <Badge tone={t.type === 'credit' ? 'positive' : 'negative'}>
                          {t.type === 'credit' ? 'Money in' : 'Money out'}
                        </Badge>
                      </td>
                      <td
                        className={cn(
                          'px-6 py-3.5 text-right tabular text-[13.5px] font-semibold whitespace-nowrap',
                          t.type === 'credit' ? 'text-positive-600' : 'text-ink-950',
                        )}
                      >
                        {t.type === 'credit' ? '+' : '−'}
                        {formatMYR(t.amount)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(hasPrev || hasNext) && (
              <div className="flex items-center justify-between px-6 py-3.5 border-t border-ink-950/[0.06]">
                <span className="text-[12.5px] text-ink-500">
                  Page {page + 1} · showing up to {PAGE_SIZE} per page
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={!hasPrev}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                  >
                    <ChevronLeft className="w-3.5 h-3.5" /> Previous
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={!hasNext}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next <ChevronRight className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
