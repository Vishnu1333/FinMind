'use client';

import { RefreshCw } from 'lucide-react';
import { Button, Card, Skeleton, cn } from './ui';

/** Full-page loading skeleton for data pages. */
export function PageSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Skeleton className="h-7 w-56" />
        <Skeleton className="h-4 w-80" />
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Card className="p-6">
        <Skeleton className="h-64" />
      </Card>
      {rows > 1 && (
        <Card className="p-6">
          <Skeleton className="h-40" />
        </Card>
      )}
    </div>
  );
}

/** Beautiful empty state with icon, title, description and optional action. */
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-14 px-6', className)}>
      <div className="w-14 h-14 rounded-2xl bg-ink-50 text-ink-400 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-[15px] font-semibold text-ink-900">{title}</h3>
      <p className="text-[13px] text-ink-500 mt-1 max-w-sm text-balance">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/** Error state with a human message and retry. */
export function ErrorState({
  message,
  onRetry,
  className,
}: {
  message: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-12 px-6', className)}>
      <div className="w-12 h-12 rounded-full bg-negative-50 text-negative-600 flex items-center justify-center mb-4">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h3 className="text-[15px] font-semibold text-ink-900">Something went wrong</h3>
      <p className="text-[13px] text-ink-500 mt-1 max-w-md text-balance">{message}</p>
      {onRetry && (
        <Button variant="secondary" size="sm" className="mt-5" onClick={onRetry}>
          <RefreshCw className="w-3.5 h-3.5" />
          Try again
        </Button>
      )}
    </div>
  );
}

/** Standard page header used by every page. */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
      <div>
        <h1 className="text-[22px] font-semibold tracking-tight text-ink-950">{title}</h1>
        {subtitle && <p className="text-[13.5px] text-ink-500 mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
