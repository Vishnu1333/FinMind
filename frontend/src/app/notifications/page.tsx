'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Bell, CheckCheck, Info, ShieldAlert, AlertTriangle } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { formatDateTime } from '@/lib/format';
import { Card, Badge, Button, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface Notification {
  id: number;
  title: string;
  message: string;
  read: boolean;
  severity: string;
  created_at?: string;
}

function SeverityIcon({ severity }: { severity: string }) {
  if (severity === 'warning')
    return <AlertTriangle className="w-4 h-4 text-caution-600" />;
  if (severity === 'critical' || severity === 'error')
    return <ShieldAlert className="w-4 h-4 text-negative-600" />;
  return <Info className="w-4 h-4 text-primary-600" />;
}

export default function NotificationsPage() {
  const notifications = useApi<Notification[]>('/notifications/');
  const [markingAll, setMarkingAll] = useState(false);
  const [markingId, setMarkingId] = useState<number | null>(null);
  const [actionError, setActionError] = useState('');

  const items = notifications.data ?? [];
  const unreadCount = items.filter((n) => !n.read).length;

  const markAllRead = async () => {
    setMarkingAll(true);
    setActionError('');
    try {
      await api.post('/notifications/mark-all-read', {});
      notifications.refetch();
    } catch (err) {
      setActionError(apiErrorMessage(err, 'marking notifications read'));
    } finally {
      setMarkingAll(false);
    }
  };

  const markOneRead = async (id: number) => {
    setMarkingId(id);
    setActionError('');
    try {
      await api.post(`/notifications/${id}/read`, {});
      notifications.refetch();
    } catch (err) {
      setActionError(apiErrorMessage(err, 'marking the notification read'));
    } finally {
      setMarkingId(null);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Notifications"
        subtitle={
          unreadCount > 0
            ? `You have ${unreadCount} unread notification${unreadCount === 1 ? '' : 's'}.`
            : 'You are all caught up.'
        }
        actions={
          unreadCount > 0 ? (
            <Button variant="secondary" size="sm" onClick={markAllRead} loading={markingAll}>
              <CheckCheck className="w-3.5 h-3.5" />
              Mark all read
            </Button>
          ) : undefined
        }
      />

      {actionError && (
        <p className="text-[13px] text-negative-600">{actionError}</p>
      )}

      <Card className="overflow-hidden animate-fade-up">
        {notifications.loading ? (
          <div className="p-5 space-y-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-16" />
            ))}
          </div>
        ) : notifications.error ? (
          <ErrorState
            message={apiErrorMessage(notifications.error, 'notifications')}
            onRetry={notifications.refetch}
          />
        ) : items.length === 0 ? (
          <EmptyState
            icon={<Bell className="w-6 h-6" />}
            title="No notifications"
            description="Account alerts and security notices will appear here."
          />
        ) : (
          <div className="divide-y divide-ink-950/[0.05]">
            {items.map((n) => (
              <div
                key={n.id}
                className={cn(
                  'flex items-start gap-4 px-6 py-4 transition-colors',
                  !n.read && 'bg-primary-50/30',
                )}
              >
                <span className="w-9 h-9 rounded-xl bg-ink-50 flex items-center justify-center shrink-0 mt-0.5">
                  <SeverityIcon severity={n.severity} />
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <p className="text-[13.5px] font-semibold text-ink-950">{n.title}</p>
                    {!n.read && <Badge tone="primary">New</Badge>}
                  </div>
                  <p className="text-[12.5px] text-ink-600 mt-0.5 leading-relaxed">{n.message}</p>
                  {n.created_at && (
                    <p className="text-[11.5px] text-ink-400 mt-1">{formatDateTime(n.created_at)}</p>
                  )}
                </div>
                {!n.read && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => markOneRead(n.id)}
                    loading={markingId === n.id}
                    className="shrink-0"
                  >
                    Mark read
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      <p className="text-[12px] text-ink-400">
        Need help with something?{' '}
        <Link href="/help-desk" className="text-primary-600 font-medium hover:text-primary-700">
          Raise a support ticket
        </Link>
      </p>
    </div>
  );
}
