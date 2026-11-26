'use client';

import { useState } from 'react';
import { LifeBuoy, Plus, CheckCircle2 } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { formatDateTime } from '@/lib/format';
import { Card, CardHeader, Badge, Button, Input, Select, Skeleton } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface Ticket {
  id: number;
  title: string;
  description?: string;
  status: string;
  priority: string;
  category?: string;
  created_at?: string;
}

interface CreateTicketResponse {
  id: number;
  title: string;
  status: string;
  priority: string;
  category: string;
}

const PRIORITIES = ['low', 'normal', 'high'];
const CATEGORIES = ['general', 'transactions', 'cards', 'account', 'documents'];

function PriorityBadge({ priority }: { priority: string }) {
  const tone =
    priority === 'high' ? 'negative' : priority === 'low' ? 'neutral' : 'primary';
  return <Badge tone={tone}>{priority}</Badge>;
}

export default function HelpDeskPage() {
  const tickets = useApi<Ticket[]>('/help-desk/');
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState('normal');
  const [category, setCategory] = useState('general');
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState('');
  const [success, setSuccess] = useState('');

  const submitTicket = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setFormError('');
    setSuccess('');
    try {
      const created = await api.post<CreateTicketResponse>('/help-desk/', {
        title,
        description,
        priority,
        category,
      });
      setSuccess(`Ticket #${created.id} created. Our support team will follow up.`);
      setTitle('');
      setDescription('');
      setPriority('normal');
      setCategory('general');
      setOpen(false);
      tickets.refetch();
      setTimeout(() => setSuccess(''), 6000);
    } catch (err) {
      setFormError(apiErrorMessage(err, 'creating the ticket'));
    } finally {
      setCreating(false);
    }
  };

  const items = tickets.data ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Help Desk"
        subtitle="Raise support requests and follow their progress."
        actions={
          <Button onClick={() => setOpen((v) => !v)} variant={open ? 'secondary' : 'primary'}>
            {!open && <Plus className="w-4 h-4" />}
            {open ? 'Close form' : 'New ticket'}
          </Button>
        }
      />

      {success && (
        <div className="rounded-xl bg-positive-50 border border-positive-600/15 px-4 py-3 text-[13px] text-positive-600 flex items-center gap-2 animate-fade-in">
          <CheckCircle2 className="w-4 h-4" /> {success}
        </div>
      )}

      {open && (
        <Card className="p-6 animate-fade-up">
          <h3 className="text-[15px] font-semibold text-ink-950 mb-4">Describe your issue</h3>
          <form onSubmit={submitTicket} className="space-y-4">
            <div>
              <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Title</label>
              <Input
                required
                placeholder="Short summary, e.g. Unexpected charge on my card"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Details</label>
              <textarea
                required
                rows={4}
                placeholder="What happened, and what would you like us to do?"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="focus-ring w-full rounded-xl border border-ink-950/10 bg-white px-3.5 py-2.5 text-sm text-ink-900 placeholder:text-ink-400 hover:border-ink-950/20 focus:border-primary-500 resize-y"
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Priority</label>
                <Select value={priority} onChange={(e) => setPriority(e.target.value)}>
                  {PRIORITIES.map((p) => (
                    <option key={p} value={p}>
                      {p[0].toUpperCase() + p.slice(1)}
                    </option>
                  ))}
                </Select>
              </div>
              <div>
                <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Category</label>
                <Select value={category} onChange={(e) => setCategory(e.target.value)}>
                  {CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c[0].toUpperCase() + c.slice(1)}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
            {formError && <p className="text-[13px] text-negative-600">{formError}</p>}
            <div className="flex gap-2">
              <Button type="submit" loading={creating}>
                Submit ticket
              </Button>
              <Button type="button" variant="secondary" onClick={() => setOpen(false)}>
                Cancel
              </Button>
            </div>
          </form>
        </Card>
      )}

      <Card className="overflow-hidden animate-fade-up stagger-1">
        <CardHeader
          title="Your tickets"
          subtitle={items.length > 0 ? `${items.length} ticket${items.length === 1 ? '' : 's'}` : 'No tickets yet'}
        />
        {tickets.loading ? (
          <div className="px-6 pb-6 space-y-3">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-14" />
            ))}
          </div>
        ) : tickets.error ? (
          <ErrorState message={apiErrorMessage(tickets.error, 'tickets')} onRetry={tickets.refetch} />
        ) : items.length === 0 ? (
          <EmptyState
            icon={<LifeBuoy className="w-6 h-6" />}
            title="No support tickets"
            description="If something looks wrong — a transaction, a card, a document — raise a ticket and the team will investigate."
          />
        ) : (
          <div className="divide-y divide-ink-950/[0.05]">
            {items.map((t) => (
              <div key={t.id} className="px-6 py-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-[13.5px] font-semibold text-ink-950">
                      <span className="text-ink-400 font-normal tabular mr-1.5">#{t.id}</span>
                      {t.title}
                    </p>
                    <p className="text-[12px] text-ink-500 mt-0.5">
                      {t.category ? `${t.category} · ` : ''}
                      {t.created_at ? formatDateTime(t.created_at) : ''}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <PriorityBadge priority={t.priority} />
                    <Badge tone={t.status === 'open' ? 'caution' : 'positive'}>{t.status}</Badge>
                  </div>
                </div>
                {t.description && (
                  <p className="text-[12.5px] text-ink-600 mt-2 leading-relaxed line-clamp-2">
                    {t.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
