'use client';

import { useState } from 'react';
import { CreditCard, ShieldCheck, ShieldBan, Sparkles } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { Card, Button, Modal, Skeleton } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';

interface CardItem {
  id: number;
  last_four: string;
  status: string;
}

function CardVisual({ card, onBlock }: { card: CardItem; onBlock: (card: CardItem) => void }) {
  const blocked = card.status === 'blocked';

  return (
    <div className="relative rounded-2xl overflow-hidden animate-fade-up">
      {/* Card face */}
      <div
        className={`relative h-52 p-6 flex flex-col justify-between ${
          blocked
            ? 'bg-ink-700'
            : 'bg-ink-950'
        }`}
      >
        {!blocked && (
          <>
            <div
              className="absolute -top-14 -right-10 w-44 h-44 rounded-full opacity-25 blur-2xl"
              style={{ background: 'radial-gradient(circle, #7C3AED, transparent 70%)' }}
            />
            <div
              className="absolute -bottom-20 -left-8 w-40 h-40 rounded-full opacity-20 blur-2xl"
              style={{ background: 'radial-gradient(circle, #4F46E5, transparent 70%)' }}
            />
          </>
        )}
        <div className="relative flex items-start justify-between">
          <span className="flex items-center gap-2 text-white/90">
            <Sparkles className="w-4 h-4" />
            <span className="text-[12px] font-semibold tracking-widest uppercase">FinMind</span>
          </span>
          <span
            className={`px-2.5 py-0.5 rounded-full text-[10.5px] font-bold uppercase tracking-wide ${
              blocked ? 'bg-negative-500/20 text-negative-500' : 'bg-positive-500/15 text-positive-500'
            }`}
          >
            {blocked ? 'Blocked' : 'Active'}
          </span>
        </div>
        <div className="relative">
          <p className="tabular text-white text-[19px] font-medium tracking-[0.22em]">
            •••• •••• •••• {card.last_four}
          </p>
          <div className="flex items-end justify-between mt-3">
            <div>
              <p className="text-[9.5px] text-white/40 uppercase tracking-wider">Business card</p>
              <p className="text-[12px] text-white/70 mt-0.5">Virtual · Simulated environment</p>
            </div>
            <span className="w-9 h-7 rounded-md bg-gradient-to-br from-amber-300 to-amber-500 opacity-90" />
          </div>
        </div>
      </div>

      {/* Action bar */}
      <div className="bg-white border border-t-0 border-ink-950/[0.07] rounded-b-2xl px-5 py-4 flex items-center justify-between">
        {blocked ? (
          <p className="text-[12.5px] text-ink-500 flex items-center gap-1.5">
            <ShieldBan className="w-4 h-4 text-negative-500" />
            Blocked — contact the help desk to request a replacement.
          </p>
        ) : (
          <p className="text-[12.5px] text-ink-500 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-positive-600" />
            Ready for new transactions.
          </p>
        )}
        {!blocked && (
          <Button variant="danger" size="sm" onClick={() => onBlock(card)}>
            <ShieldBan className="w-3.5 h-3.5" />
            Block card
          </Button>
        )}
      </div>
    </div>
  );
}

export default function CardsPage() {
  const cards = useApi<CardItem[]>('/cards/');
  const [confirmCard, setConfirmCard] = useState<CardItem | null>(null);
  const [blocking, setBlocking] = useState(false);
  const [blockError, setBlockError] = useState('');
  const [blockSuccess, setBlockSuccess] = useState('');

  const executeBlock = async () => {
    if (!confirmCard) return;
    setBlocking(true);
    setBlockError('');
    try {
      await api.post(`/cards/${confirmCard.id}/block`, {});
      setBlockSuccess(`Card •••• ${confirmCard.last_four} has been blocked.`);
      setConfirmCard(null);
      cards.refetch();
      setTimeout(() => setBlockSuccess(''), 6000);
    } catch (err) {
      setBlockError(apiErrorMessage(err, 'the card block'));
    } finally {
      setBlocking(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Cards"
        subtitle="Your business's virtual cards. Blocking is immediate and permanent."
      />

      {blockSuccess && (
        <div className="rounded-xl bg-positive-50 border border-positive-600/15 px-4 py-3 text-[13px] text-positive-600 animate-fade-in">
          {blockSuccess}
        </div>
      )}

      {cards.loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-4xl">
          {[0, 1].map((i) => (
            <Skeleton key={i} className="h-72 rounded-2xl" />
          ))}
        </div>
      ) : cards.error ? (
        <ErrorState message={apiErrorMessage(cards.error, 'cards')} onRetry={cards.refetch} />
      ) : (cards.data ?? []).length === 0 ? (
        <Card>
          <EmptyState
            icon={<CreditCard className="w-6 h-6" />}
            title="No cards issued"
            description="Your business has no cards. Contact the help desk to request one."
          />
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-4xl">
          {(cards.data ?? []).map((card) => (
            <CardVisual key={card.id} card={card} onBlock={setConfirmCard} />
          ))}
        </div>
      )}

      <Modal
        open={!!confirmCard}
        onClose={() => setConfirmCard(null)}
        title="Block this card?"
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmCard(null)} disabled={blocking}>
              Cancel
            </Button>
            <Button variant="danger" onClick={executeBlock} loading={blocking}>
              <ShieldBan className="w-4 h-4" />
              Block permanently
            </Button>
          </>
        }
      >
        {confirmCard && (
          <>
            <p>
              Card <span className="font-semibold text-ink-900">•••• {confirmCard.last_four}</span>{' '}
              will be blocked immediately. It will stop working for all new transactions.
            </p>
            <p className="mt-3 text-[12.5px] text-ink-500">
              Blocking is permanent: cards cannot be unblocked through FinMind or the AI
              assistant. If you need a replacement, raise a help desk ticket after blocking.
            </p>
            {blockError && (
              <p className="mt-3 text-[12.5px] text-negative-600">{blockError}</p>
            )}
          </>
        )}
      </Modal>
    </div>
  );
}
