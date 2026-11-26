'use client';

import { useEffect, useRef, useState } from 'react';
import { Sparkles, Send, ShieldCheck, User, Bot, ShieldAlert, Check, X } from 'lucide-react';
import { api, apiErrorMessage } from '@/lib/api';
import { Button, Badge, Toggle, cn } from '@/components/ui';

interface Step {
  tool: string;
  kind: string;
  text_lines: string[];
}

interface PendingAction {
  type: string;
  last_four: string;
  card_id: number;
}

interface AssistantReply {
  response: string;
  grounded: boolean;
  intent: string;
  intents: string[];
  provider: string;
  refused: boolean;
  actions: PendingAction[];
  steps: Step[];
}

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  steps?: Step[];
  actions?: PendingAction[];
  provider?: string;
  grounded?: boolean;
  refused?: boolean;
  acted?: boolean;
}

const KIND_TONE: Record<string, string> = {
  DATA: 'bg-ink-100 text-ink-600',
  CALCULATION: 'bg-primary-50 text-primary-600',
  PREDICTION: 'bg-[#F3EEFE] text-accent-600',
  EVIDENCE: 'bg-positive-50 text-positive-600',
  STATUS: 'bg-caution-50 text-caution-600',
};

const SUGGESTIONS = [
  "What's my current account balance?",
  'Show my last 10 transactions.',
  'How much did I spend on inventory this month?',
  'Analyze my cash flow.',
  'What will my cash flow look like over the next 30 days?',
  'Show unusual transactions.',
  'Show my notifications.',
  'Show my cards.',
];

function StepCard({ step }: { step: Step }) {
  return (
    <div className="rounded-xl border border-ink-950/[0.07] bg-white px-4 py-3">
      <div className="flex items-center gap-2 mb-2">
        <span
          className={cn(
            'px-2 py-0.5 rounded-full text-[10.5px] font-bold uppercase tracking-wide',
            KIND_TONE[step.kind] ?? 'bg-ink-100 text-ink-600',
          )}
        >
          {step.kind}
        </span>
        <span className="text-[11px] text-ink-400 font-medium">{step.tool.replace(/_/g, ' ')}</span>
      </div>
      <ul className="space-y-1">
        {step.text_lines.map((line, i) => (
          <li key={i} className="text-[12.5px] text-ink-600 leading-relaxed whitespace-pre-wrap">
            {line}
          </li>
        ))}
      </ul>
    </div>
  );
}

function AssistantBubble({ message, onAction }: { message: Message; onAction: (text: string) => void }) {
  const pending = (message.actions ?? []).find((a) => a.type === 'confirm_block' && !message.acted);

  return (
    <div className="flex gap-3 animate-fade-up">
      <span className="w-8 h-8 rounded-full brand-gradient text-white flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-4 h-4" />
      </span>
      <div className="min-w-0 max-w-[min(680px,90%)] space-y-3">
        <div className="rounded-2xl rounded-tl-md bg-white border border-ink-950/[0.07] px-4 py-3 shadow-[var(--shadow-card)]">
          <p className="text-[13.5px] text-ink-800 leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>

        {message.steps && message.steps.length > 0 && (
          <div className="space-y-2">
            {message.steps.map((step, i) => (
              <StepCard key={i} step={step} />
            ))}
          </div>
        )}

        {pending && (
          <div className="rounded-xl border border-caution-600/25 bg-caution-50/60 p-4">
            <div className="flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-caution-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <p className="text-[13px] text-ink-700">
                  Confirm blocking card <span className="font-semibold">•••• {pending.last_four}</span>?
                </p>
                <div className="flex gap-2 mt-3">
                  <Button
                    size="sm"
                    variant="danger"
                    onClick={() => onAction('confirm')}
                    className="gap-1.5"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Yes, block it
                  </Button>
                  <Button size="sm" variant="secondary" onClick={() => onAction('cancel')}>
                    <X className="w-3.5 h-3.5" />
                    Cancel
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 text-[10.5px] text-ink-400">
          {message.grounded && (
            <span className="inline-flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" /> Verified data
            </span>
          )}
          <span>{message.provider === 'llm' ? 'AI language layer' : 'Direct tool report'}</span>
          {message.refused && (
            <span className="inline-flex items-center gap-1 text-caution-600">
              <ShieldAlert className="w-3 h-3" /> Policy refusal
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: 'assistant',
      content:
        "Hello, I'm FinMind — your financial assistant. I can look up accounts, transactions, spending and forecasts, and I can manage cards, notifications and support tickets. Everything I report comes from your real business data. How can I help?",
      provider: 'deterministic',
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [privacy, setPrivacy] = useState(false);
  const [error, setError] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages, isTyping]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isTyping) return;
    setError('');
    const userMessage: Message = { id: Date.now(), role: 'user', content: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const reply = await api.post<AssistantReply>('/assistant/chat', {
        message: trimmed,
        privacy_sensitive: privacy,
      });
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: reply.response,
          steps: reply.steps,
          actions: reply.actions,
          provider: reply.provider,
          grounded: reply.grounded,
          refused: reply.refused,
          acted: false,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: apiErrorMessage(err, 'the assistant'),
          provider: 'deterministic',
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const markActed = (messageId: number) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, acted: true } : m)),
    );
  };

  const handleAction = (sourceMessageId: number, text: string) => {
    markActed(sourceMessageId);
    void send(text);
  };

  const showSuggestions =
    messages.filter((m) => m.role === 'user').length === 0 && !isTyping;

  return (
    <div className="flex flex-col h-[calc(100vh-11rem)] min-h-[520px] animate-fade-up">
      {/* Header card */}
      <div className="rounded-2xl bg-ink-950 px-6 py-5 mb-4 relative overflow-hidden">
        <div
          className="absolute -top-16 -right-10 w-56 h-56 rounded-full opacity-30 blur-3xl"
          style={{ background: 'radial-gradient(circle, #7C3AED, transparent 65%)' }}
        />
        <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <Sparkles className="w-5 h-5 text-primary-400" />
              <h1 className="text-[18px] font-semibold text-white tracking-tight">FinMind AI</h1>
            </div>
            <p className="text-[12.5px] text-ink-300 mt-1">
              Deterministic tools, real data, transparent evidence. Every answer is grounded.
            </p>
          </div>
          <div className="rounded-xl bg-white/[0.06] border border-white/[0.08] px-4 py-3 sm:w-72">
            <Toggle
              checked={privacy}
              onChange={setPrivacy}
              label="Privacy mode"
              description={
                privacy
                  ? 'Local-model only — never sent to the cloud.'
                  : 'Cloud AI allowed with local fallback.'
              }
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-5 px-1 pb-4">
        {showSuggestions && (
          <div className="flex flex-wrap gap-2 pt-1">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => void send(s)}
                className="focus-ring rounded-full border border-ink-950/10 bg-white px-3.5 py-1.5 text-[12.5px] text-ink-600 hover:border-primary-500/40 hover:text-primary-600 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((message) =>
          message.role === 'user' ? (
            <div key={message.id} className="flex gap-3 justify-end animate-fade-up">
              <div className="max-w-[min(560px,85%)] rounded-2xl rounded-tr-md bg-primary-600 text-white px-4 py-3">
                <p className="text-[13.5px] leading-relaxed">{message.content}</p>
              </div>
              <span className="w-8 h-8 rounded-full bg-ink-200 text-ink-600 flex items-center justify-center shrink-0 mt-0.5">
                <User className="w-4 h-4" />
              </span>
            </div>
          ) : (
            <AssistantBubble
              key={message.id}
              message={message}
              onAction={(text) => handleAction(message.id, text)}
            />
          ),
        )}

        {isTyping && (
          <div className="flex gap-3 animate-fade-in">
            <span className="w-8 h-8 rounded-full brand-gradient text-white flex items-center justify-center shrink-0 mt-0.5">
              <Bot className="w-4 h-4" />
            </span>
            <div className="bg-white border border-ink-950/[0.07] rounded-2xl rounded-tl-md px-4 py-3.5 shadow-[var(--shadow-card)] flex items-center gap-1.5">
              <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary-500" />
              <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary-500" />
              <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary-500" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {error && (
        <p className="text-[12.5px] text-negative-600 mb-2">{error}</p>
      )}

      {/* Composer */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="rounded-2xl border border-ink-950/[0.08] bg-white p-2.5 shadow-[var(--shadow-card)] flex items-center gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            privacy
              ? 'Ask anything — processed privately on the local model…'
              : 'Ask about your finances, documents, cards…'
          }
          className="flex-1 bg-transparent px-3 py-2 text-[14px] text-ink-900 placeholder:text-ink-400 outline-none"
          disabled={isTyping}
        />
        {privacy && (
          <Badge tone="ink" className="hidden sm:inline-flex">
            <ShieldCheck className="w-3 h-3" /> Private
          </Badge>
        )}
        <Button type="submit" disabled={!input.trim() || isTyping} aria-label="Send">
          <Send className="w-4 h-4" />
        </Button>
      </form>
    </div>
  );
}
