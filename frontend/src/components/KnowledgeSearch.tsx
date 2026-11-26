'use client';

import { useState } from 'react';
import { Sparkles, FileSearch, Quote } from 'lucide-react';
import { api, apiErrorMessage } from '@/lib/api';
import { Button, Card, Input, Skeleton, Badge, cn } from '@/components/ui';

interface Source {
  document_id: number;
  chunk_id: number;
  filename: string | null;
  document_type: string | null;
  source_pages: number[] | null;
  metadata: Record<string, unknown> | null;
}

interface QueryResult {
  grounded: boolean;
  answer: string;
  sources: Source[];
  error: string | null;
}

const EXAMPLES = [
  'What are the payment terms in my documents?',
  'Summarise the key obligations in my contract',
  'What invoice amounts are due?',
];

export default function KnowledgeSearch() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState('');

  const runQuery = async (q: string) => {
    const question = q.trim();
    if (!question) return;
    setQuery(question);
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.post<QueryResult>('/knowledge/query', { query: question, top_k: 5 });
      setResult(res);
    } catch (err) {
      setError(apiErrorMessage(err, 'the knowledge search'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void runQuery(query);
        }}
        className="flex flex-col sm:flex-row gap-3"
      >
        <Input
          placeholder="Ask about your uploaded documents…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1"
        />
        <Button type="submit" loading={loading} className="shrink-0">
          <FileSearch className="w-4 h-4" />
          Search
        </Button>
      </form>

      {!result && !loading && !error && (
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => void runQuery(ex)}
              className="focus-ring rounded-full border border-ink-950/10 bg-white px-3.5 py-1.5 text-[12.5px] text-ink-600 hover:border-primary-500/40 hover:text-primary-600 transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {loading && (
        <Card className="p-6 space-y-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-2/3" />
        </Card>
      )}

      {error && (
        <p className="text-[13px] text-negative-600">{error}</p>
      )}

      {result && !loading && (
        <Card className="p-6 animate-fade-up">
          {result.grounded ? (
            <>
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-primary-600" />
                <Badge tone="primary">Answer grounded in your documents</Badge>
              </div>
              <p className="text-[14px] text-ink-800 leading-relaxed whitespace-pre-wrap">
                {result.answer}
              </p>
              {result.sources.length > 0 && (
                <div className="mt-6 pt-5 border-t border-ink-950/[0.06]">
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-400 mb-3">
                    Evidence
                  </p>
                  <div className="space-y-2.5">
                    {result.sources.map((s, i) => (
                      <div
                        key={`${s.document_id}-${s.chunk_id}-${i}`}
                        className={cn(
                          'flex items-start gap-2.5 rounded-xl bg-ink-50/60 border border-ink-950/[0.05] px-3.5 py-2.5 text-[12.5px] text-ink-600',
                        )}
                      >
                        <Quote className="w-3.5 h-3.5 text-ink-300 mt-0.5 shrink-0" />
                        <span>
                          <span className="font-medium text-ink-800">
                            {s.filename || `Document #${s.document_id}`}
                          </span>
                          {s.document_type ? ` · ${s.document_type}` : ''}
                          {Array.isArray(s.source_pages) && s.source_pages.length > 0
                            ? ` · page ${s.source_pages.join(', ')}`
                            : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-start gap-3">
              <FileSearch className="w-5 h-5 text-ink-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-[14px] text-ink-700 leading-relaxed">
                  {result.answer ||
                    'I could not find sufficient information in your uploaded documents to answer that.'}
                </p>
                <p className="text-[12.5px] text-ink-400 mt-2">
                  Tip: upload documents on the Library tab, then ask again. Answers are only
                  produced from evidence found in your documents.
                </p>
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
