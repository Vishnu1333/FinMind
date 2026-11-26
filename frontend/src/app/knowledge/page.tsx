'use client';

import { BookOpen } from 'lucide-react';
import { Card } from '@/components/ui';
import { PageHeader } from '@/components/states';
import KnowledgeSearch from '@/components/KnowledgeSearch';

export default function KnowledgePage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Knowledge Search"
        subtitle="Retrieval-augmented answers grounded strictly in your uploaded documents, with cited evidence."
      />
      <Card className="p-6 animate-fade-up">
        <div className="flex items-start gap-3 mb-6 rounded-xl bg-primary-50/50 border border-primary-600/10 px-4 py-3">
          <BookOpen className="w-4 h-4 text-primary-600 mt-0.5 shrink-0" />
          <p className="text-[12.5px] text-ink-600 leading-relaxed">
            Every answer is produced only from passages retrieved from your business documents.
            If the evidence is insufficient, FinMind says so instead of guessing.
          </p>
        </div>
        <KnowledgeSearch />
      </Card>
    </div>
  );
}
