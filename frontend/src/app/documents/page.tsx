'use client';

import { useRef, useState } from 'react';
import Link from 'next/link';
import { FileText, UploadCloud, CheckCircle2, Clock, XCircle, BookOpen } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { titleCase } from '@/lib/format';
import { Card, CardHeader, Badge, Button, Tabs, Skeleton, cn } from '@/components/ui';
import { EmptyState, ErrorState, PageHeader } from '@/components/states';
import KnowledgeSearch from '@/components/KnowledgeSearch';

interface Document {
  id: number;
  filename: string;
  status: string;
  document_type: string;
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'processed')
    return (
      <Badge tone="positive">
        <CheckCircle2 className="w-3 h-3" /> Processed
      </Badge>
    );
  if (status === 'error')
    return (
      <Badge tone="negative">
        <XCircle className="w-3 h-3" /> Failed
      </Badge>
    );
  return (
    <Badge tone="caution">
      <Clock className="w-3 h-3" /> {titleCase(status)}
    </Badge>
  );
}

export default function DocumentsPage() {
  const [tab, setTab] = useState('library');
  const documents = useApi<Document[]>('/documents/');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError('');
    setUploadSuccess('');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      await api.post('/documents/upload', formData, true);
      setUploadSuccess(`“${selectedFile.name}” uploaded and processed. It is now searchable.`);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      documents.refetch();
    } catch (err) {
      setUploadError(apiErrorMessage(err, 'the document upload'));
    } finally {
      setUploading(false);
    }
  };

  const docs = documents.data ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Documents"
        subtitle="Upload financial documents and ask questions grounded in their content."
        actions={
          <Tabs
            tabs={[
              { id: 'library', label: 'Library', icon: <FileText className="w-3.5 h-3.5" /> },
              { id: 'knowledge', label: 'Ask Documents', icon: <BookOpen className="w-3.5 h-3.5" /> },
            ]}
            active={tab}
            onChange={setTab}
          />
        }
      />

      {tab === 'knowledge' ? (
        <Card className="p-6 animate-fade-up">
          <KnowledgeSearch />
        </Card>
      ) : (
        <>
          {/* Upload */}
          <Card className="p-6 animate-fade-up">
            <h3 className="text-[15px] font-semibold text-ink-950 mb-4">Upload a document</h3>
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4">
              <button
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'focus-ring flex-1 rounded-xl border-2 border-dashed px-6 py-6 text-center transition-colors',
                  selectedFile
                    ? 'border-primary-500/50 bg-primary-50/40'
                    : 'border-ink-200 hover:border-primary-500/40 bg-ink-50/40',
                )}
              >
                <UploadCloud
                  className={cn(
                    'mx-auto w-9 h-9 mb-2',
                    selectedFile ? 'text-primary-600' : 'text-ink-300',
                  )}
                />
                <span className="text-[13.5px] font-medium text-ink-800">
                  {selectedFile ? selectedFile.name : 'Choose a file to upload'}
                </span>
                <span className="block text-[12px] text-ink-500 mt-1">
                  PDF or plain text, up to 5 MB
                </span>
              </button>
              <input
                ref={fileInputRef}
                id="document-upload"
                type="file"
                accept=".pdf,.txt"
                className="sr-only"
                onChange={(e) => {
                  setSelectedFile(e.target.files?.[0] ?? null);
                  setUploadError('');
                  setUploadSuccess('');
                }}
              />
              <Button onClick={handleUpload} disabled={!selectedFile} loading={uploading} className="shrink-0">
                Upload & process
              </Button>
            </div>
            {uploadError && (
              <p className="mt-3 text-[13px] text-negative-600 animate-fade-in">{uploadError}</p>
            )}
            {uploadSuccess && (
              <p className="mt-3 text-[13px] text-positive-600 animate-fade-in">{uploadSuccess}</p>
            )}
          </Card>

          {/* Library */}
          <Card className="animate-fade-up stagger-1">
            <CardHeader
              title="Your documents"
              subtitle={docs.length > 0 ? `${docs.length} uploaded` : 'Nothing uploaded yet'}
              action={
                <Link href="/knowledge">
                  <Button variant="ghost" size="sm">
                    <BookOpen className="w-3.5 h-3.5" />
                    Ask a question
                  </Button>
                </Link>
              }
            />
            {documents.loading ? (
              <div className="px-6 pb-6 space-y-3">
                {[0, 1].map((i) => (
                  <Skeleton key={i} className="h-14" />
                ))}
              </div>
            ) : documents.error ? (
              <ErrorState
                message={apiErrorMessage(documents.error, 'documents')}
                onRetry={documents.refetch}
              />
            ) : docs.length === 0 ? (
              <EmptyState
                icon={<FileText className="w-6 h-6" />}
                title="No documents yet"
                description="Upload an invoice, statement or contract above. Once processed, you can ask questions about its contents."
              />
            ) : (
              <div className="divide-y divide-ink-950/[0.05]">
                {docs.map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between gap-4 px-6 py-4">
                    <div className="flex items-center gap-3.5 min-w-0">
                      <span className="w-10 h-10 rounded-xl bg-ink-50 text-ink-500 flex items-center justify-center shrink-0">
                        <FileText className="w-5 h-5" />
                      </span>
                      <div className="min-w-0">
                        <p className="text-[13.5px] font-medium text-ink-950 truncate">
                          {doc.filename}
                        </p>
                        <p className="text-[12px] text-ink-500">{titleCase(doc.document_type)}</p>
                      </div>
                    </div>
                    <StatusBadge status={doc.status} />
                  </div>
                ))}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
