'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, ArrowRight } from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import { Button, Input } from '@/components/ui';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (api.getToken()) {
      router.replace('/dashboard');
      return;
    }
    const reason = new URLSearchParams(window.location.search).get('reason');
    if (reason === 'expired') {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time hydration of an external-system value (URL param)
      setNotice('Your session has expired. Please sign in again.');
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      api.clearToken();
      const response = await api.post<{ access_token: string }>('/auth/login', {
        email: email.trim(),
        password,
      });
      api.setToken(response.access_token);
      router.replace('/dashboard');
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 401) {
          setError('Incorrect email or password. Please try again.');
        } else if (err.status === 422) {
          setError('Please enter a valid email address and password.');
        } else if (err.status >= 500) {
          setError('FinMind server error. Please try again later.');
        } else {
          setError(err.message || 'Unable to sign in. Please try again.');
        }
      } else {
        setError(
          'FinMind service is unavailable. Please make sure the backend is running.',
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-canvas">
      {/* Brand panel */}
      <div className="hidden lg:flex w-[46%] bg-ink-950 relative overflow-hidden flex-col justify-between p-12">
        <div
          className="absolute -top-40 -right-40 w-[480px] h-[480px] rounded-full opacity-25 blur-3xl"
          style={{ background: 'radial-gradient(circle, #7C3AED, transparent 65%)' }}
        />
        <div
          className="absolute -bottom-52 -left-32 w-[420px] h-[420px] rounded-full opacity-20 blur-3xl"
          style={{ background: 'radial-gradient(circle, #4F46E5, transparent 65%)' }}
        />
        <div className="flex items-center gap-2.5 relative">
          <span className="w-9 h-9 rounded-xl brand-gradient flex items-center justify-center">
            <Sparkles className="w-4.5 h-4.5 text-white" />
          </span>
          <span className="text-lg font-semibold tracking-tight text-white">
            FIN<span className="text-primary-400">MIND</span>
          </span>
        </div>
        <div className="relative">
          <h1 className="text-[34px] leading-[1.15] font-semibold tracking-tight text-white max-w-md text-balance">
            Financial intelligence for your business, in one calm place.
          </h1>
          <p className="mt-4 text-[15px] text-ink-300 max-w-md leading-relaxed">
            Accounts, transactions, cash-flow forecasting and document intelligence — with an AI
            assistant that answers from your real data.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {['Cash-flow forecasts', 'Anomaly detection', 'Document intelligence', 'AI assistant'].map(
              (tag) => (
                <span
                  key={tag}
                  className="px-3 py-1 rounded-full bg-white/[0.06] border border-white/[0.08] text-[12px] text-ink-200"
                >
                  {tag}
                </span>
              ),
            )}
          </div>
        </div>
        <p className="relative text-[12px] text-ink-500">
          Simulated banking environment for SME financial intelligence.
        </p>
      </div>

      {/* Form panel */}
      <div className="flex-1 flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-[400px] animate-fade-up">
          <div className="lg:hidden flex items-center gap-2.5 mb-10">
            <span className="w-9 h-9 rounded-xl brand-gradient flex items-center justify-center">
              <Sparkles className="w-4.5 h-4.5 text-white" />
            </span>
            <span className="text-lg font-semibold tracking-tight text-ink-950">
              FIN<span className="text-primary-600">MIND</span>
            </span>
          </div>

          <h2 className="text-[24px] font-semibold tracking-tight text-ink-950">Welcome back</h2>
          <p className="text-[14px] text-ink-500 mt-1">Sign in to your FinMind workspace.</p>

          {notice && (
            <div className="mt-6 rounded-xl bg-caution-50 border border-caution-600/15 px-4 py-3 text-[13px] text-caution-600">
              {notice}
            </div>
          )}
          {error && (
            <div className="mt-6 rounded-xl bg-negative-50 border border-negative-600/15 px-4 py-3 text-[13px] text-negative-600 animate-fade-in">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-8 space-y-4">
            <div>
              <label htmlFor="email" className="block text-[13px] font-medium text-ink-700 mb-1.5">
                Email address
              </label>
              <Input
                id="email"
                type="email"
                required
                autoComplete="email"
                placeholder="you@business.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-[13px] font-medium text-ink-700 mb-1.5">
                Password
              </label>
              <Input
                id="password"
                type="password"
                required
                autoComplete="current-password"
                placeholder="Your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <Button type="submit" loading={isLoading} className="w-full mt-2">
              Sign in
              {!isLoading && <ArrowRight className="w-4 h-4" />}
            </Button>
          </form>

          <p className="mt-6 text-center text-[13px] text-ink-500">
            New to FinMind?{' '}
            <Link href="/register" className="font-medium text-primary-600 hover:text-primary-700">
              Create a business account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}