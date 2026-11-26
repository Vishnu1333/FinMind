'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, ArrowRight } from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import { Button, Input } from '@/components/ui';

export default function RegisterPage() {
  const router = useRouter();
  const [businessName, setBusinessName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      // Clear any previous token before registering a new workspace
      api.clearToken();
      // Registration returns tokens directly — no second login call needed.
      const response = await api.post<{ access_token: string }>('/auth/register', {
        email: email.trim(),
        password,
        business_name: businessName.trim(),
      });
      api.setToken(response.access_token);
      router.replace('/dashboard');
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 409) {
          setError(err.message || 'An account with this email or business name already exists.');
        } else if (err.status === 400) {
          if (err.message && err.message.toLowerCase().includes('already exists')) {
            setError(err.message);
          } else {
            setError(err.message || 'Please check the registration details.');
          }
        } else if (err.status === 401) {
          setError('Authentication problem. Please try again.');
        } else if (err.status === 422) {
          setError(
            err.message
              ? `Please check the registration details: ${err.message}`
              : 'Please check the registration details.',
          );
        } else if (err.status >= 500) {
          setError('FinMind could not complete registration. Please try again.');
        } else {
          setError(err.message || 'Registration failed. Please try again.');
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
    <div className="min-h-screen flex items-center justify-center px-6 py-16 bg-canvas">
      <div className="w-full max-w-[420px] animate-fade-up">
        <div className="flex items-center gap-2.5 mb-10">
          <span className="w-9 h-9 rounded-xl brand-gradient flex items-center justify-center">
            <Sparkles className="w-4.5 h-4.5 text-white" />
          </span>
          <span className="text-lg font-semibold tracking-tight text-ink-950">
            FIN<span className="text-primary-600">MIND</span>
          </span>
        </div>

        <h1 className="text-[24px] font-semibold tracking-tight text-ink-950">
          Create your workspace
        </h1>
        <p className="text-[14px] text-ink-500 mt-1">
          Your business gets a simulated banking environment with accounts, cards and six months of
          transaction history to explore.
        </p>

        {error && (
          <div className="mt-6 rounded-xl bg-negative-50 border border-negative-600/15 px-4 py-3 text-[13px] text-negative-600 animate-fade-in">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-8 space-y-4">
          <div>
            <label htmlFor="business" className="block text-[13px] font-medium text-ink-700 mb-1.5">
              Business name
            </label>
            <Input
              id="business"
              type="text"
              required
              minLength={1}
              placeholder="e.g. Warung Mak Lim Sdn Bhd"
              value={businessName}
              onChange={(e) => setBusinessName(e.target.value)}
            />
          </div>
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
              minLength={8}
              autoComplete="new-password"
              placeholder="At least 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <p className="text-[11.5px] text-ink-400 mt-1.5">Minimum 8 characters.</p>
          </div>
          <Button type="submit" loading={isLoading} className="w-full mt-2">
            Create account
            {!isLoading && <ArrowRight className="w-4 h-4" />}
          </Button>
        </form>

        <p className="mt-6 text-center text-[13px] text-ink-500">
          Already have an account?{' '}
          <Link href="/login" className="font-medium text-primary-600 hover:text-primary-700">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
