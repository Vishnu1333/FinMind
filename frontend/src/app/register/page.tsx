'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, ArrowRight, MailCheck } from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import { Button, Input } from '@/components/ui';

export default function RegisterPage() {
  const router = useRouter();
  const [businessName, setBusinessName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [resendIn, setResendIn] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (resendIn <= 0) return;
    const timer = window.setTimeout(() => setResendIn((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearTimeout(timer);
  }, [resendIn]);

  const requestCode = async () => {
    setError('');
    setNotice('');
    setIsLoading(true);
    try {
      api.clearToken();
      const result = await api.post<{ detail: string; resend_after_seconds: number }>('/auth/register', {
        email: email.trim().toLowerCase(),
        password,
        business_name: businessName.trim(),
      });
      setOtpSent(true);
      setResendIn(result.resend_after_seconds || 60);
      setNotice(result.detail || 'A verification code was sent to your email.');
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'FinMind service is unavailable. Check that the API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      const result = await api.post<{ access_token: string }>('/auth/register/verify', {
        email: email.trim().toLowerCase(),
        code: code.trim(),
      });
      api.setToken(result.access_token);
      router.replace('/dashboard');
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to verify code. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const resendCode = async () => {
    setError('');
    setIsLoading(true);
    try {
      const result = await api.post<{ detail: string; resend_after_seconds: number }>('/auth/otp/resend', {
        email: email.trim().toLowerCase(),
        purpose: 'registration',
      });
      setResendIn(result.resend_after_seconds || 60);
      setNotice(result.detail);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not resend the code.');
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
          <span className="text-lg font-semibold tracking-tight text-ink-950">FIN<span className="text-primary-600">MIND</span></span>
        </div>

        <h1 className="text-[24px] font-semibold tracking-tight text-ink-950">
          {otpSent ? 'Verify your email' : 'Create your workspace'}
        </h1>
        <p className="text-[14px] text-ink-500 mt-1">
          {otpSent
            ? `Enter the six-digit code sent to ${email.trim().toLowerCase()}. The code expires in 5 minutes.`
            : 'Create your business workspace and verify your email to secure your account.'}
        </p>

        {notice && <div className="mt-6 rounded-xl bg-positive-50 border border-positive-600/15 px-4 py-3 text-[13px] text-positive-700">{notice}</div>}
        {error && <div className="mt-6 rounded-xl bg-negative-50 border border-negative-600/15 px-4 py-3 text-[13px] text-negative-600">{error}</div>}

        {!otpSent ? (
          <form onSubmit={(e) => { e.preventDefault(); void requestCode(); }} className="mt-8 space-y-4">
            <div>
              <label htmlFor="business" className="block text-[13px] font-medium text-ink-700 mb-1.5">Business name</label>
              <Input id="business" type="text" required minLength={1} placeholder="e.g. Warung Mak Lim Sdn Bhd" value={businessName} onChange={(e) => setBusinessName(e.target.value)} />
            </div>
            <div>
              <label htmlFor="email" className="block text-[13px] font-medium text-ink-700 mb-1.5">Email address</label>
              <Input id="email" type="email" required autoComplete="email" placeholder="you@business.com" value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <label htmlFor="password" className="block text-[13px] font-medium text-ink-700 mb-1.5">Password</label>
              <Input id="password" type="password" required minLength={8} autoComplete="new-password" placeholder="At least 8 characters" value={password} onChange={(e) => setPassword(e.target.value)} />
              <p className="text-[11.5px] text-ink-400 mt-1.5">Minimum 8 characters.</p>
            </div>
            <Button type="submit" loading={isLoading} className="w-full mt-2">Send verification code {!isLoading && <ArrowRight className="w-4 h-4" />}</Button>
          </form>
        ) : (
          <form onSubmit={verifyCode} className="mt-8 space-y-4">
            <div>
              <label htmlFor="otp" className="block text-[13px] font-medium text-ink-700 mb-1.5">Email verification code</label>
              <Input id="otp" type="text" inputMode="numeric" autoComplete="one-time-code" required minLength={6} maxLength={6} pattern="[0-9]{6}" placeholder="000000" value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))} />
            </div>
            <Button type="submit" loading={isLoading} className="w-full">Verify and create account {!isLoading && <MailCheck className="w-4 h-4" />}</Button>
            <button type="button" onClick={resendCode} disabled={isLoading || resendIn > 0} className="w-full text-[13px] font-medium text-primary-600 disabled:text-ink-400">
              {resendIn > 0 ? `Resend code in ${resendIn}s` : 'Resend verification code'}
            </button>
            <button type="button" onClick={() => { setOtpSent(false); setCode(''); setNotice(''); setError(''); }} className="w-full text-[12px] text-ink-500">Use a different email or edit details</button>
          </form>
        )}

        <p className="mt-6 text-center text-[13px] text-ink-500">
          Already have an account? <Link href="/login" className="font-medium text-primary-600 hover:text-primary-700">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
