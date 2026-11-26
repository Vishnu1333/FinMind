'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, ArrowRight, MailCheck } from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import { Button, Input } from '@/components/ui';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [resendIn, setResendIn] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (api.getToken()) {
      router.replace('/dashboard');
      return;
    }
    const reason = new URLSearchParams(window.location.search).get('reason');
    if (reason === 'expired') setNotice('Your session has expired. Please sign in again.');
  }, [router]);

  useEffect(() => {
    if (resendIn <= 0) return;
    const timer = window.setTimeout(() => setResendIn((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearTimeout(timer);
  }, [resendIn]);

  const requestCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setNotice('');
    setIsLoading(true);
    try {
      api.clearToken();
      const result = await api.post<{ detail: string; resend_after_seconds: number }>('/auth/login', {
        email: email.trim().toLowerCase(),
        password,
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
      const result = await api.post<{ access_token: string }>('/auth/login/verify', {
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
        purpose: 'login',
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
    <div className="min-h-screen flex flex-col lg:flex-row bg-canvas">
      <div className="hidden lg:flex w-[46%] bg-ink-950 relative overflow-hidden flex-col justify-between p-12">
        <div className="absolute -top-40 -right-40 w-[480px] h-[480px] rounded-full opacity-25 blur-3xl" style={{ background: 'radial-gradient(circle, #7C3AED, transparent 65%)' }} />
        <div className="flex items-center gap-2.5 relative">
          <span className="w-9 h-9 rounded-xl brand-gradient flex items-center justify-center"><Sparkles className="w-4.5 h-4.5 text-white" /></span>
          <span className="text-lg font-semibold tracking-tight text-white">FIN<span className="text-primary-400">MIND</span></span>
        </div>
        <div className="relative">
          <h1 className="text-[34px] leading-[1.15] font-semibold tracking-tight text-white max-w-md text-balance">Financial intelligence for your business, in one calm place.</h1>
          <p className="mt-4 text-[15px] text-ink-300 max-w-md leading-relaxed">Accounts, transactions, cash-flow forecasting and document intelligence — with an AI assistant that answers from your real data.</p>
          <div className="mt-8 flex flex-wrap gap-2">{['Cash-flow forecasts', 'Anomaly detection', 'Document intelligence', 'AI assistant'].map((tag) => <span key={tag} className="px-3 py-1 rounded-full bg-white/[0.06] border border-white/[0.08] text-[12px] text-ink-200">{tag}</span>)}</div>
        </div>
        <p className="relative text-[12px] text-ink-500">Simulated banking environment for SME financial intelligence.</p>
      </div>

      <div className="flex-1 flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-[400px] animate-fade-up">
          <div className="lg:hidden flex items-center gap-2.5 mb-10">
            <span className="w-9 h-9 rounded-xl brand-gradient flex items-center justify-center"><Sparkles className="w-4.5 h-4.5 text-white" /></span>
            <span className="text-lg font-semibold tracking-tight text-ink-950">FIN<span className="text-primary-600">MIND</span></span>
          </div>
          <h2 className="text-[24px] font-semibold tracking-tight text-ink-950">{otpSent ? 'Check your email' : 'Welcome back'}</h2>
          <p className="text-[14px] text-ink-500 mt-1">{otpSent ? `Enter the six-digit code sent to ${email.trim().toLowerCase()}. It expires in 5 minutes.` : 'Sign in to your FinMind workspace. Email verification is required.'}</p>

          {notice && <div className="mt-6 rounded-xl bg-caution-50 border border-caution-600/15 px-4 py-3 text-[13px] text-caution-600">{notice}</div>}
          {error && <div className="mt-6 rounded-xl bg-negative-50 border border-negative-600/15 px-4 py-3 text-[13px] text-negative-600">{error}</div>}

          {!otpSent ? (
            <form onSubmit={requestCode} className="mt-8 space-y-4">
              <div>
                <label htmlFor="email" className="block text-[13px] font-medium text-ink-700 mb-1.5">Email address</label>
                <Input id="email" type="email" required autoComplete="email" placeholder="you@business.com" value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>
              <div>
                <label htmlFor="password" className="block text-[13px] font-medium text-ink-700 mb-1.5">Password</label>
                <Input id="password" type="password" required autoComplete="current-password" placeholder="Your password" value={password} onChange={(e) => setPassword(e.target.value)} />
              </div>
              <Button type="submit" loading={isLoading} className="w-full mt-2">Continue with email OTP {!isLoading && <ArrowRight className="w-4 h-4" />}</Button>
            </form>
          ) : (
            <form onSubmit={verifyCode} className="mt-8 space-y-4">
              <div>
                <label htmlFor="otp" className="block text-[13px] font-medium text-ink-700 mb-1.5">Email verification code</label>
                <Input id="otp" type="text" inputMode="numeric" autoComplete="one-time-code" required minLength={6} maxLength={6} pattern="[0-9]{6}" placeholder="000000" value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))} />
              </div>
              <Button type="submit" loading={isLoading} className="w-full">Verify and sign in {!isLoading && <MailCheck className="w-4 h-4" />}</Button>
              <button type="button" onClick={resendCode} disabled={isLoading || resendIn > 0} className="w-full text-[13px] font-medium text-primary-600 disabled:text-ink-400">{resendIn > 0 ? `Resend code in ${resendIn}s` : 'Resend verification code'}</button>
              <button type="button" onClick={() => { setOtpSent(false); setCode(''); setNotice(''); setError(''); }} className="w-full text-[12px] text-ink-500">Back to sign in</button>
            </form>
          )}

          <p className="mt-6 text-center text-[13px] text-ink-500">New to FinMind? <Link href="/register" className="font-medium text-primary-600 hover:text-primary-700">Create a business account</Link></p>
        </div>
      </div>
    </div>
  );
}
