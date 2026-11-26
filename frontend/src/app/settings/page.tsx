'use client';

import { useEffect, useState } from 'react';
import { CheckCircle2 } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { api, apiErrorMessage } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { Card, CardHeader, Button, Input, Select, Toggle, Skeleton } from '@/components/ui';
import { ErrorState, PageHeader } from '@/components/states';

interface SettingsData {
  profile_data: Record<string, unknown> | null;
  security_data: Record<string, unknown> | null;
  notifications_data: Record<string, unknown> | null;
}

export default function SettingsPage() {
  const { user } = useAuth();
  const settings = useApi<SettingsData>('/settings/');
  const [profile, setProfile] = useState<Record<string, unknown>>({});
  const [security, setSecurity] = useState<Record<string, unknown>>({});
  const [notifications, setNotifications] = useState<Record<string, unknown>>({});
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (settings.data && !loaded) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time hydration of the form from fetched data
      setProfile((settings.data.profile_data as Record<string, unknown>) ?? {});
      setSecurity((settings.data.security_data as Record<string, unknown>) ?? {});
      setNotifications((settings.data.notifications_data as Record<string, unknown>) ?? {});
      setLoaded(true);
    }
  }, [settings.data, loaded]);

  const save = async () => {
    setSaving(true);
    setSuccess('');
    setError('');
    try {
      await api.put('/settings', {
        profile_data: profile,
        security_data: security,
        notifications_data: notifications,
      });
      setSuccess('Settings saved.');
      setTimeout(() => setSuccess(''), 4000);
    } catch (err) {
      setError(apiErrorMessage(err, 'saving settings'));
    } finally {
      setSaving(false);
    }
  };

  if (settings.loading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Settings" subtitle="Profile, notifications and security preferences." />
        {[0, 1].map((i) => (
          <Card key={i} className="p-6">
            <Skeleton className="h-40" />
          </Card>
        ))}
      </div>
    );
  }

  if (settings.error) {
    return (
      <ErrorState message={apiErrorMessage(settings.error, 'settings')} onRetry={settings.refetch} />
    );
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <PageHeader
        title="Settings"
        subtitle="Profile, notifications and security preferences for your workspace."
        actions={
          <Button onClick={save} loading={saving}>
            Save changes
          </Button>
        }
      />

      {success && (
        <div className="rounded-xl bg-positive-50 border border-positive-600/15 px-4 py-3 text-[13px] text-positive-600 flex items-center gap-2 animate-fade-in">
          <CheckCircle2 className="w-4 h-4" /> {success}
        </div>
      )}
      {error && (
        <div className="rounded-xl bg-negative-50 border border-negative-600/15 px-4 py-3 text-[13px] text-negative-600 animate-fade-in">
          {error}
        </div>
      )}

      {/* Account */}
      <Card className="animate-fade-up">
        <CardHeader title="Account" subtitle="Your sign-in identity for this workspace" />
        <div className="px-6 pb-6 space-y-3">
          <div className="flex items-center gap-3.5">
            <span className="w-11 h-11 rounded-full bg-ink-950 text-white flex items-center justify-center text-[13px] font-semibold">
              {(user?.email ?? 'U').slice(0, 2).toUpperCase()}
            </span>
            <div>
              <p className="text-[13.5px] font-semibold text-ink-950">{user?.email}</p>
              <p className="text-[12px] text-ink-500">Business: {user?.business_name}</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Profile */}
      <Card className="animate-fade-up stagger-1">
        <CardHeader title="Profile" subtitle="How FinMind addresses you" />
        <div className="px-6 pb-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Display name</label>
            <Input
              placeholder="Your name"
              value={(profile.name as string) ?? ''}
              onChange={(e) => setProfile({ ...profile, name: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-ink-700 mb-1.5">Language</label>
            <Select
              value={(profile.language as string) ?? 'en'}
              onChange={(e) => setProfile({ ...profile, language: e.target.value })}
            >
              <option value="en">English</option>
              <option value="ms">Bahasa Melayu</option>
              <option value="zh">中文</option>
            </Select>
          </div>
        </div>
      </Card>

      {/* Notifications */}
      <Card className="animate-fade-up stagger-2">
        <CardHeader title="Notification preferences" subtitle="Choose what FinMind tells you about" />
        <div className="px-6 pb-6 space-y-4">
          <Toggle
            checked={!!notifications.email}
            onChange={(v) => setNotifications({ ...notifications, email: v })}
            label="Important account alerts"
            description="Security notices and significant account events."
          />
          <Toggle
            checked={!!notifications.push}
            onChange={(v) => setNotifications({ ...notifications, push: v })}
            label="Financial insights"
            description="Anomaly findings and forecast highlights."
          />
        </div>
      </Card>

      {/* Security */}
      <Card className="animate-fade-up stagger-3">
        <CardHeader title="Security preferences" subtitle="Saved to your business security profile" />
        <div className="px-6 pb-6 space-y-4">
          <Toggle
            checked={!!security['2fa']}
            onChange={(v) => setSecurity({ ...security, '2fa': v })}
            label="Two-factor authentication preference"
            description="Records your preference for an additional sign-in factor."
          />
          <p className="text-[12px] text-ink-400 leading-relaxed">
            Sessions are protected by short-lived signed tokens and refresh automatically only with
            your stored credentials. Privacy-sensitive AI requests are routed exclusively to local
            models and never to the cloud.
          </p>
        </div>
      </Card>
    </div>
  );
}
