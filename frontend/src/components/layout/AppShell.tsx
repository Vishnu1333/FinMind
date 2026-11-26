'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  LayoutDashboard,
  Wallet,
  ArrowLeftRight,
  CreditCard,
  LineChart,
  TrendingUp,
  ShieldAlert,
  FileText,
  Sparkles,
  Bell,
  LifeBuoy,
  Settings,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { cn } from '@/components/ui';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_SECTIONS: { label: string; items: NavItem[] }[] = [
  {
    label: 'Overview',
    items: [{ name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard }],
  },
  {
    label: 'Banking',
    items: [
      { name: 'Accounts', href: '/accounts', icon: Wallet },
      { name: 'Transactions', href: '/transactions', icon: ArrowLeftRight },
      { name: 'Cards', href: '/cards', icon: CreditCard },
    ],
  },
  {
    label: 'Financial Intelligence',
    items: [
      { name: 'Cash Flow', href: '/cash-flow', icon: LineChart },
      { name: 'Analytics', href: '/analytics', icon: TrendingUp },
      { name: 'Predictions', href: '/predictions', icon: TrendingUp },
      { name: 'Anomalies', href: '/anomalies', icon: ShieldAlert },
    ],
  },
  {
    label: 'Knowledge & AI',
    items: [
      { name: 'Documents', href: '/documents', icon: FileText },
      { name: 'FinMind AI', href: '/assistant', icon: Sparkles },
    ],
  },
  {
    label: 'Support',
    items: [
      { name: 'Notifications', href: '/notifications', icon: Bell },
      { name: 'Help Desk', href: '/help-desk', icon: LifeBuoy },
      { name: 'Settings', href: '/settings', icon: Settings },
    ],
  },
];

const PAGE_TITLES: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/accounts': 'Accounts',
  '/transactions': 'Transactions',
  '/cards': 'Cards',
  '/cash-flow': 'Cash Flow',
  '/analytics': 'Analytics',
  '/predictions': 'Predictions',
  '/anomalies': 'Anomalies',
  '/documents': 'Documents',
  '/knowledge': 'Knowledge Search',
  '/assistant': 'FinMind AI',
  '/notifications': 'Notifications',
  '/help-desk': 'Help Desk',
  '/settings': 'Settings',
};

function Wordmark() {
  return (
    <Link href="/dashboard" className="flex items-center gap-2.5 px-2 focus-ring rounded-lg">
      <span className="w-8 h-8 rounded-xl brand-gradient flex items-center justify-center shadow-[0_2px_8px_rgba(79,70,229,0.4)]">
        <Sparkles className="w-4 h-4 text-white" />
      </span>
      <span className="text-[17px] font-semibold tracking-tight text-white">
        FIN<span className="text-primary-400">MIND</span>
      </span>
    </Link>
  );
}

function NavLink({ item, onNavigate, badge }: { item: NavItem; onNavigate?: () => void; badge?: number }) {
  const pathname = usePathname();
  const active = pathname === item.href || pathname.startsWith(item.href + '/');
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      className={cn(
        'focus-ring group relative flex items-center gap-3 h-9 px-3 rounded-xl text-[13.5px] font-medium transition-all duration-150',
        active
          ? 'bg-white/[0.08] text-white'
          : 'text-ink-300 hover:text-white hover:bg-white/[0.05]',
      )}
    >
      {active && (
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-r-full bg-primary-400" />
      )}
      <Icon className={cn('w-[18px] h-[18px] shrink-0', active ? 'text-primary-400' : '')} />
      <span className="truncate">{item.name}</span>
      {badge !== undefined && badge > 0 && (
        <span className="ml-auto tabular text-[10.5px] font-bold px-1.5 py-0.5 rounded-full bg-primary-500 text-white min-w-[18px] text-center">
          {badge > 9 ? '9+' : badge}
        </span>
      )}
    </Link>
  );
}

function SidebarNav({ onNavigate, unread }: { onNavigate?: () => void; unread: number }) {
  return (
    <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
      {NAV_SECTIONS.map((section) => (
        <div key={section.label}>
          <p className="px-3 mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-500">
            {section.label}
          </p>
          <div className="space-y-0.5">
            {section.items.map((item) => (
              <NavLink
                key={item.href}
                item={item}
                onNavigate={onNavigate}
                badge={item.href === '/notifications' ? unread : undefined}
              />
            ))}
          </div>
        </div>
      ))}
    </nav>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, logout } = useAuth();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [unread, setUnread] = useState(0);

  const isAuthPage = pathname === '/login' || pathname === '/register';
  const isRoot = pathname === '/';

  // Root redirects to the app entry point.
  useEffect(() => {
    if (isRoot) {
      router.replace(api.getToken() ? '/dashboard' : '/login');
    }
  }, [isRoot, router]);

  // Require authentication for every app page.
  useEffect(() => {
    if (!loading && !isAuthPage && !isRoot && !api.getToken()) {
      router.replace('/login');
    }
  }, [loading, isAuthPage, isRoot, router]);

  // Unread notifications badge (live, refreshed on route change).
  useEffect(() => {
    if (isAuthPage || isRoot || !api.getToken()) return;
    let cancelled = false;
    api
      .get<{ unread_count: number }>('/notifications/unread-count')
      .then((res) => {
        if (!cancelled) setUnread(res.unread_count ?? 0);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [pathname, isAuthPage, isRoot]);

  if (isAuthPage || isRoot) {
    return <>{children}</>;
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-canvas">
        <div className="flex items-center gap-3 text-ink-400">
          <span className="w-8 h-8 rounded-xl brand-gradient animate-pulse" />
          <span className="text-sm font-medium">Loading FinMind…</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return <>{children}</>;
  }

  const title = PAGE_TITLES[pathname] ?? 'FinMind';
  const today = new Date().toLocaleDateString('en-MY', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });

  return (
    <div className="min-h-screen bg-canvas">
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex fixed inset-y-0 left-0 w-[248px] flex-col bg-ink-950 z-40">
        <div className="flex items-center h-16 px-5 shrink-0">
          <Wordmark />
        </div>
        <SidebarNav unread={unread} />
        <div className="p-3 border-t border-white/[0.06]">
          <div className="flex items-center gap-3 px-2 py-2">
            <span className="w-8 h-8 rounded-full bg-ink-700 text-ink-200 flex items-center justify-center text-[11px] font-semibold shrink-0">
              {user.email.slice(0, 2).toUpperCase()}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[12.5px] font-medium text-white truncate">{user.email}</p>
              <p className="text-[11px] text-ink-400 truncate">{user.business_name}</p>
            </div>
            <button
              onClick={() => logout()}
              title="Sign out"
              className="focus-ring p-1.5 rounded-lg text-ink-400 hover:text-white hover:bg-white/[0.06] transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Mobile drawer */}
      {drawerOpen && (
        <div className="lg:hidden fixed inset-0 z-50">
          <button
            aria-label="Close menu"
            className="absolute inset-0 bg-ink-950/50 animate-fade-in"
            onClick={() => setDrawerOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 w-[280px] bg-ink-950 flex flex-col animate-fade-in">
            <div className="flex items-center justify-between h-16 px-5">
              <Wordmark />
              <button
                onClick={() => setDrawerOpen(false)}
                className="focus-ring p-2 rounded-lg text-ink-300 hover:text-white"
                aria-label="Close menu"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <SidebarNav onNavigate={() => setDrawerOpen(false)} unread={unread} />
            <div className="p-3 border-t border-white/[0.06]">
              <button
                onClick={() => logout()}
                className="focus-ring w-full flex items-center gap-3 px-3 h-10 rounded-xl text-[13.5px] font-medium text-ink-300 hover:text-white hover:bg-white/[0.05]"
              >
                <LogOut className="w-[18px] h-[18px]" />
                Sign out
              </button>
            </div>
          </aside>
        </div>
      )}

      {/* Main column */}
      <div className="lg:pl-[248px] flex flex-col min-h-screen">
        {/* Topbar */}
        <header className="sticky top-0 z-30 h-16 bg-canvas/85 backdrop-blur-md border-b border-ink-950/[0.06]">
          <div className="h-full px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <button
                onClick={() => setDrawerOpen(true)}
                className="lg:hidden focus-ring p-2 -ml-2 rounded-lg text-ink-600 hover:bg-ink-100"
                aria-label="Open menu"
              >
                <Menu className="w-5 h-5" />
              </button>
              <div className="min-w-0">
                <h2 className="text-[15px] font-semibold text-ink-950 leading-tight truncate">
                  {title}
                </h2>
                <p className="text-[11.5px] text-ink-400 truncate hidden sm:block">{today}</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <Link
                href="/assistant"
                className="focus-ring hidden sm:flex items-center gap-1.5 h-9 px-3 rounded-xl text-[13px] font-medium text-primary-600 bg-primary-50 hover:bg-primary-50/70 transition-colors"
              >
                <Sparkles className="w-4 h-4" />
                Ask FinMind AI
              </Link>
              <Link
                href="/notifications"
                className="focus-ring relative p-2 rounded-xl text-ink-500 hover:text-ink-900 hover:bg-ink-100 transition-colors"
                aria-label="Notifications"
              >
                <Bell className="w-5 h-5" />
                {unread > 0 && (
                  <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-negative-500 ring-2 ring-canvas" />
                )}
              </Link>
              <button
                onClick={() => logout()}
                className="focus-ring hidden sm:block p-2 rounded-xl text-ink-500 hover:text-ink-900 hover:bg-ink-100 transition-colors"
                aria-label="Sign out"
                title="Sign out"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </header>

        <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 lg:py-8 max-w-[1200px] w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
