'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useRouter } from 'next/navigation';
import { api, setUnauthorizedHandler } from '@/lib/api';

export interface CurrentUser {
  id: number;
  email: string;
  business_id: number;
  business_name: string;
}

interface AuthContextValue {
  user: CurrentUser | null;
  loading: boolean;
  logout: (reason?: string) => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  logout: () => {},
  refresh: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);

  const logout = useCallback(
    (reason?: string) => {
      api.clearToken();
      setUser(null);
      const params = reason ? `?reason=${encodeURIComponent(reason)}` : '';
      router.replace(`/login${params}`);
    },
    [router],
  );

  const refresh = useCallback(async () => {
    if (!api.getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await api.get<CurrentUser>('/auth/me');
      setUser(me);
    } catch {
      // api layer already cleared the token and called the 401 handler
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => logout('expired'));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial auth check synchronises with the API
    void refresh();
  }, [refresh, logout]);

  const value = useMemo(
    () => ({ user, loading, logout, refresh }),
    [user, loading, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
