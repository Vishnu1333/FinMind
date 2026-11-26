const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Session-expiry message rendered by pages when a 401 redirects to login. */
export const SESSION_EXPIRED_MSG =
  'Your session has expired. Please sign in again.';

let onUnauthorized: (() => void) | null = null;
/** Register the global handler invoked on any 401 (used by the auth context). */
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

async function handleResponse<T>(response: Response): Promise<T> {
  const isJson = response.headers.get('content-type')?.includes('application/json');
  const data = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    if (response.status === 401) {
      api.clearToken();
      onUnauthorized?.();
    }
    let detail: string;
    if (typeof data === 'object' && data !== null && 'detail' in data) {
      const rawDetail = (data as Record<string, unknown>).detail;
      if (typeof rawDetail === 'string') {
        detail = rawDetail;
      } else if (Array.isArray(rawDetail)) {
        detail = rawDetail
          .map((item: any) => (typeof item === 'object' && item?.msg ? item.msg : String(item)))
          .join('; ');
      } else {
        detail = JSON.stringify(rawDetail);
      }
    } else {
      detail = typeof data === 'string' && data ? data : response.statusText || 'Request failed';
    }
    throw new ApiError(response.status, detail);
  }

  return data as T;
}

export const api = {
  getToken: () => {
    if (typeof window !== 'undefined') {
      return window.localStorage.getItem('finmind_token');
    }
    return null;
  },

  setToken: (token: string) => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('finmind_token', token);
    }
  },

  clearToken: () => {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem('finmind_token');
    }
  },

  getHeaders: (json = true, isPublicAuth = false): Record<string, string> => {
    const headers: Record<string, string> = {};
    if (json) headers['Content-Type'] = 'application/json';
    if (!isPublicAuth) {
      const token = api.getToken();
      if (token) headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  },

  get: async <T>(endpoint: string): Promise<T> => {
    const isPublicAuth = endpoint.startsWith('/auth/register') || endpoint.startsWith('/auth/login');
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'GET',
      headers: api.getHeaders(true, isPublicAuth),
    });
    return handleResponse<T>(response);
  },

  post: async <T>(
    endpoint: string,
    body?: unknown,
    isFormData: boolean = false,
  ): Promise<T> => {
    const isPublicAuth = endpoint.startsWith('/auth/register') || endpoint.startsWith('/auth/login');
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: api.getHeaders(!isFormData, isPublicAuth),
      body: isFormData ? (body as FormData) : JSON.stringify(body ?? {}),
    });
    return handleResponse<T>(response);
  },

  put: async <T>(endpoint: string, body?: unknown): Promise<T> => {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'PUT',
      headers: api.getHeaders(),
      body: JSON.stringify(body ?? {}),
    });
    return handleResponse<T>(response);
  },

  delete: async <T>(endpoint: string): Promise<T> => {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'DELETE',
      headers: api.getHeaders(),
    });
    return handleResponse<T>(response);
  },
};

/** Human-friendly message for API errors shown in the UI. */
export function apiErrorMessage(err: unknown, context: string): string {
  if (typeof err === 'string') {
    const lower = err.toLowerCase();
    if (
      err === SESSION_EXPIRED_MSG ||
      lower.includes('credential') ||
      lower.includes('unauthorized') ||
      lower.includes('expired') ||
      lower.includes('token')
    ) {
      return SESSION_EXPIRED_MSG;
    }
    if (
      lower.includes('unreachable') ||
      lower.includes('failed to fetch') ||
      lower.includes('networkerror') ||
      lower.includes('connection')
    ) {
      return `Unable to load ${context}. The service is unreachable — check that the FinMind API is running, then try again.`;
    }
    return err || `Unable to load ${context}.`;
  }
  if (err instanceof ApiError) {
    if (err.status === 0 || err.status >= 500) {
      return `Unable to load ${context}. The service is unreachable — check that the FinMind API is running, then try again.`;
    }
    if (err.status === 401) return SESSION_EXPIRED_MSG;
    return err.message || `Unable to load ${context}.`;
  }
  return `Unable to load ${context}. Check your connection and try again.`;
}
