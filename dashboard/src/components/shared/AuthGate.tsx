import { useState, useEffect, type ReactNode } from 'react';
import { isAuthenticated, setApiKey, clearApiKey } from '@/lib/auth';
import { checkAuth } from '@/api/auth';
import { Alert } from '@/components/ui/alert';
import { Shield, AlertTriangle, KeyRound } from 'lucide-react';

export function AuthGate({ children }: { children: ReactNode }) {
  const [authed, setAuthed] = useState(isAuthenticated());
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    const handleUnauthorized = () => {
      setAuthed(false);
      setError('Session expired or unauthorized. Please re-enter your API key.');
    };
    window.addEventListener('rbta:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('rbta:unauthorized', handleUnauthorized);
  }, []);

  const handleSignIn = async () => {
    const trimmed = input.trim();
    if (!trimmed) {
      setError('API key cannot be empty.');
      return;
    }

    setIsVerifying(true);
    setError(null);

    try {
      setApiKey(trimmed);
      const ok = await checkAuth();
      if (ok) {
        setAuthed(true);
      } else {
        clearApiKey();
        setError('Invalid API key. Please check your credentials.');
      }
    } catch {
      clearApiKey();
      setError('Failed connecting to REST API service.');
    } finally {
      setIsVerifying(false);
    }
  };

  if (authed) return <>{children}</>;

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ background: 'var(--bg-app)' }}>
      <div
        className="p-8 rounded-[7px] border w-full max-w-sm shadow-sm"
        style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex items-center gap-2 mb-2">
          <Shield size={20} style={{ color: 'var(--brand-orange)' }} />
          <h1 className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
            RBTA Security Analytics
          </h1>
        </div>
        <p className="text-xs mb-4" style={{ color: 'var(--text-secondary)' }}>
          Enter your authorized operational API key to access the control plane.
        </p>

        {error && (
          <Alert variant="danger" className="mb-4">
            <AlertTriangle size={14} className="shrink-0 mt-0.5" />
            <div className="text-xs">{error}</div>
          </Alert>
        )}

        <div className="space-y-3">
          <div className="relative">
            <KeyRound size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-disabled)' }} />
            <input
              type="password"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSignIn();
                }
              }}
              placeholder="Enter RBTA API Key..."
              className="w-full pl-8 pr-3 py-2 text-xs font-mono rounded-[5px] border outline-none focus:ring-1 focus:ring-[var(--action-blue)]"
              style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
              autoFocus
            />
          </div>

          <button
            onClick={handleSignIn}
            disabled={isVerifying || !input.trim()}
            className="w-full py-2 text-xs font-medium text-white rounded-[5px] cursor-pointer disabled:opacity-50 transition-opacity"
            style={{ background: 'var(--action-blue)' }}
          >
            {isVerifying ? 'Verifying Credential...' : 'Sign In to Control Plane'}
          </button>
        </div>

        <p className="mt-4 text-[11px] text-center" style={{ color: 'var(--text-tertiary)' }}>
          Session key is securely held in browser memory (<code className="font-mono">sessionStorage</code>).
        </p>
      </div>
    </div>
  );
}
