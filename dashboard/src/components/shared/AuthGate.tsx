import { useState, useEffect, type ReactNode } from 'react';
import { isAuthenticated, setApiKey, clearApiKey } from '@/lib/auth';
import { checkAuth } from '@/api/auth';
import { Button } from '@cloudflare/kumo/components/button';
import { InputGroup } from '@cloudflare/kumo/components/input-group';
import { Banner } from '@cloudflare/kumo/components/banner';
import { Shield, Key } from '@phosphor-icons/react';

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
    <div className="min-h-screen flex items-center justify-center p-4 bg-kumo-canvas">
      <div className="p-8 rounded-lg border border-kumo-hairline bg-kumo-base w-full max-w-sm">
        <div className="flex items-center gap-2 mb-2">
          <Shield size={20} className="text-kumo-brand" />
          <h1 className="text-sm font-semibold tracking-tight text-kumo-default">
            RBTA Security Analytics
          </h1>
        </div>
        <p className="text-xs mb-4 text-kumo-subtle">
          Enter your authorized operational API key to access the control plane.
        </p>

        {error && (
          <div className="mb-4">
            <Banner variant="error" size="sm" description={error} />
          </div>
        )}

        <div className="space-y-3">
          <InputGroup>
            <InputGroup.Addon align="start"><Key size={14} /></InputGroup.Addon>
            <InputGroup.Input
              type="password"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSignIn();
                }
              }}
              placeholder="Enter RBTA API Key..."
              className="font-mono text-xs"
              autoFocus
            />
          </InputGroup>

          <Button
            onClick={handleSignIn}
            disabled={isVerifying || !input.trim()}
            className="w-full justify-center"
            variant="primary"
          >
            {isVerifying ? 'Verifying Credential...' : 'Sign In to Control Plane'}
          </Button>
        </div>

        <p className="mt-4 text-[11px] text-center text-kumo-inactive">
          Session key is securely held in browser memory (<code className="font-mono">sessionStorage</code>).
        </p>
      </div>
    </div>
  );
}
