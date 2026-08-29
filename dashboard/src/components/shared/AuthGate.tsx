import { useState, type ReactNode } from 'react';
import { isAuthenticated, setApiKey } from '@/lib/auth';

export function AuthGate({ children }: { children: ReactNode }) {
  const [authed, setAuthed] = useState(isAuthenticated());
  const [input, setInput] = useState('');

  if (authed) return <>{children}</>;

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-app)' }}>
      <div className="p-8 rounded-[7px] border w-full max-w-sm" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
        <h1 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>RBTA Dashboard</h1>
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>Enter your API key to continue.</p>
        <input
          type="password"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && input.trim()) {
              setApiKey(input.trim());
              setAuthed(true);
            }
          }}
          placeholder="API Key"
          className="w-full px-3 py-2 text-sm rounded-[5px] border mb-3 outline-none focus:ring-2"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-surface)' }}
          autoFocus
        />
        <button
          onClick={() => {
            if (input.trim()) {
              setApiKey(input.trim());
              setAuthed(true);
            }
          }}
          className="w-full px-4 py-2 text-sm font-medium text-white rounded-[5px] cursor-pointer"
          style={{ background: 'var(--action-blue)' }}
        >
          Sign In
        </button>
      </div>
    </div>
  );
}
