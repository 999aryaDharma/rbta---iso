import { useState } from "react";

export type LiveContext = "live" | "replay";
export type SwitcherStatus = "idle" | "loading" | "error" | "empty";

interface LiveReplaySwitcherProps {
  mode: LiveContext;
  onChange: (mode: LiveContext) => void;
  status?: SwitcherStatus;
  errorMessage?: string;
  disabled?: boolean;
}

const STATUS_LABEL: Record<Exclude<SwitcherStatus, "idle">, string> = {
  loading: "Memuat konteks…",
  error: "Gagal memuat konteks",
  empty: "Belum ada MetaAlert pada konteks ini",
};

export function LiveReplaySwitcher({
  mode,
  onChange,
  status = "idle",
  errorMessage,
  disabled = false,
}: LiveReplaySwitcherProps) {
  const [pending, setPending] = useState<LiveContext | null>(null);
  const busy = status === "loading" || pending !== null;

  const select = (next: LiveContext) => {
    if (next === mode || disabled || busy) return;
    setPending(next);
    try {
      onChange(next);
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="p-4 rounded-xl border border-kumo-hairline bg-kumo-canvas shadow-xs space-y-3">
      <div
        role="radiogroup"
        aria-label="Konteks data: live atau replay"
        className="flex gap-2"
      >
        {(["live", "replay"] as const).map((value) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={mode === value}
            aria-pressed={mode === value}
            disabled={disabled || busy}
            onClick={() => select(value)}
            className="px-4 py-2 rounded-lg text-sm font-semibold border border-kumo-hairline data-[active=true]:bg-kumo-brand"
            data-active={mode === value}
          >
            {value === "live" ? "Live" : "Replay"}
          </button>
        ))}
      </div>

      {status === "loading" && (
        <p role="status" className="text-xs text-kumo-subtle">
          {STATUS_LABEL.loading}
        </p>
      )}
      {status === "error" && (
        <p role="alert" className="text-xs text-red-600">
          {STATUS_LABEL.error}
          {errorMessage ? `: ${errorMessage}` : ""}
        </p>
      )}
      {status === "empty" && (
        <p role="status" className="text-xs text-kumo-subtle">
          {STATUS_LABEL.empty}
        </p>
      )}
      {status === "idle" && (
        <p className="text-xs text-kumo-subtle">
          Konteks aktif:{" "}
          {mode === "live"
            ? "Live — MetaAlert live memakai kontrak provenance yang sama dengan replay."
            : "Replay — dataset historis untuk demo sidang."}
        </p>
      )}
    </div>
  );
}
