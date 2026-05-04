import { useEffect, useRef, useState } from "react";
import { Server, Cpu, ChevronDown, AlertTriangle } from "lucide-react";
import { cn } from "../lib/utils";

type OverrideMode = "local" | "remote" | null;

interface ExecutorStatus {
  env_default: "local" | "remote";
  override: OverrideMode;
  effective: "local" | "remote";
  remote_url_present: boolean;
  healthy: boolean | null;
  api_version?: number;
  last_check: number | null;
  gpu_info: Record<string, unknown>;
  models_loaded: string[];
  error?: string | null;
}

export function ExecutorBadge() {
  const [status, setStatus] = useState<ExecutorStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/system/executor");
      if (res.ok) setStatus(await res.json());
    } catch {
      // silently ignore
    }
  };

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 30_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const setOverride = async (value: OverrideMode) => {
    setSaving(true);
    try {
      const res = await fetch("/api/system/executor", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ override: value }),
      });
      if (res.ok) {
        await fetchStatus();
        setOpen(false);
      }
    } finally {
      setSaving(false);
    }
  };

  if (!status) return null;

  const effective = status.effective;
  const gpuName = (status.gpu_info as any)?.name ?? "";
  const workerOutdated = effective === "remote" && status.healthy && (status.api_version ?? 1) < 2;

  const label = effective === "remote"
    ? `Remote${status.healthy ? " ✓" : " ✗"}`
    : "Local";

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex items-center gap-1.5 text-xs rounded px-1.5 py-0.5 hover:bg-muted transition-colors",
          effective === "remote" && status.healthy && !workerOutdated ? "text-green-500" : "",
          effective === "remote" && (!status.healthy || workerOutdated) ? "text-yellow-500" : "",
          effective === "local" ? "text-muted-foreground" : "",
        )}
        title={
          effective === "remote"
            ? workerOutdated
              ? `Remote GPU (v${status.api_version ?? 1}) is outdated — commit & push latest code, then restart Colab worker`
              : `Remote GPU${gpuName ? ` · ${gpuName}` : ""}${status.models_loaded.length ? ` · ${status.models_loaded.length} model(s)` : ""}${!status.healthy && status.error ? ` — ${status.error}` : ""}`
            : "Running locally"
        }
      >
        {effective === "remote" ? <Server size={13} /> : <Cpu size={13} />}
        {label}
        {workerOutdated && <AlertTriangle size={11} className="text-yellow-500" />}
        <ChevronDown size={11} className={cn("transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 bg-popover border rounded-lg shadow-md p-2 min-w-[200px] space-y-1">
          <p className="text-[10px] text-muted-foreground px-2 pb-1">Executor for new jobs</p>

          {workerOutdated && (
            <div className="mx-2 mb-1 px-2 py-1.5 rounded bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-[10px] text-yellow-700 dark:text-yellow-400">
              Worker v{status.api_version ?? 1} is outdated. Push latest code &amp; restart Colab worker.
            </div>
          )}

          {(["auto", "local", "remote"] as const).map((opt) => {
            const current = status.override === null ? "auto" : status.override;
            const disabled = opt === "remote" && !status.remote_url_present;
            return (
              <button
                key={opt}
                disabled={saving || disabled}
                onClick={() => setOverride(opt === "auto" ? null : opt)}
                className={cn(
                  "w-full text-left text-xs px-2 py-1.5 rounded transition-colors",
                  current === opt ? "bg-primary/10 text-primary font-medium" : "hover:bg-muted",
                  disabled && "opacity-40 cursor-not-allowed",
                )}
              >
                {opt === "auto" && `Auto (${status.env_default})`}
                {opt === "local" && "Force Local"}
                {opt === "remote" && (
                  <span>
                    Force Remote
                    {!status.remote_url_present && <span className="ml-1 text-[10px] text-muted-foreground">(no URL)</span>}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
