import { useEffect, useState } from "react";
import { Server, Cpu } from "lucide-react";
import { cn } from "../lib/utils";

interface ExecutorStatus {
  mode: "local" | "remote";
  remote_url: string | null;
  healthy: boolean | null;
  last_check: number | null;
  gpu_info: Record<string, unknown>;
  models_loaded: string[];
  error?: string | null;
}

export function ExecutorBadge() {
  const [status, setStatus] = useState<ExecutorStatus | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/system/executor");
      if (res.ok) setStatus(await res.json());
    } catch {
      // silently ignore — badge just won't render
    }
  };

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!status) return null;

  if (status.mode === "local") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Cpu size={13} />
        Local
      </span>
    );
  }

  const ok = status.healthy;
  const gpuName = (status.gpu_info as any)?.name ?? "";
  const tooltip = ok
    ? `Remote GPU worker healthy${gpuName ? ` · ${gpuName}` : ""}${
        status.models_loaded.length
          ? ` · ${status.models_loaded.length} model(s) loaded`
          : ""
      }`
    : `Remote GPU worker unreachable${status.error ? ` — ${status.error}` : ""}`;

  return (
    <span
      className={cn(
        "flex items-center gap-1.5 text-xs",
        ok ? "text-green-500" : "text-destructive",
      )}
      title={tooltip}
    >
      <Server size={13} />
      Remote {ok ? "✓" : "✗"}
    </span>
  );
}
