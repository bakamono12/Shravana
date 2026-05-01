import { useState } from "react";
import { ChevronDown, ChevronUp, CheckCircle, XCircle, Loader, Clock } from "lucide-react";
import { useModelStatus } from "../hooks/useModelStatus";
import { fmtBytes } from "../lib/utils";
import { cn } from "../lib/utils";

const MODEL_LABELS: Record<string, string> = {
  qwen_lid: "Qwen3-ASR-0.6B (Language ID)",
  qwen3_asr: "Qwen3-ASR-1.7B (Hindi / code-switch)",
  parakeet: "Parakeet TDT 1.1B (English)",
  whisper_turbo: "Whisper Large-v3-Turbo (Marathi / fallback)",
  forced_aligner: "Qwen3-ForcedAligner-0.6B",
};

export function ModelStatusPanel() {
  const { models } = useModelStatus();
  const [open, setOpen] = useState(false);

  const allDone = models.length > 0 && models.every((m) => m.status === "done");
  const anyFailed = models.some((m) => m.status === "failed");
  const downloading = models.filter((m) => m.status === "downloading").length;

  return (
    <div className="rounded-xl border bg-card overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-muted/40 transition-colors"
      >
        <div className="flex items-center gap-2">
          {allDone ? (
            <CheckCircle size={16} className="text-green-500" />
          ) : anyFailed ? (
            <XCircle size={16} className="text-destructive" />
          ) : downloading > 0 ? (
            <Loader size={16} className="text-primary animate-spin" />
          ) : (
            <Clock size={16} className="text-muted-foreground" />
          )}
          <span className="text-sm font-medium">
            {allDone
              ? "All models ready"
              : downloading > 0
              ? `Downloading models (${downloading} active)`
              : "Model status"}
          </span>
          {!allDone && (
            <span className="text-xs text-muted-foreground">
              {models.filter((m) => m.status === "done").length}/{models.length} ready
            </span>
          )}
        </div>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-border pt-3">
          {models.length === 0 && (
            <p className="text-sm text-muted-foreground">No models tracked yet.</p>
          )}
          {models.map((m) => {
            const pct = m.bytes_total > 0 ? Math.round((m.bytes_downloaded / m.bytes_total) * 100) : 0;
            return (
              <div key={m.name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm truncate">{MODEL_LABELS[m.name] ?? m.name}</span>
                  <span className={cn(
                    "text-xs ml-2 shrink-0",
                    m.status === "done" && "text-green-500",
                    m.status === "failed" && "text-destructive",
                    m.status === "downloading" && "text-primary",
                    m.status === "queued" && "text-muted-foreground",
                  )}>
                    {m.status === "downloading" && m.bytes_total > 0
                      ? `${fmtBytes(m.bytes_downloaded)} / ${fmtBytes(m.bytes_total)}`
                      : m.status}
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      m.status === "done" && "bg-green-500",
                      m.status === "failed" && "bg-destructive",
                      m.status === "downloading" && "bg-primary",
                      m.status === "queued" && "bg-muted-foreground/40",
                    )}
                    style={{ width: m.status === "done" ? "100%" : `${pct}%` }}
                  />
                </div>
                {m.error_message && (
                  <p className="text-xs text-destructive mt-0.5">{m.error_message}</p>
                )}
              </div>
            );
          })}
          {!allDone && (
            <p className="text-xs text-muted-foreground">
              Total model size: ~20 GB. Uploads will queue until required models are ready.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
