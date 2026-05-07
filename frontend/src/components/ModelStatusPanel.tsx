import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, CheckCircle, XCircle, Loader, Clock } from "lucide-react";
import { useModelStatus } from "../hooks/useModelStatus";
import { fmtBytes } from "../lib/utils";
import { cn } from "../lib/utils";

type Sample = { t: number; bytes: number };

function useSpeed(name: string, status: string, bytes: number) {
  const lastRef = useRef<Sample | null>(null);
  const [speed, setSpeed] = useState<number>(0); // bytes/sec

  useEffect(() => {
    if (status !== "downloading") {
      lastRef.current = null;
      setSpeed(0);
      return;
    }
    const now = Date.now();
    const prev = lastRef.current;
    if (prev && now > prev.t && bytes >= prev.bytes) {
      const dt = (now - prev.t) / 1000;
      const db = bytes - prev.bytes;
      // smooth slightly: blend new sample with previous speed
      const inst = db / dt;
      setSpeed((s) => (s > 0 ? s * 0.5 + inst * 0.5 : inst));
    }
    lastRef.current = { t: now, bytes };
  }, [name, status, bytes]);

  return speed;
}

function fmtEta(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

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
          {models.map((m) => (
            <ModelRow key={m.name} model={m as ModelRowProps["model"]} />
          ))}
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

interface ModelRowProps {
  model: {
    name: string;
    status: string;
    bytes_downloaded: number;
    bytes_total: number;
    error_message: string | null;
    attempt?: number;
    note?: string;
  };
}

function ModelRow({ model: m }: ModelRowProps) {
  const speed = useSpeed(m.name, m.status, m.bytes_downloaded);
  const pct = m.bytes_total > 0 ? Math.round((m.bytes_downloaded / m.bytes_total) * 100) : 0;
  const remaining = m.bytes_total > 0 ? m.bytes_total - m.bytes_downloaded : 0;
  const etaSeconds = speed > 0 && remaining > 0 ? remaining / speed : 0;

  let rightLabel: string;
  if (m.status === "downloading" && m.bytes_total > 0) {
    rightLabel = `${fmtBytes(m.bytes_downloaded)} / ${fmtBytes(m.bytes_total)}`;
  } else if (m.status === "downloading" && m.bytes_downloaded > 0) {
    rightLabel = fmtBytes(m.bytes_downloaded);
  } else {
    rightLabel = m.status;
  }
  if (m.status === "downloading" && m.attempt && m.attempt > 1) {
    rightLabel += ` (attempt ${m.attempt}/4)`;
  }

  const speedLine =
    m.status === "downloading" && speed > 0
      ? `${fmtBytes(speed)}/s${etaSeconds > 0 ? ` · ETA ${fmtEta(etaSeconds)}` : ""}`
      : "";

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm truncate">{MODEL_LABELS[m.name] ?? m.name}</span>
        <span
          className={cn(
            "text-xs ml-2 shrink-0",
            m.status === "done" && "text-green-500",
            m.status === "failed" && "text-destructive",
            m.status === "downloading" && "text-primary",
            m.status === "queued" && "text-muted-foreground",
          )}
        >
          {rightLabel}
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
      {(m.note || speedLine) && (
        <p className="text-xs text-muted-foreground mt-0.5">
          {[m.note, speedLine].filter(Boolean).join(" · ")}
        </p>
      )}
      {m.error_message && (
        <p className="text-xs text-destructive mt-0.5">{m.error_message}</p>
      )}
    </div>
  );
}
