import { useParams, Link, useNavigate } from "react-router-dom";
import { ArrowLeft, RefreshCw, CheckCircle, XCircle, Loader, Clock, AlertCircle, Trash2, Languages, Download } from "lucide-react";
import { useJob } from "../hooks/useJob";
import { MediaPreview } from "../components/MediaPreview";
import { api, LangInfo } from "../lib/api";
import { fmtDate, fmtEta } from "../lib/utils";
import { cn } from "../lib/utils";
import { useState, useEffect } from "react";

const PHASES = ["pending", "extracting", "denoising", "chunking", "processing", "translating", "assembling", "done"];

const PHASE_LABELS: Record<string, string> = {
  pending: "Queued",
  extracting: "Extracting audio",
  denoising: "Denoising & vocal separation",
  chunking: "Chunking audio",
  processing: "Transcribing",
  translating: "Translating",
  assembling: "Assembling subtitles",
  done: "Done",
  failed: "Failed",
  waiting_for_models: "Waiting for models",
};

function PhaseTimeline({ status }: { status: string }) {
  const currentIdx = PHASES.indexOf(status);
  return (
    <div
      className="grid w-full"
      style={{ gridTemplateColumns: `repeat(${PHASES.length}, 1fr)` }}
    >
      {PHASES.map((phase, i) => {
        const done = i < currentIdx || status === "done";
        const active = phase === status || (status === "processing" && phase === "processing");
        const failed = status === "failed" && i === currentIdx;
        return (
          <div key={phase} className="relative flex flex-col items-center min-w-0">
            {/* Connector: left:50% = this circle centre, right:-50% = next circle centre.
                top-3 = 12px = vertical centre of the 24px circle. */}
            {i < PHASES.length - 1 && (
              <div
                className={cn("absolute h-0.5 top-3", i < currentIdx ? "bg-green-500" : "bg-muted")}
                style={{ left: "50%", right: "-50%" }}
              />
            )}
            {/* Circle — z-10 paints over the connector */}
            <div className={cn(
              "relative z-10 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
              done && "bg-green-500 text-white",
              active && !done && "bg-primary text-primary-foreground",
              failed && "bg-destructive text-destructive-foreground",
              !done && !active && !failed && "bg-muted text-muted-foreground",
            )}>
              {done ? "✓" : i + 1}
            </div>
            {/* Label wraps within its column — can't push connectors out of place */}
            <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight px-1 hidden sm:block">
              {PHASE_LABELS[phase]}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { job, error } = useJob(jobId);
  const [retrying, setRetrying] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [translateTarget, setTranslateTarget] = useState("");
  const [supportedLangs, setSupportedLangs] = useState<Record<string, LangInfo>>({});

  useEffect(() => {
    api.listLanguages().then(setSupportedLangs).catch(() => {});
  }, []);

  const handleTranslate = async () => {
    if (!jobId || !translateTarget) return;
    setTranslating(true);
    try { await api.translateJob(jobId, translateTarget); } catch (err) {
      alert(err instanceof Error ? err.message : "Translation request failed");
    } finally { setTranslating(false); }
  };

  if (error) return (
    <div className="text-destructive">
      <Link to="/" className="flex items-center gap-1 text-sm mb-4"><ArrowLeft size={14} /> Back</Link>
      {error}
    </div>
  );

  if (!job) return (
    <div className="flex items-center gap-2 text-muted-foreground">
      <Loader size={16} className="animate-spin" /> Loading…
    </div>
  );

  const pct = job.total_chunks > 0 ? Math.round((job.completed_chunks / job.total_chunks) * 100) : 0;
  const isActive = !["done", "failed", "pending", "waiting_for_models"].includes(job.status);

  const retry = async () => {
    if (!jobId) return;
    setRetrying(true);
    try { await api.retryJob(jobId); } catch {} finally { setRetrying(false); }
  };

  const handleDelete = async () => {
    if (!jobId || !job) return;
    if (!confirm(`Delete job "${job.filename}"?`)) return;
    setDeleting(true);
    try {
      await api.deleteJob(jobId);
      navigate("/");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Delete failed");
      setDeleting(false);
    }
  };

  const ACTIVE_STATUSES = new Set(["extracting", "denoising", "chunking", "processing", "assembling"]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Link to="/" className="text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft size={18} />
        </Link>
        <h1 className="text-xl font-semibold truncate">{job.filename}</h1>
      </div>

      {/* Phase timeline */}
      <div className="bg-card border rounded-xl p-4">
        <PhaseTimeline status={job.status} />
      </div>

      {/* Progress */}
      <div className="bg-card border rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {job.status === "done" && <CheckCircle size={18} className="text-green-500" />}
            {job.status === "failed" && <XCircle size={18} className="text-destructive" />}
            {job.status === "waiting_for_models" && <AlertCircle size={18} className="text-yellow-500" />}
            {isActive && <Loader size={18} className="text-primary animate-spin" />}
            {job.status === "pending" && <Clock size={18} className="text-muted-foreground" />}
            <span className="font-medium">{PHASE_LABELS[job.status] ?? job.status}</span>
          </div>
          <div className="flex items-center gap-3">
            {["failed", "waiting_for_models"].includes(job.status) && (
              <button
                onClick={retry}
                disabled={retrying}
                className="flex items-center gap-1 text-sm text-primary hover:underline"
              >
                <RefreshCw size={14} className={cn(retrying && "animate-spin")} /> Retry
              </button>
            )}
            {!ACTIVE_STATUSES.has(job.status) && (
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="flex items-center gap-1 text-sm text-destructive hover:underline disabled:opacity-50"
              >
                <Trash2 size={14} /> {deleting ? "Deleting…" : "Delete"}
              </button>
            )}
          </div>
        </div>

        {job.total_chunks > 0 && (
          <>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{job.completed_chunks} / {job.total_chunks} chunks</span>
              <span>{pct}%</span>
            </div>
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </>
        )}

        <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
          {job.executor && (
            <div>
              <span className="text-foreground font-medium">Executor:</span>{" "}
              <span className={job.executor === "remote" ? "text-green-500" : ""}>
                {job.executor === "remote" ? "Remote GPU" : "Local"}
              </span>
            </div>
          )}
          {job.detected_language && (
            <div><span className="text-foreground font-medium">Source lang:</span> {job.detected_language.toUpperCase()}</div>
          )}
          {job.translate && job.target_language && (
            <div className="flex items-center gap-1">
              <Languages size={12} className="text-primary" />
              <span className="text-foreground font-medium">Translated to:</span> {(supportedLangs[job.target_language]?.name ?? job.target_language)}
              {job.translator_mode && <span className="ml-1 text-xs bg-muted rounded px-1">{job.translator_mode}</span>}
            </div>
          )}
          {job.translate && job.translation_status && (
            <div>
              <span className="text-foreground font-medium">Translation:</span>{" "}
              <span className={cn(
                job.translation_status === "done" && "text-green-600 dark:text-green-400",
                job.translation_status === "failed" && "text-destructive",
              )}>{job.translation_status}</span>
            </div>
          )}
          <div><span className="text-foreground font-medium">Created:</span> {fmtDate(job.created_at)}</div>
          <div><span className="text-foreground font-medium">Updated:</span> {fmtDate(job.updated_at)}</div>
        </div>

        {/* Subtitle downloads */}
        {job.status === "done" && jobId && (
          <div className="flex flex-wrap gap-2 pt-1">
            {/* Source language downloads */}
            {(["srt", "vtt"] as const).map((fmt) => (
              <a
                key={`src-${fmt}`}
                href={`/api/subtitles/${jobId}/${fmt}${job.detected_language ? `?lang=${job.detected_language}` : ""}`}
                download
                className="flex items-center gap-1 text-xs border border-input rounded px-2 py-1 hover:bg-muted transition-colors"
              >
                <Download size={12} /> {fmt.toUpperCase()}{job.detected_language ? ` (${job.detected_language.toUpperCase()})` : ""}
              </a>
            ))}
            {/* Translated language downloads */}
            {job.translate && job.target_language && job.translation_status === "done" && (
              (["srt", "vtt"] as const).map((fmt) => (
                <a
                  key={`tl-${fmt}`}
                  href={`/api/subtitles/${jobId}/${fmt}?lang=${job.target_language}`}
                  download
                  className="flex items-center gap-1 text-xs border border-primary/40 rounded px-2 py-1 hover:bg-primary/5 transition-colors text-primary"
                >
                  <Download size={12} /> {fmt.toUpperCase()} ({(supportedLangs[job.target_language!]?.name ?? job.target_language!).slice(0, 6)})
                </a>
              ))
            )}
          </div>
        )}

        {job.status === "failed" && job.error_message && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-sm text-destructive">
            {job.error_message}
          </div>
        )}
        {job.status === "waiting_for_models" && job.waiting_for_model && (
          <div className="rounded-md bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 px-3 py-2 text-sm text-yellow-700 dark:text-yellow-400">
            Waiting for model to finish downloading: <strong>{job.waiting_for_model}</strong>. Job will resume automatically.
          </div>
        )}
      </div>

      {/* Media + subtitle preview */}
      {job.status === "done" && jobId && (
        <MediaPreview jobId={jobId} detectedLanguage={job.detected_language} />
      )}

      {/* Retro-translate panel */}
      {job.status === "done" && jobId && (
        <div className="bg-card border rounded-xl p-4 space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Languages size={16} className="text-primary" />
            Add / re-run translation
          </h3>
          <div className="flex items-center gap-3 flex-wrap">
            <select
              value={translateTarget}
              onChange={(e) => setTranslateTarget(e.target.value)}
              className="rounded-md border border-input bg-background px-3 py-1.5 text-sm"
              disabled={translating}
            >
              <option value="">Select target language…</option>
              {Object.entries(supportedLangs).map(([code, info]) => (
                <option key={code} value={code}>{info.name} ({code})</option>
              ))}
            </select>
            <button
              onClick={handleTranslate}
              disabled={!translateTarget || translating}
              className="flex items-center gap-1 bg-primary text-primary-foreground hover:bg-primary/90 px-4 py-1.5 rounded-md text-sm font-medium transition-colors disabled:opacity-50"
            >
              {translating ? <><Loader size={14} className="animate-spin" /> Translating…</> : "Translate"}
            </button>
          </div>
        </div>
      )}

      {/* Chunks detail */}
      {job.chunks && job.chunks.length > 0 && (
        <div className="bg-card border rounded-xl p-4">
          <h3 className="text-sm font-medium mb-3">Chunks ({job.chunks.length})</h3>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {job.chunks.map((c) => (
              <div key={c.id} className="flex items-center gap-2 text-xs py-0.5">
                <span className={cn(
                  "w-2 h-2 rounded-full shrink-0",
                  c.status === "done" && "bg-green-500",
                  c.status === "failed" && "bg-destructive",
                  c.status === "processing" && "bg-primary animate-pulse",
                  c.status === "skipped" && "bg-muted",
                  c.status === "pending" && "bg-muted-foreground/40",
                )} />
                <span className="text-muted-foreground font-mono">#{c.sequence + 1}</span>
                <span>{c.start_time.toFixed(1)}s – {c.end_time.toFixed(1)}s</span>
                {c.detected_language && <span className="uppercase text-muted-foreground">{c.detected_language}</span>}
                {c.assigned_model && <span className="text-muted-foreground">{c.assigned_model}</span>}
                <span className={cn(
                  "ml-auto",
                  c.status === "done" && "text-green-600 dark:text-green-400",
                  c.status === "failed" && "text-destructive",
                  c.status === "skipped" && "text-muted-foreground",
                )}>{c.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
