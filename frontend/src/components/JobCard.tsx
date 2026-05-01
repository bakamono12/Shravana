import { useState } from "react";
import { type Job, api } from "../lib/api";
import { cn, fmtDate, fmtEta } from "../lib/utils";
import { CheckCircle, XCircle, Clock, Loader, AlertCircle, Download, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const STATUS_LABELS: Record<string, string> = {
  pending: "Queued",
  extracting: "Extracting audio",
  denoising: "Denoising",
  chunking: "Chunking",
  processing: "Transcribing",
  assembling: "Assembling",
  done: "Done",
  failed: "Failed",
  waiting_for_models: "Waiting for models",
};

function StatusIcon({ status }: { status: string }) {
  if (status === "done") return <CheckCircle size={16} className="text-green-500" />;
  if (status === "failed") return <XCircle size={16} className="text-destructive" />;
  if (status === "pending") return <Clock size={16} className="text-muted-foreground" />;
  if (status === "waiting_for_models") return <AlertCircle size={16} className="text-yellow-500" />;
  return <Loader size={16} className="text-primary animate-spin" />;
}

interface Props {
  job: Job;
  onDelete?: (id: string) => void;
}

const ACTIVE_STATUSES = new Set(["extracting", "denoising", "chunking", "processing", "assembling"]);

export function JobCard({ job, onDelete }: Props) {
  const navigate = useNavigate();
  const [deleting, setDeleting] = useState(false);
  const pct = job.total_chunks > 0 ? Math.round((job.completed_chunks / job.total_chunks) * 100) : 0;
  const isActive = !["done", "failed", "pending", "waiting_for_models"].includes(job.status);

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Delete job "${job.filename}"?`)) return;
    setDeleting(true);
    try {
      await api.deleteJob(job.id);
      onDelete?.(job.id);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => navigate(`/jobs/${job.id}`)}
      onKeyDown={(e) => e.key === "Enter" && navigate(`/jobs/${job.id}`)}
      className={cn(
        "block rounded-xl border bg-card p-4 hover:border-primary/40 transition-colors cursor-pointer"
      )}
    >
      <div className="flex items-start gap-3">
        <StatusIcon status={job.status} />
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{job.filename}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{fmtDate(job.created_at)}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-1.5">
            <span className={cn(
              "text-xs px-2 py-0.5 rounded-full font-medium",
              job.status === "done" && "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400",
              job.status === "failed" && "bg-destructive/10 text-destructive",
              job.status === "waiting_for_models" && "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
              !["done","failed","waiting_for_models"].includes(job.status) && "bg-primary/10 text-primary",
            )}>
              {STATUS_LABELS[job.status] ?? job.status}
            </span>
            {!ACTIVE_STATUSES.has(job.status) && (
              <button
                onClick={handleDelete}
                disabled={deleting}
                title="Delete job"
                className="text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50"
              >
                <Trash2 size={14} />
              </button>
            )}
          </div>
          {job.detected_language && (
            <span className="text-xs text-muted-foreground uppercase">{job.detected_language}</span>
          )}
        </div>
      </div>

      {isActive && job.total_chunks > 0 && (
        <div className="mt-3">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>{job.completed_chunks}/{job.total_chunks} chunks</span>
            <span>{pct}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {job.status === "done" && (
        <div className="mt-3 flex gap-2">
          {["srt", "vtt"].map((fmt) => (
            <a
              key={fmt}
              href={`/api/subtitles/${job.id}/${fmt}`}
              download
              onClick={(e) => e.stopPropagation()}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-secondary text-secondary-foreground hover:bg-muted transition-colors"
            >
              <Download size={12} /> {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      )}

      {job.status === "failed" && job.error_message && (
        <p className="mt-2 text-xs text-destructive truncate">{job.error_message}</p>
      )}
      {job.status === "waiting_for_models" && job.waiting_for_model && (
        <p className="mt-2 text-xs text-yellow-600 dark:text-yellow-400">
          Waiting for model: {job.waiting_for_model}
        </p>
      )}
    </div>
  );
}
