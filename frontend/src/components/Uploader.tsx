import { useState, useRef, useCallback } from "react";
import { Upload, FileAudio, FileVideo, X } from "lucide-react";
import { cn } from "../lib/utils";
import { api } from "../lib/api";
import { fmtBytes } from "../lib/utils";

const ACCEPTED = ".mp4,.mkv,.mov,.avi,.webm,.mp3,.wav,.m4a,.flac,.ogg,.aac";
const LANGS = [
  { value: "", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi" },
  { value: "mr", label: "Marathi" },
];

interface Props {
  onJobCreated: (jobId: string) => void;
}

export function Uploader({ onJobCreated }: Props) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [langHint, setLangHint] = useState("");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<(() => void) | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress(0);

    // Simulate progress (actual XHR progress would need XMLHttpRequest)
    const fakeInterval = setInterval(() => setProgress((p) => Math.min(p + 5, 90)), 300);

    const { promise, abort } = api.uploadFile(file, langHint || undefined);
    abortRef.current = abort;

    try {
      const res = await promise;
      clearInterval(fakeInterval);
      setProgress(100);
      onJobCreated(res.job_id);
      setFile(null);
      setProgress(0);
    } catch (err: unknown) {
      clearInterval(fakeInterval);
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const cancel = () => {
    abortRef.current?.();
    setUploading(false);
    setProgress(0);
  };

  const isVideo = file && /\.(mp4|mkv|mov|avi|webm)$/i.test(file.name);

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !file && inputRef.current?.click()}
        className={cn(
          "border-2 border-dashed rounded-xl p-10 flex flex-col items-center gap-3 transition-colors cursor-pointer",
          dragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/40",
          file && "cursor-default"
        )}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED} className="hidden" onChange={handleFile} />
        {file ? (
          <>
            {isVideo ? <FileVideo size={40} className="text-primary" /> : <FileAudio size={40} className="text-primary" />}
            <div className="text-center">
              <p className="font-medium text-foreground">{file.name}</p>
              <p className="text-sm text-muted-foreground">{fmtBytes(file.size)}</p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              className="text-muted-foreground hover:text-destructive transition-colors"
            >
              <X size={18} />
            </button>
          </>
        ) : (
          <>
            <Upload size={40} className="text-muted-foreground" />
            <div className="text-center">
              <p className="font-medium">Drop a file here or click to browse</p>
              <p className="text-sm text-muted-foreground mt-1">
                Video: mp4, mkv, mov, avi, webm · Audio: mp3, wav, m4a, flac, ogg, aac
              </p>
            </div>
          </>
        )}
      </div>

      {/* Options + upload */}
      {file && (
        <div className="mt-4 flex flex-col sm:flex-row gap-3 items-start sm:items-center">
          <div className="flex items-center gap-2 flex-1">
            <label className="text-sm text-muted-foreground whitespace-nowrap">Language hint:</label>
            <select
              value={langHint}
              onChange={(e) => setLangHint(e.target.value)}
              className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm"
              disabled={uploading}
            >
              {LANGS.map((l) => (
                <option key={l.value} value={l.value}>{l.label}</option>
              ))}
            </select>
          </div>

          {uploading ? (
            <div className="flex items-center gap-3 w-full sm:w-auto">
              <div className="flex-1 sm:w-32 h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300 rounded-full"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="text-sm text-muted-foreground">{progress}%</span>
              <button onClick={cancel} className="text-sm text-destructive hover:underline">Cancel</button>
            </div>
          ) : (
            <button
              onClick={upload}
              className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-md text-sm font-medium transition-colors"
            >
              Transcribe
            </button>
          )}
        </div>
      )}

      {error && (
        <p className="mt-3 text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">{error}</p>
      )}
    </div>
  );
}
