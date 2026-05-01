import { useRef, useState } from "react";
import { Download, Eye, ChevronDown, ChevronUp } from "lucide-react";
import { SubtitlePreview } from "./SubtitlePreview";
import { cn } from "../lib/utils";

interface Props {
  jobId: string;
  detectedLanguage?: string | null;
}

export function MediaPreview({ jobId, detectedLanguage }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isAudioOnly, setIsAudioOnly] = useState(false);
  const [showBlocks, setShowBlocks] = useState(false);

  const handleMetadata = () => {
    if (videoRef.current && videoRef.current.videoHeight === 0) {
      setIsAudioOnly(true);
    }
  };

  return (
    <div className="bg-card border rounded-xl p-3 sm:p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Eye size={15} className="text-muted-foreground" />
        <span className="text-sm font-medium">Preview</span>
      </div>

      {/* Player */}
      <div className={cn(
        "w-full overflow-hidden rounded-lg bg-black",
        isAudioOnly ? "h-14" : "aspect-video max-h-[70vh]",
      )}>
        <video
          ref={videoRef}
          controls
          preload="metadata"
          onLoadedMetadata={handleMetadata}
          className="w-full h-full"
          style={{ display: "block" }}
        >
          <source src={`/api/jobs/${jobId}/media`} />
          <track
            default
            kind="subtitles"
            label="Subtitles"
            srcLang={detectedLanguage ?? "en"}
            src={`/api/subtitles/${jobId}/vtt`}
          />
        </video>
      </div>

      {/* Download buttons */}
      <div className="flex flex-wrap gap-2">
        {(["srt", "vtt"] as const).map((fmt) => (
          <a
            key={fmt}
            href={`/api/subtitles/${jobId}/${fmt}`}
            download
            className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <Download size={14} /> Download {fmt.toUpperCase()}
          </a>
        ))}
        <a
          href={`/api/subtitles/${jobId}/json`}
          download
          className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-secondary text-secondary-foreground hover:bg-muted transition-colors"
        >
          <Download size={14} /> Raw JSON
        </a>
      </div>

      {/* Collapsible subtitle block list */}
      <div>
        <button
          onClick={() => setShowBlocks((v) => !v)}
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          {showBlocks ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {showBlocks ? "Hide" : "Show"} subtitle blocks
        </button>
        {showBlocks && (
          <div className="mt-3">
            <SubtitlePreview jobId={jobId} hideDownloads />
          </div>
        )}
      </div>
    </div>
  );
}
