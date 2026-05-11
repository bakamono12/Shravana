import { useRef, useState } from "react";
import { Download, Eye, ChevronDown, ChevronUp } from "lucide-react";
import { SubtitlePreview } from "./SubtitlePreview";
import { cn } from "../lib/utils";

interface Props {
  jobId: string;
  detectedLanguage?: string | null;
  targetLanguage?: string | null;
  targetLanguageName?: string;
  translationDone?: boolean;
}

export function MediaPreview({ jobId, detectedLanguage, targetLanguage, targetLanguageName, translationDone }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isAudioOnly, setIsAudioOnly] = useState(false);
  const [showBlocks, setShowBlocks] = useState(false);
  const hasTranslation = !!(translationDone && targetLanguage);
  const [selectedLang, setSelectedLang] = useState<string | null>(null);
  const activeLang = selectedLang ?? detectedLanguage ?? null;

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

      {/* Language toggle */}
      {hasTranslation && (
        <div className="flex gap-1 rounded-lg border border-input p-0.5 w-fit text-sm">
          <button
            onClick={() => setSelectedLang(detectedLanguage ?? null)}
            className={cn(
              "px-3 py-1 rounded-md transition-colors",
              activeLang !== targetLanguage ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {detectedLanguage ? detectedLanguage.toUpperCase() : "Original"}
          </button>
          <button
            onClick={() => setSelectedLang(targetLanguage!)}
            className={cn(
              "px-3 py-1 rounded-md transition-colors",
              activeLang === targetLanguage ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {targetLanguageName ?? targetLanguage!.toUpperCase()}
          </button>
        </div>
      )}

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
            label={activeLang ? activeLang.toUpperCase() : "Subtitles"}
            srcLang={activeLang ?? "en"}
            src={`/api/subtitles/${jobId}/vtt${activeLang ? `?lang=${activeLang}` : ""}`}
          />
        </video>
      </div>

      {/* Download buttons */}
      <div className="flex flex-wrap gap-2">
        {(["srt", "vtt"] as const).map((fmt) => (
          <a
            key={fmt}
            href={`/api/subtitles/${jobId}/${fmt}${activeLang ? `?lang=${activeLang}` : ""}`}
            download
            className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <Download size={14} /> Download {fmt.toUpperCase()}
            {activeLang && <span className="opacity-70 text-xs ml-0.5">({activeLang.toUpperCase()})</span>}
          </a>
        ))}
        <a
          href={`/api/subtitles/${jobId}/json${activeLang ? `?lang=${activeLang}` : ""}`}
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
            <SubtitlePreview jobId={jobId} lang={activeLang ?? undefined} hideDownloads />
          </div>
        )}
      </div>
    </div>
  );
}
