import { useState, useEffect } from "react";
import { Download, Eye } from "lucide-react";

interface Props {
  jobId: string;
  lang?: string;
  hideDownloads?: boolean;
}

interface Block {
  index: number;
  start: string;
  end: string;
  text: string;
}

function parseSrt(srt: string): Block[] {
  const blocks: Block[] = [];
  const parts = srt.trim().split(/\n\n+/);
  for (const part of parts) {
    const lines = part.trim().split("\n");
    if (lines.length < 3) continue;
    const index = parseInt(lines[0]);
    const [start, end] = lines[1].split(" --> ");
    const text = lines.slice(2).join("\n");
    blocks.push({ index, start, end, text });
  }
  return blocks;
}

export function SubtitlePreview({ jobId, lang, hideDownloads }: Props) {
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const url = `/api/subtitles/${jobId}/srt${lang ? `?lang=${lang}` : ""}`;
    setLoading(true);
    setError(false);
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((text) => { setBlocks(parseSrt(text)); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [jobId, lang]);

  if (loading) return <div className="text-sm text-muted-foreground animate-pulse">Loading preview…</div>;
  if (error) return <div className="text-sm text-muted-foreground">Preview unavailable.</div>;

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Eye size={15} className="text-muted-foreground" />
        <span className="text-sm font-medium">Subtitle preview</span>
        <span className="text-xs text-muted-foreground">({blocks.length} blocks)</span>
      </div>
      <div className="space-y-2 max-h-80 overflow-y-auto rounded-lg border bg-muted/20 p-3">
        {blocks.slice(0, 50).map((b) => (
          <div key={b.index} className="text-sm">
            <span className="text-xs text-muted-foreground font-mono">{b.start} → {b.end}</span>
            <p className="mt-0.5">{b.text}</p>
          </div>
        ))}
        {blocks.length > 50 && (
          <p className="text-xs text-muted-foreground">…and {blocks.length - 50} more blocks</p>
        )}
      </div>
      {!hideDownloads && (
        <div className="flex gap-2 mt-3">
          {["srt", "vtt"].map((fmt) => (
            <a
              key={fmt}
              href={`/api/subtitles/${jobId}/${fmt}${lang ? `?lang=${lang}` : ""}`}
              download
              className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              <Download size={14} /> Download {fmt.toUpperCase()}
            </a>
          ))}
          <a
            href={`/api/subtitles/${jobId}/json${lang ? `?lang=${lang}` : ""}`}
            download
            className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-secondary text-secondary-foreground hover:bg-muted transition-colors"
          >
            <Download size={14} /> Raw JSON
          </a>
        </div>
      )}
    </div>
  );
}
