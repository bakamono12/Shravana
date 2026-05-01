"""Split clean.wav into 60s chunks with 5s overlap."""
from __future__ import annotations
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import settings


@dataclass
class ChunkManifestEntry:
    sequence: int
    filename: str
    path: str
    start_time: float
    end_time: float
    duration: float


def chunk_audio(
    clean_wav: str,
    chunks_dir: str,
    chunk_duration: int | None = None,
    overlap: int | None = None,
) -> List[ChunkManifestEntry]:
    chunk_duration = chunk_duration or settings.CHUNK_DURATION_SECONDS
    overlap = overlap or settings.CHUNK_OVERLAP_SECONDS

    out = Path(chunks_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = _get_duration(clean_wav)

    # Split using FFmpeg segmentation
    pattern = str(out / "chunk_%04d.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", clean_wav,
            "-f", "segment",
            "-segment_time", str(chunk_duration),
            "-segment_time_delta", "0.5",
            "-reset_timestamps", "1",
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            "-ac", "1",
            pattern,
        ],
        check=True,
        capture_output=True,
    )

    chunk_files = sorted(out.glob("chunk_*.wav"), key=lambda p: p.name)
    manifest: List[ChunkManifestEntry] = []

    for i, chunk_path in enumerate(chunk_files):
        dur = _get_duration(str(chunk_path))
        if dur < 0.5:
            chunk_path.unlink(missing_ok=True)
            continue

        start = i * chunk_duration
        if i > 0:
            # Overlap: earlier chunks extend by 'overlap' seconds into next
            # We record the real start_time from the segment pattern (approximately)
            start = max(0.0, i * chunk_duration - overlap)

        end = min(total_duration, start + dur)
        manifest.append(ChunkManifestEntry(
            sequence=i,
            filename=chunk_path.name,
            path=str(chunk_path),
            start_time=round(start, 3),
            end_time=round(end, 3),
            duration=round(dur, 3),
        ))

    return manifest


def _get_duration(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])
