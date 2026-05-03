"""Regroup STT segments into topic-coherent translation units.

Distinct from the audio chunker (which slices audio for STT).
This chunker works on already-transcribed text segments and groups them
so the translation model receives meaningful blocks rather than one line at a time.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationUnit:
    """One semantic chunk that will be fed to the translator as a unit."""
    sequence: int
    start_time: float
    end_time: float
    source_text: str
    speaker_id: str | None
    # Indices into the original segment list so we can map translations back
    segment_indices: list[int]


def chunk_for_translation(
    segments: list[dict[str, Any]],
    max_seconds: float = 20.0,
    silence_gap: float = 1.5,
) -> list[TranslationUnit]:
    """Group segments into translation units.

    Boundaries triggered by:
    - accumulated duration exceeds max_seconds
    - gap between segment end and next segment start exceeds silence_gap
    - speaker change (if speaker_id is present in segment dicts)

    Each unit carries the indices of its constituent segments so
    distribute_translation_to_segments can map translated text back.
    """
    if not segments:
        return []

    units: list[TranslationUnit] = []
    current: list[int] = []          # indices into `segments`
    current_start: float = 0.0
    current_end: float = 0.0
    current_speaker: str | None = None

    def _flush(seq: int) -> TranslationUnit:
        segs = [segments[i] for i in current]
        text = " ".join(s["text"].strip() for s in segs if s.get("text", "").strip())
        return TranslationUnit(
            sequence=seq,
            start_time=current_start,
            end_time=current_end,
            source_text=text,
            speaker_id=current_speaker,
            segment_indices=list(current),
        )

    for idx, seg in enumerate(segments):
        start: float = seg.get("start", 0.0)
        end: float = seg.get("end", start)
        speaker: str | None = seg.get("speaker_id")

        if not current:
            current_start = start
            current_end = end
            current_speaker = speaker
            current.append(idx)
            continue

        duration_so_far = end - current_start
        gap = start - current_end
        speaker_changed = speaker is not None and current_speaker is not None and speaker != current_speaker

        if duration_so_far > max_seconds or gap > silence_gap or speaker_changed:
            units.append(_flush(len(units)))
            current = [idx]
            current_start = start
            current_end = end
            current_speaker = speaker
        else:
            current.append(idx)
            current_end = end
            # keep first speaker of the chunk
            if current_speaker is None and speaker is not None:
                current_speaker = speaker

    if current:
        units.append(_flush(len(units)))

    return units
