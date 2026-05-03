"""Map translated semantic chunks back to original STT segment timing.

Each TranslationUnit covers several original STT segments. We need to
distribute the translated text across those segments, preserving the
original [start, end] timestamps so subtitle_generator works unchanged.

Strategy:
  - Split the translated text into sentences (by punctuation).
  - Distribute sentences across original segments proportionally by their
    source-text character count.
  - Segments with no source text are left empty.
  - If split produces fewer pieces than segments, the last piece fills the rest.

The result is a List[TranscriptSegment] ready for subtitle_generator.
"""
from __future__ import annotations
import re
from typing import Any

from app.ml.base import TranscriptSegment, TranslatedSegment, WordTimestamp
from app.pipeline.semantic_chunker import TranslationUnit


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common boundary punctuation."""
    parts = re.split(r'(?<=[.!?।॥])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _distribute_words(text: str, start: float, end: float) -> list[WordTimestamp]:
    words = text.split()
    if not words:
        return []
    n = len(words)
    step = (end - start) / n if n else 0
    return [
        WordTimestamp(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]


def distribute_translation_to_segments(
    original_segments: list[TranscriptSegment],
    translated_units: list[TranslationUnit],
    translated_texts: list[str],        # parallel to translated_units; final translated_text
    target_language: str,
) -> list[TranscriptSegment]:
    """Return a new segment list where text/words are translated but timing is original.

    Args:
        original_segments: full merged STT segments (after reassemble).
        translated_units: semantic chunks (each knows which original segment indices it covers).
        translated_texts: final translated text per unit (parallel list).
        target_language: BCP-47 code for the output language.
    """
    if not translated_units:
        return original_segments

    # Start from the original segments as mutable dicts; we'll rebuild TranscriptSegment
    result: list[TranscriptSegment | None] = [None] * len(original_segments)

    for unit, tl_text in zip(translated_units, translated_texts):
        if not unit.segment_indices:
            continue

        indices = unit.segment_indices
        source_segs = [original_segments[i] for i in indices if i < len(original_segments)]
        if not source_segs:
            continue

        # Split translated text into sentences, then map onto source segments
        sentences = _split_sentences(tl_text) if tl_text else []
        n_segs = len(source_segs)

        if not sentences:
            # Fallback: use source text unchanged
            for i, seg in zip(indices, source_segs):
                if i < len(result):
                    result[i] = seg
            continue

        # Distribute sentences across segments by source character count proportion
        src_char_counts = [max(len(s.text), 1) for s in source_segs]
        total_chars = sum(src_char_counts)
        total_tl_chars = max(len(tl_text), 1)

        # Assign each segment a proportional slice of the translated text
        tl_chars = list(tl_text)
        seg_texts: list[str] = []
        cursor = 0
        for k, src_count in enumerate(src_char_counts):
            share = src_count / total_chars
            n_chars = round(share * total_tl_chars)
            if k == n_segs - 1:
                piece = tl_text[cursor:].strip()
            else:
                end_pos = cursor + n_chars
                # Snap to nearest sentence boundary
                best = end_pos
                for sent in sentences:
                    pos = tl_text.find(sent, cursor)
                    if pos != -1 and pos >= cursor:
                        candidate_end = pos + len(sent)
                        if abs(candidate_end - end_pos) < abs(best - end_pos):
                            best = candidate_end
                piece = tl_text[cursor:best].strip()
                cursor = best
            seg_texts.append(piece)

        for idx, seg, tl_seg_text in zip(indices, source_segs, seg_texts):
            if idx >= len(result):
                continue
            result[idx] = TranscriptSegment(
                text=tl_seg_text,
                start=seg.start,
                end=seg.end,
                words=_distribute_words(tl_seg_text, seg.start, seg.end),
                language=target_language,
            )

    # Fill any segments that weren't covered by a translation unit (edge case)
    for i, seg in enumerate(original_segments):
        if result[i] is None:
            result[i] = seg

    return [s for s in result if s is not None]
