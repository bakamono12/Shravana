"""Merge per-chunk transcripts into one timeline-correct transcript."""
from __future__ import annotations
import json
from typing import List, Tuple
from app.ml.base import TranscriptSegment, WordTimestamp
from app.config import settings


def offset_segment(seg: TranscriptSegment, chunk_start: float) -> TranscriptSegment:
    """Shift all timestamps in a segment by chunk_start seconds."""
    words = [
        WordTimestamp(word=w.word, start=w.start + chunk_start, end=w.end + chunk_start, confidence=w.confidence)
        for w in seg.words
    ]
    return TranscriptSegment(
        text=seg.text,
        start=seg.start + chunk_start,
        end=seg.end + chunk_start,
        words=words,
        language=seg.language,
    )


def deduplicate_overlap(
    segs_a: List[TranscriptSegment],
    segs_b: List[TranscriptSegment],
    overlap: float,
) -> Tuple[List[TranscriptSegment], List[TranscriptSegment]]:
    """
    At the seam between chunk N and N+1, trim overlap/2 from end of A and start of B.
    """
    half = overlap / 2.0

    if segs_a:
        cutoff_a = (segs_a[-1].end if segs_a else 0.0) - half
        segs_a = [s for s in segs_a if s.start < cutoff_a]

    if segs_b:
        cutoff_b = (segs_b[0].start if segs_b else 0.0) + half
        segs_b = [s for s in segs_b if s.end > cutoff_b]

    return segs_a, segs_b


def reassemble(
    chunk_data: List[dict],  # [{"sequence": N, "start_time": F, "transcript_json": "..."}]
    overlap: float | None = None,
) -> List[TranscriptSegment]:
    """
    chunk_data: list of chunk DB rows sorted by sequence.
    Returns merged, time-offset transcript segments.
    """
    overlap = overlap if overlap is not None else float(settings.CHUNK_OVERLAP_SECONDS)
    sorted_chunks = sorted(chunk_data, key=lambda c: c["sequence"])

    all_segments: List[TranscriptSegment] = []

    for i, chunk in enumerate(sorted_chunks):
        raw = chunk.get("transcript_json")
        if not raw:
            continue
        segs = _deserialize(raw, chunk["start_time"])
        if not segs:
            continue

        if all_segments and i > 0:
            all_segments, segs = deduplicate_overlap(all_segments, segs, overlap)

        all_segments.extend(segs)

    return all_segments


def _deserialize(transcript_json: str, chunk_start: float) -> List[TranscriptSegment]:
    data = json.loads(transcript_json)
    segments = []
    for item in data:
        words = [
            WordTimestamp(
                word=w["word"],
                start=w["start"] + chunk_start,
                end=w["end"] + chunk_start,
                confidence=w.get("confidence", 1.0),
            )
            for w in item.get("words", [])
        ]
        segments.append(TranscriptSegment(
            text=item["text"],
            start=item["start"] + chunk_start,
            end=item["end"] + chunk_start,
            words=words,
            language=item.get("language", "hi"),
        ))
    return segments


def serialize_segments(segments: List[TranscriptSegment]) -> str:
    return json.dumps([
        {
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "language": s.language,
            "words": [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence} for w in s.words],
        }
        for s in segments
    ])
