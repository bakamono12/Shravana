"""Convert merged TranscriptSegments to SRT and VTT subtitle files."""
from __future__ import annotations
from typing import List
from app.ml.base import TranscriptSegment


MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MIN_DURATION = 1.0
MAX_DURATION = 7.0


def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _group_words_into_blocks(segments: List[TranscriptSegment]) -> List[dict]:
    """Group words into subtitle blocks respecting max chars and duration."""
    blocks = []
    current_words = []
    current_start = None
    current_chars = 0

    def flush():
        nonlocal current_words, current_start, current_chars
        if not current_words:
            return
        text = " ".join(w.word for w in current_words).strip()
        start = current_words[0].start
        end = current_words[-1].end
        duration = end - start
        duration = max(MIN_DURATION, min(MAX_DURATION, duration))
        end = start + duration
        blocks.append({"text": text, "start": start, "end": end})
        current_words = []
        current_start = None
        current_chars = 0

    for seg in segments:
        words = seg.words if seg.words else [
            type("W", (), {"word": seg.text, "start": seg.start, "end": seg.end})()
        ]
        for word in words:
            word_len = len(word.word) + 1
            line_count = (current_chars + word_len) // MAX_CHARS_PER_LINE + 1
            if (current_chars + word_len > MAX_CHARS_PER_LINE * MAX_LINES) and current_words:
                flush()
            if current_start is None:
                current_start = word.start
            current_words.append(word)
            current_chars += word_len

    flush()
    return blocks


def generate_srt(segments: List[TranscriptSegment], output_path: str) -> str:
    blocks = _group_words_into_blocks(segments)
    lines = []
    for i, block in enumerate(blocks, start=1):
        lines.append(str(i))
        lines.append(f"{_format_time_srt(block['start'])} --> {_format_time_srt(block['end'])}")
        lines.append(block["text"])
        lines.append("")
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def generate_vtt(segments: List[TranscriptSegment], output_path: str) -> str:
    blocks = _group_words_into_blocks(segments)
    lines = ["WEBVTT", ""]
    for block in blocks:
        lines.append(f"{_format_time_vtt(block['start'])} --> {_format_time_vtt(block['end'])}")
        lines.append(block["text"])
        lines.append("")
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
