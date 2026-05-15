"""Translation orchestrator: mode selection, first-pass + optional refinement,
prev/next sliding window, keyframe extraction.

Called per *semantic* chunk (from semantic_chunker.py), not per STT audio chunk.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.ml.base import ContextBundle, TranscriptSegment, TranslatedSegment, WordTimestamp
from app.pipeline.semantic_chunker import TranslationUnit

if TYPE_CHECKING:
    from app.pipeline.glossary import Glossary

logger = logging.getLogger(__name__)



# ------------------------------------------------------------------ #
# Keyframe extraction (used by context_builder and VLM translator)    #
# ------------------------------------------------------------------ #

def extract_keyframes(video_path: str, start: float, end: float, n: int = 3) -> list[Path]:
    """Extract n evenly-spaced I-frames from [start, end] of video_path.

    Returns file paths. Returns [] on any failure (graceful degradation).
    """
    if not video_path:
        return []
    try:
        out_dir = Path(video_path).parent / "_kf_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        duration = max(end - start, 0.1)
        fps = n / duration
        out_pattern = str(out_dir / f"kf_{start:.1f}_%02d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-vf", f"fps={fps:.4f}",
            "-q:v", "3",
            "-frames:v", str(n),
            out_pattern,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        frames = sorted(out_dir.glob(f"kf_{start:.1f}_*.jpg"))
        return frames[:n]
    except Exception as exc:
        logger.warning(f"Keyframe extraction failed for [{start:.1f}-{end:.1f}]: {exc}")
        return []


# ------------------------------------------------------------------ #
# Distribute words across time (utility)                              #
# ------------------------------------------------------------------ #

def _distribute_words(text: str, start: float, end: float) -> list[WordTimestamp]:
    words = text.split()
    if not words:
        return []
    n = len(words)
    step = (end - start) / n
    return [
        WordTimestamp(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]


# ------------------------------------------------------------------ #
# Core translation logic per semantic chunk                           #
# ------------------------------------------------------------------ #

def translate_unit(
    unit: TranslationUnit,
    video_path: str | None,
    source_lang: str,
    target_lang: str,
    history: list[TranslatedSegment],
    prev_text: str,
    next_text: str,
    context: ContextBundle | None,
    glossary: "Glossary | None",
    enable_refinement: bool,
) -> tuple[str, str]:
    """Translate one semantic unit. Returns (first_pass, final_translated_text).

    Qwen2.5-VL handles all translation: with keyframes when video_path is given,
    text-only when audio-only.
    """
    from app.ml.registry import get as registry_get
    from app.ml.base import ModelNotReady

    glossary_text = glossary.as_prompt_text() if glossary else ""

    seg = TranscriptSegment(
        text=unit.source_text,
        start=unit.start_time,
        end=unit.end_time,
        language=source_lang,
    )

    # ---- First pass ----
    first_pass_text = unit.source_text  # fallback: keep source
    try:
        vlm = registry_get("qwen2_5_vl")
        frames = extract_keyframes(
            video_path, unit.start_time, unit.end_time,
            n=settings.VLM_KEYFRAMES_PER_CHUNK,
        ) if video_path else []
        results = vlm.translate(
            segments=[seg],
            source_lang=source_lang,
            target_lang=target_lang,
            history=history,
            audio_path=None,
            frames=frames or None,
            context=context,
            glossary_text=glossary_text,
        )
        if results:
            first_pass_text = results[0].text
    except ModelNotReady:
        raise
    except Exception as exc:
        logger.warning(f"VLM first-pass failed for unit {unit.sequence}: {exc}")

    # ---- Refinement pass (text-only) ----
    final_text = first_pass_text
    if enable_refinement:
        try:
            vlm = registry_get("qwen2_5_vl")
            refined = vlm.refine(
                first_pass=first_pass_text,
                source_text=unit.source_text,
                prev_chunk=prev_text,
                next_chunk=next_text,
                source_lang=source_lang,
                target_lang=target_lang,
                start=unit.start_time,
                end=unit.end_time,
                context=context,
                glossary_text=glossary_text,
            )
            final_text = refined or first_pass_text
        except ModelNotReady:
            logger.warning("Qwen2.5-VL not available for refinement, keeping first pass")
        except Exception as exc:
            logger.warning(f"Refinement failed for unit {unit.sequence}: {exc}")

    return first_pass_text, final_text
