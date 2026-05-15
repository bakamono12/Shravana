"""Pure-compute translation pipeline (no DB, no async).

Called by both the local pipeline_runner (via run_in_executor) and the remote
worker's pipeline_handler so the translation logic stays a single source of truth.
"""
from __future__ import annotations
import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def run_translation_compute(
    source_segments,            # List[TranscriptSegment] — already reassembled
    source_lang: str,
    target_lang: str,
    enable_refinement: bool,
    user_glossary_text: str,    # raw user glossary string (may be empty)
    video_path: Optional[str],
    job_dir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List, List[dict], object]:
    """
    Translate source_segments from source_lang to target_lang.

    Returns (translated_segments, unit_results, context_bundle):
    - translated_segments: List[TranslatedSegment] aligned to source segment boundaries
    - unit_results: list of dicts with keys sequence, source_text, start_time,
      end_time, first_pass, final_text  (for DB persistence by the caller)
    - context_bundle: ContextBundle built during this run

    on_progress(unit_done: int, total_units: int) is called after each semantic unit.
    """
    from app.config import settings
    from app.ml.base import ContextBundle, TranslatedSegment
    from app.pipeline.context_builder import build_context
    from app.pipeline.glossary import Glossary
    from app.pipeline.semantic_chunker import chunk_for_translation
    from app.pipeline.translator import translate_unit
    from app.pipeline.translation_reassembler import distribute_translation_to_segments

    is_video = bool(video_path)

    seg_dicts = [
        {"text": s.text, "start": s.start, "end": s.end}
        for s in source_segments
    ]

    # Build context
    try:
        context = build_context(
            source_lang, target_lang, seg_dicts,
            video_path if is_video else None, job_dir, is_video,
        )
    except Exception as exc:
        logger.warning(f"Context build failed ({exc}), using default context")
        context = ContextBundle(
            domain="casual", format="monologue",
            source_language=source_lang, target_language=target_lang,
        )

    # Build glossary (domain defaults merged with user overrides)
    glossary = Glossary.from_domain(context.domain)
    if user_glossary_text:
        try:
            glossary = glossary.merge(Glossary.from_user_text(user_glossary_text))
        except Exception:
            pass

    # Semantic chunking
    translation_units = chunk_for_translation(
        seg_dicts,
        settings.TRANSLATION_MAX_CHUNK_SECONDS,
        settings.TRANSLATION_SILENCE_GAP,
    )
    n_units = len(translation_units)

    # Translate each semantic unit
    translated_texts: List[str] = []
    unit_results: List[dict] = []
    history: list = []

    for i, unit in enumerate(translation_units):
        prev_text = translation_units[i - 1].source_text if i > 0 else ""
        next_text = translation_units[i + 1].source_text if i < n_units - 1 else ""

        first_pass = unit.source_text
        final_text = unit.source_text
        try:
            first_pass, final_text = translate_unit(
                unit,
                video_path if is_video else None,
                source_lang, target_lang,
                history, prev_text, next_text,
                context, glossary, enable_refinement,
            )
        except Exception as exc:
            logger.error(f"Translation unit {unit.sequence} failed: {exc}")

        translated_texts.append(final_text)
        unit_results.append({
            "sequence": unit.sequence,
            "source_text": unit.source_text,
            "start_time": unit.start_time,
            "end_time": unit.end_time,
            "first_pass": first_pass,
            "final_text": final_text,
        })

        dummy = TranslatedSegment(
            text=final_text, start=unit.start_time, end=unit.end_time,
            source_text=unit.source_text, source_language=source_lang,
            target_language=target_lang,
        )
        history = (history + [dummy])[-settings.TRANSLATION_CONTEXT_WINDOW:]

        if on_progress:
            on_progress(i + 1, n_units)

    # Distribute translated text back onto original segment boundaries
    translated_segs = distribute_translation_to_segments(
        source_segments, translation_units, translated_texts, target_lang,
    )

    return translated_segs, unit_results, context
