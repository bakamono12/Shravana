"""Build a ContextBundle once per job after transcription finishes.

Strategy:
  - For video: sample keyframes every CONTEXT_KEYFRAME_INTERVAL_S seconds.
    Ask Qwen2.5-VL for a one-line scene description per frame, then combine.
  - For all inputs: send the joined transcript to Qwen2.5-VL (text-only mode)
    with a strict-JSON system prompt to extract domain, format, named_entities, idioms.
  - Every external call is wrapped with a fallback; failures produce an empty
    ContextBundle so the pipeline is never blocked.
"""
from __future__ import annotations
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from app.ml.base import ContextBundle
from app.config import settings

logger = logging.getLogger(__name__)

_DOMAIN_CHOICES = (
    "tech finance spirituality sports news casual academic legal medical entertainment"
).split()
_FORMAT_CHOICES = (
    "interview monologue panel lecture debate storytelling podcast"
).split()


def _empty_bundle(source_lang: str, target_lang: str) -> ContextBundle:
    return ContextBundle(
        domain="casual",
        format="monologue",
        source_language=source_lang,
        target_language=target_lang,
    )


def _sample_keyframes(video_path: str, job_dir: str, interval_s: int) -> list[Path]:
    """Extract one frame every interval_s seconds to job_dir/frames/."""
    frames_dir = Path(job_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval_s}",
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except Exception as exc:
        logger.warning(f"Context keyframe extraction failed: {exc}")
        return []

    return sorted(frames_dir.glob("frame_*.jpg"))


def _describe_frames_with_vlm(frames: list[Path]) -> str:
    """Send all keyframes in one multi-image VLM call; return joined descriptions."""
    if not frames:
        return ""
    try:
        from app.ml.registry import get as registry_get
        vlm = registry_get("qwen2_5_vl")
    except Exception as exc:
        logger.warning(f"VLM not available for frame description: {exc}")
        return ""

    capped = frames[:8]

    if len(capped) > 1:
        # Single multi-image call — reduces N serial round-trips to 1.
        numbered_prompt = (
            f"I will show you {len(capped)} video frames in order. "
            "For each frame write ONE sentence: setting, activity, subject matter. "
            f"Number your answers 1 through {len(capped)}. No other text."
        )
        try:
            raw = vlm._call_model(
                source_text=numbered_prompt,
                system_prompt="You are a concise scene analyst.",
                history_text="",
                frames=capped,
                max_new_tokens=64 * len(capped),
            )
            lines = [
                re.sub(r"^\d+[.)]\s*", "", line).strip()
                for line in raw.splitlines()
                if re.match(r"^\d+[.)]", line.strip())
            ]
            if lines:
                return " | ".join(lines)
            # Model didn't number answers — return the raw text as-is.
            return raw.strip()
        except Exception as exc:
            logger.warning(f"Batched frame description failed ({exc}), falling back to serial")

    # Serial fallback: one frame at a time (also handles the single-frame case).
    descriptions: list[str] = []
    for frame in capped:
        try:
            desc = vlm._call_model(
                source_text="Describe this scene in one sentence: setting, activity, and apparent subject matter.",
                system_prompt="You are a concise scene analyst. Answer in one sentence only.",
                history_text="",
                frames=[frame],
                max_new_tokens=64,
            )
            descriptions.append(desc.strip())
        except Exception as exc:
            logger.warning(f"Frame description failed for {frame.name}: {exc}")

    return " | ".join(descriptions)


def _extract_context_from_transcript(
    transcript: str,
    source_lang: str,
    target_lang: str,
    scene_description: str,
) -> dict[str, Any]:
    """Ask Qwen2.5-VL (text-only) to return structured context JSON."""
    try:
        from app.ml.registry import get as registry_get
        vlm = registry_get("qwen2_5_vl")
    except Exception as exc:
        logger.warning(f"VLM not available for context extraction: {exc}")
        return {}

    scene_note = f"\nScene: {scene_description}" if scene_description else ""
    system_prompt = (
        "You are a media analyst. Analyze the transcript and return ONLY a JSON object with these keys:\n"
        '{"domain": one of [' + ", ".join(_DOMAIN_CHOICES) + "],\n"
        ' "format": one of [' + ", ".join(_FORMAT_CHOICES) + "],\n"
        ' "named_entities": [list of proper nouns, brand names, place names that must NOT be translated],\n'
        ' "idioms_detected": [list of idiomatic or culture-specific phrases found],\n'
        ' "notes": "any other context useful for a translator"}\n'
        "Return only the JSON. No explanation."
    )
    user_text = f"Source language: {source_lang}. Target language: {target_lang}.{scene_note}\n\nTranscript:\n{transcript[:4000]}"

    try:
        raw = vlm._call_model(
            source_text=user_text,
            system_prompt=system_prompt,
            history_text="",
            frames=None,
            max_new_tokens=300,
        )
        # Extract JSON even if the model wraps it in markdown fences
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as exc:
        logger.warning(f"Context JSON extraction failed: {exc}")

    return {}


def build_context(
    source_lang: str,
    target_lang: str,
    all_segments: list[dict],
    video_path: str | None,
    job_dir: str,
    is_video: bool,
) -> ContextBundle:
    """Build a ContextBundle for a job.

    Degradation: on any failure returns an empty bundle so the pipeline continues.
    """
    transcript = " ".join(
        s.get("text", "").strip() for s in all_segments if s.get("text", "").strip()
    )

    scene_description: str = ""
    if is_video and video_path:
        try:
            frames = _sample_keyframes(video_path, job_dir, settings.CONTEXT_KEYFRAME_INTERVAL_S)
            scene_description = _describe_frames_with_vlm(frames)
        except Exception as exc:
            logger.warning(f"Scene description skipped: {exc}")

    if not transcript:
        logger.warning("Empty transcript, returning empty ContextBundle")
        return _empty_bundle(source_lang, target_lang)

    ctx_data = _extract_context_from_transcript(
        transcript=transcript,
        source_lang=source_lang,
        target_lang=target_lang,
        scene_description=scene_description,
    )

    return ContextBundle(
        domain=ctx_data.get("domain", "casual") if ctx_data.get("domain") in _DOMAIN_CHOICES else "casual",
        format=ctx_data.get("format", "monologue") if ctx_data.get("format") in _FORMAT_CHOICES else "monologue",
        source_language=source_lang,
        target_language=target_lang,
        named_entities=ctx_data.get("named_entities", []) if isinstance(ctx_data.get("named_entities"), list) else [],
        idioms_detected=ctx_data.get("idioms_detected", []) if isinstance(ctx_data.get("idioms_detected"), list) else [],
        scene_description=scene_description or None,
        notes=ctx_data.get("notes"),
    )
