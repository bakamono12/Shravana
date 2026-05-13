"""Qwen2.5-VL translator — vision-language first-pass and LLM refiner.

Roles:
  1. First-pass translator for *video* jobs: builds a multimodal prompt with
     sampled keyframes + source transcript + context → translated text.
  2. LLM refiner for *both* modes when enable_refinement=True: text-only mode,
     takes the first-pass draft + prev/next chunks + ContextBundle + glossary
     and returns a context-aware improved translation.

Using one model for both roles avoids pulling in Ollama / a separate LLM.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.ml.base import (
    BaseTranslator, TranscriptSegment, TranslatedSegment, WordTimestamp,
    ContextBundle, get_device,
)

logger = logging.getLogger(__name__)


def _distribute_words(text: str, start: float, end: float) -> List[WordTimestamp]:
    words = text.split()
    if not words:
        return []
    n = len(words)
    step = (end - start) / n
    return [
        WordTimestamp(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]


def _build_system_prompt(
    source_lang: str,
    target_lang: str,
    context: Optional[ContextBundle],
    glossary_text: str,
) -> str:
    domain = context.domain if context else "general"
    fmt = context.format if context else "speech"
    entities = ", ".join(context.named_entities) if (context and context.named_entities) else "none"
    idioms = ", ".join(context.idioms_detected) if (context and context.idioms_detected) else "none"
    scene = context.scene_description if (context and context.scene_description) else "not available"

    parts = [
        f"You are a professional subtitle translator. "
        f"Translate from {source_lang} to {target_lang}.",
        f"Content domain: {domain}. Format: {fmt}.",
        f"Scene context: {scene}.",
        f"Named entities to keep as-is: {entities}.",
        f"Idioms to translate culturally (not literally): {idioms}.",
        "Preserve the speaker's register (casual stays casual, formal stays formal).",
        "Keep subtitles concise — max 2 lines.",
    ]
    if glossary_text:
        parts.append(glossary_text)
    parts.append("Output ONLY the translated text. No explanation, no quotes.")
    return " ".join(parts)


class QwenVLTranslator(BaseTranslator):
    name = "qwen2_5_vl"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        logger.info(f"Loading Qwen2.5-VL from {self.model_id}")

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            max_pixels=settings.VLM_MAX_PIXELS,
        )

        quantization_config = None
        if settings.VLM_USE_4BIT and get_device() == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit NF4 quantization (bitsandbytes)")
            except ImportError:
                logger.warning(
                    "VLM_USE_4BIT=True but bitsandbytes is not installed; "
                    "falling back to full precision. Install with: pip install bitsandbytes>=0.43.0"
                )

        load_kwargs: dict = {
            "torch_dtype": torch.float16 if get_device() == "cuda" else "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        self._model.eval()
        logger.info("Qwen2.5-VL loaded")

    def translate(
        self,
        segments: List[TranscriptSegment],
        source_lang: str,
        target_lang: str,
        history: List[TranslatedSegment],
        audio_path: Optional[str] = None,
        frames: Optional[List[Path]] = None,
        context: Optional[ContextBundle] = None,
        glossary_text: str = "",
    ) -> List[TranslatedSegment]:
        """First-pass translation for video jobs (uses frames when available)."""
        if not segments:
            return []

        system_prompt = _build_system_prompt(source_lang, target_lang, context, glossary_text)
        history_text = " ".join(s.text for s in history[-4:]) if history else ""
        results: List[TranslatedSegment] = []

        for seg in segments:
            try:
                translated_text = self._call_model(
                    source_text=seg.text,
                    system_prompt=system_prompt,
                    history_text=history_text,
                    frames=frames,
                    max_new_tokens=256,
                )
            except Exception as exc:
                logger.warning(f"Qwen2.5-VL translate segment failed ({exc}), using source text")
                translated_text = seg.text

            results.append(TranslatedSegment(
                text=translated_text,
                start=seg.start,
                end=seg.end,
                words=_distribute_words(translated_text, seg.start, seg.end),
                source_text=seg.text,
                source_language=source_lang,
                target_language=target_lang,
            ))

        return results

    def refine(
        self,
        first_pass: str,
        source_text: str,
        prev_chunk: str,
        next_chunk: str,
        source_lang: str,
        target_lang: str,
        start: float,
        end: float,
        context: Optional[ContextBundle] = None,
        glossary_text: str = "",
    ) -> str:
        """Context-aware LLM refinement pass (text-only; called for both audio and video modes)."""
        system_prompt = _build_system_prompt(source_lang, target_lang, context, glossary_text)

        user_parts = []
        if prev_chunk:
            user_parts.append(f"[Previous context, do not translate]: {prev_chunk}")
        user_parts.append(f"[Source text]: {source_text}")
        user_parts.append(f"[Machine translation draft]: {first_pass}")
        if next_chunk:
            user_parts.append(f"[Next context, do not translate]: {next_chunk}")
        user_parts.append("Provide the improved translation only.")

        user_text = "\n".join(user_parts)

        try:
            return self._call_model(
                source_text=user_text,
                system_prompt=system_prompt,
                history_text="",
                frames=None,
                max_new_tokens=256,
            )
        except Exception as exc:
            logger.warning(f"Qwen2.5-VL refine failed ({exc}), keeping first pass")
            return first_pass

    def _call_model(
        self,
        source_text: str,
        system_prompt: str,
        history_text: str,
        frames: Optional[List[Path]],
        max_new_tokens: int = 256,
    ) -> str:
        from qwen_vl_utils import process_vision_info
        import torch

        messages = [{"role": "system", "content": system_prompt}]

        user_content = []
        if frames:
            for frame_path in frames:
                user_content.append({"type": "image", "image": str(frame_path)})

        if history_text:
            user_content.append({"type": "text", "text": f"[Recent dialogue]: {history_text}"})

        user_content.append({"type": "text", "text": source_text})
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()
