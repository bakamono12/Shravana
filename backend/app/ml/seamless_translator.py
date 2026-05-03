"""SeamlessM4T v2 Large — audio-context speech-to-text translator.

Used as the first-pass translator for audio-only jobs (and as fallback for video jobs
when the VLM is not available).

The model does direct speech-to-text-translation: it reads the audio WAV and outputs
translated text directly, without going through a separate transcription step.
We still use the already-transcribed STT segments for their timestamps; Seamless gives
us the translated text.

Context is provided via a text-priming approach: we prepend the recent dialogue history
as a plain-text string to the model's text context input.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

from app.ml.base import (
    BaseTranslator, TranscriptSegment, TranslatedSegment, WordTimestamp,
    ContextBundle,
)
from app.pipeline.translation_languages import seamless_code

logger = logging.getLogger(__name__)

# Languages natively supported by SeamlessM4T v2 as speech input sources
_SEAMLESS_SPEECH_SOURCES = {
    "en", "hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "ur",
    "fr", "de", "es", "ja", "zh", "ar", "ru", "pt", "ko",
}


def _distribute_words(text: str, start: float, end: float) -> List[WordTimestamp]:
    """Distribute translated words linearly across [start, end]."""
    words = text.split()
    if not words:
        return []
    n = len(words)
    duration = end - start
    step = duration / n
    return [
        WordTimestamp(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]


class SeamlessM4TTranslator(BaseTranslator):
    name = "seamless_v2"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        import torch

        logger.info(f"Loading SeamlessM4T v2 from {self.model_id}")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = SeamlessM4Tv2Model.from_pretrained(self.model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        logger.info("SeamlessM4T v2 loaded")

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
        if not segments:
            return []

        import torchaudio, torch

        tgt_code = seamless_code(target_lang)

        # Try to get the source Seamless code; fall back to text-only mode if speech not supported
        src_code = None
        try:
            src_code = seamless_code(source_lang)
        except ValueError:
            pass

        results: List[TranslatedSegment] = []

        for seg in segments:
            try:
                translated_text = self._translate_segment(
                    seg=seg,
                    audio_path=audio_path,
                    src_code=src_code,
                    tgt_code=tgt_code,
                    history=history,
                    context=context,
                    glossary_text=glossary_text,
                )
            except Exception as exc:
                logger.warning(f"Seamless segment translation failed ({exc}), using source text")
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

    def _translate_segment(
        self,
        seg: TranscriptSegment,
        audio_path: Optional[str],
        src_code: Optional[str],
        tgt_code: str,
        history: List[TranslatedSegment],
        context: Optional[ContextBundle],
        glossary_text: str,
    ) -> str:
        import torchaudio, torch

        # Build history context string
        history_text = ""
        if history:
            history_text = " ".join(s.text for s in history[-4:])

        # Prefer speech-to-text translation when audio is available and source is supported
        if audio_path and src_code:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(0, keepdim=True)  # mono

            # Trim to segment time range if the audio covers the full job
            start_sample = int(seg.start * 16000)
            end_sample = int(seg.end * 16000)
            if end_sample > waveform.shape[1]:
                end_sample = waveform.shape[1]
            if start_sample < end_sample:
                waveform = waveform[:, start_sample:end_sample]

            audio_inputs = self._processor(
                audios=waveform.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output_tokens = self._model.generate(
                    **audio_inputs,
                    tgt_lang=tgt_code,
                    generate_speech=False,
                )
            translated = self._processor.decode(
                output_tokens[0].tolist()[0], skip_special_tokens=True
            )
            return translated.strip()

        # Text-to-text translation fallback (Seamless also supports T2TT)
        source_text = seg.text
        if history_text:
            source_text = f"[Context: {history_text}] {source_text}"

        text_inputs = self._processor(
            text=source_text,
            src_lang=src_code or "eng",
            return_tensors="pt",
        ).to(self._device)

        import torch
        with torch.no_grad():
            output_tokens = self._model.generate(
                **text_inputs,
                tgt_lang=tgt_code,
                generate_speech=False,
            )
        translated = self._processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        )
        return translated.strip()
