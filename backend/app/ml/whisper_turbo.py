"""Whisper Large-v3-Turbo via faster-whisper — Marathi + fallback."""
from typing import List
from app.ml.base import BaseSTTModel, TranscriptSegment, WordTimestamp


class WhisperTurboModel(BaseSTTModel):
    def __init__(self, model_size: str = "large-v3-turbo"):
        self.model_size = model_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        # CPU with INT8 quantization for speed
        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str, language: str | None = None) -> List[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError("WhisperTurboModel not loaded")
        segments_iter, _ = self._model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        segments = []
        for seg in segments_iter:
            words = [
                WordTimestamp(word=w.word, start=w.start, end=w.end, confidence=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                language=language or "mr",
            ))
        return segments
