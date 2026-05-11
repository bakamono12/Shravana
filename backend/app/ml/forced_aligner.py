"""Qwen3-ForcedAligner — upgrades Whisper segment timestamps to word-level."""
from typing import List
from app.ml.base import TranscriptSegment, WordTimestamp


class ForcedAlignerModel:
    def __init__(self, model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModel
        from app.ml.base import get_device
        self._device = get_device()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device)
        self._model.eval()

    def align(self, audio_path: str, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Enrich segments (which may have no word-level timestamps) with word timestamps."""
        if self._model is None:
            raise RuntimeError("ForcedAlignerModel not loaded")
        import torch
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)

        enriched = []
        for seg in segments:
            try:
                inputs = self._processor(
                    audio=audio,
                    text=seg.text,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self._model(**inputs)
                word_timestamps = self._processor.post_process_word_timestamps(out, seg.start, seg.end)
                words = [
                    WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
                    for w in word_timestamps
                ]
                enriched.append(TranscriptSegment(
                    text=seg.text, start=seg.start, end=seg.end, words=words, language=seg.language
                ))
            except Exception:
                # Fall back to segment-level if alignment fails
                enriched.append(seg)
        return enriched
