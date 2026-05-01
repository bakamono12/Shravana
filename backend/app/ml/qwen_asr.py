"""Qwen3-ASR — used for language ID (0.6B) and Hindi/code-switch ASR (1.7B)."""
from typing import List
from app.ml.base import BaseSTTModel, TranscriptSegment, WordTimestamp


class QwenASRModel(BaseSTTModel):
    def __init__(self, model_id: str, lid_only: bool = False):
        self.model_id = model_id
        self.lid_only = lid_only
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        """Returns (iso_language_code, confidence). Uses first 10s only."""
        import torch
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Take first 10 seconds
        audio = audio[: 16000 * 10].astype(np.float32)

        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Parse language token from generated ids
        decoded = self._processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]
        lang = self._parse_language_token(decoded)
        confidence = self._estimate_confidence(generated)
        return lang, confidence

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        import torch
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)

        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated = self._model.generate(**inputs, return_timestamps=True)
        result = self._processor.batch_decode(generated, output_word_offsets=True, skip_special_tokens=True)

        segments = []
        for entry in result:
            text = entry if isinstance(entry, str) else entry.get("text", "")
            word_offsets = entry.get("word_offsets", []) if isinstance(entry, dict) else []
            words = [
                WordTimestamp(word=w["word"], start=w["start_offset"] / 100.0, end=w["end_offset"] / 100.0)
                for w in word_offsets
            ]
            start = words[0].start if words else 0.0
            end = words[-1].end if words else 0.0
            segments.append(TranscriptSegment(text=text, start=start, end=end, words=words, language="hi"))
        return segments

    @staticmethod
    def _parse_language_token(decoded: str) -> str:
        import re
        m = re.search(r"<\|([a-z]{2,3})\|>", decoded)
        return m.group(1) if m else "hi"

    @staticmethod
    def _estimate_confidence(generated) -> float:
        try:
            import torch
            scores = torch.stack(generated.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            top_prob = probs.max(dim=-1).values.mean().item()
            return float(top_prob)
        except Exception:
            return 0.5
