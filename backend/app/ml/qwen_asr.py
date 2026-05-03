"""Qwen3-ASR — used for language ID (0.6B) and Hindi/code-switch ASR (1.7B).

Uses the official `qwen-asr` Python package (Qwen3ASRModel) as the backend,
which bundles the correct transformers fork that understands the `qwen3_asr`
architecture.  Raw ``AutoModelForSpeechSeq2Seq`` cannot load these checkpoints
with the standard transformers release.
"""
from typing import List
from app.ml.base import BaseSTTModel, TranscriptSegment, WordTimestamp


class QwenASRModel(BaseSTTModel):
    def __init__(self, model_id: str, lid_only: bool = False):
        self.model_id = model_id  # may be a HF repo id or a local path string
        self.lid_only = lid_only
        self._model = None  # Qwen3ASRModel instance

    def load(self) -> None:
        from qwen_asr import Qwen3ASRModel
        import torch
        from pathlib import Path
        from app.config import settings

        # Prefer local directory if the model has already been downloaded so we
        # avoid any network calls.  The downloader places files under
        # storage/models/<registry_key>/; we derive the key from the repo name.
        repo_slug = self.model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        # Map known repo slugs to registry key names used on disk
        _SLUG_TO_KEY = {
            "qwen3_asr_0_6b": "qwen_lid",
            "qwen3_asr_1_7b": "qwen3_asr",
        }
        key = _SLUG_TO_KEY.get(repo_slug, repo_slug)
        local_dir = Path(settings.BASE_DIR / settings.MODELS_DIR) / key
        load_path = str(local_dir) if local_dir.exists() and any(local_dir.rglob("*.safetensors")) else self.model_id

        self._model = Qwen3ASRModel.from_pretrained(
            load_path,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        """Returns (iso_language_code, confidence). Uses first 10s only."""
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Use first 10 seconds for speed
        audio = audio[: 16000 * 10].astype(np.float32)

        results = self._model.transcribe(
            audio=(audio, 16000),
            language=None,  # let the model detect
        )
        if results:
            lang = self._normalize_lang(results[0].language or "")
            # qwen-asr doesn't expose a raw confidence score; use 0.9 as a
            # conservative high-confidence value when the model returns a lang.
            confidence = 0.9 if lang else 0.0
            return lang or "en", confidence
        return "en", 0.0

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)

        results = self._model.transcribe(
            audio=(audio, 16000),
            language=None,
            return_time_stamps=True,
        )

        segments: List[TranscriptSegment] = []
        for res in results:
            text = res.text or ""
            lang = self._normalize_lang(res.language or "hi")
            # time_stamps is a list of dicts/objects when return_time_stamps=True
            words: List[WordTimestamp] = []
            ts_list = res.time_stamps or []
            for ts in ts_list:
                # qwen-asr returns dicts with keys: word/text, start, end
                if isinstance(ts, dict):
                    word = ts.get("word") or ts.get("text", "")
                    start = float(ts.get("start", 0.0))
                    end = float(ts.get("end", 0.0))
                else:
                    word = getattr(ts, "word", "") or getattr(ts, "text", "")
                    start = float(getattr(ts, "start", 0.0))
                    end = float(getattr(ts, "end", 0.0))
                words.append(WordTimestamp(word=word, start=start, end=end))
            seg_start = words[0].start if words else 0.0
            seg_end = words[-1].end if words else 0.0
            segments.append(
                TranscriptSegment(text=text, start=seg_start, end=seg_end, words=words, language=lang)
            )
        return segments

    @staticmethod
    def _normalize_lang(code: str) -> str:
        """Normalise language name/code to ISO 639-1."""
        _MAP = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "arabic": "ar",
        }
        code = code.lower().strip()
        return _MAP.get(code, code)
