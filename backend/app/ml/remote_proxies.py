"""Remote proxy implementations for all model types.

Each proxy forwards calls to the remote GPU worker via remote_client, implementing
the same duck-typed interface as the corresponding local model class so that
pipeline_runner.py, transcriber.py, translator.py, and denoiser.py remain unchanged.
"""
from __future__ import annotations
import dataclasses
import io
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.ml.base import (
    BaseSTTModel, BaseTranslator,
    TranscriptSegment, TranslatedSegment, WordTimestamp, ContextBundle,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Serialisation helpers                                               #
# ------------------------------------------------------------------ #

def _seg_to_dict(s: TranscriptSegment) -> dict:
    return dataclasses.asdict(s)


def _dict_to_seg(d: dict) -> TranscriptSegment:
    words = [WordTimestamp(**w) for w in d.get("words", [])]
    return TranscriptSegment(
        text=d["text"], start=d["start"], end=d["end"],
        words=words, language=d.get("language", "en"),
    )


def _tl_to_dict(s: TranslatedSegment) -> dict:
    return dataclasses.asdict(s)


def _dict_to_tl(d: dict) -> TranslatedSegment:
    words = [WordTimestamp(**w) for w in d.get("words", [])]
    return TranslatedSegment(
        text=d["text"], start=d["start"], end=d["end"],
        words=words,
        source_text=d.get("source_text", ""),
        source_language=d.get("source_language", ""),
        target_language=d.get("target_language", ""),
    )


def _ctx_to_dict(ctx: ContextBundle) -> dict:
    return dataclasses.asdict(ctx)


# ------------------------------------------------------------------ #
# Remote STT proxy (transcribe + detect_language)                     #
# ------------------------------------------------------------------ #

class RemoteSTTProxy(BaseSTTModel):
    """Proxy for qwen3_asr."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self) -> None:
        pass  # no local weights to load

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[TranscriptSegment]:
        from app.ml.remote_client import call_multipart
        with open(audio_path, "rb") as f:
            data = {"model": self.model_name}
            if language:
                data["language"] = language
            resp = call_multipart("/v1/asr/transcribe", files={"audio": f}, data=data)
        return [_dict_to_seg(s) for s in resp.get("segments", [])]

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        from app.ml.remote_client import call_multipart
        with open(audio_path, "rb") as f:
            resp = call_multipart(
                "/v1/asr/detect_language",
                files={"audio": f},
                data={"model": self.model_name},
            )
        return resp.get("language", "en"), float(resp.get("confidence", 0.9))


# ------------------------------------------------------------------ #
# Remote translator proxy (translate + optional refine)               #
# ------------------------------------------------------------------ #

class RemoteTranslatorProxy(BaseTranslator):
    """Proxy for qwen2_5_vl."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self) -> None:
        pass

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
        from app.ml.remote_client import call_multipart, Remote524Error

        # Read file bytes upfront so they can be replayed on a 524 retry.
        audio_bytes: Optional[bytes] = None
        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        frame_bytes: List[bytes] = []
        if frames:
            for fp in frames:
                with open(str(fp), "rb") as f:
                    frame_bytes.append(f.read())

        def _build_files(segs: List[TranscriptSegment]) -> dict:
            payload = {
                "model": self.model_name,
                "segments": [_seg_to_dict(s) for s in segs],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "history": [_tl_to_dict(h) for h in history],
                "context": _ctx_to_dict(context) if context else None,
                "glossary_text": glossary_text,
            }
            files: dict = {"payload": (None, json.dumps(payload), "application/json")}
            if audio_bytes is not None:
                files["audio"] = ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
            for i, fb in enumerate(frame_bytes):
                files[f"frame_{i}"] = (f"frame_{i}.jpg", io.BytesIO(fb), "image/jpeg")
            return files

        try:
            resp = call_multipart("/v1/translate", files=_build_files(segments))
            return [_dict_to_tl(s) for s in resp.get("segments", [])]
        except Remote524Error:
            # Cloudflare upstream timeout — retry once with segments split in half.
            # The worker also batches ≤4 segs per model call, so this is a belt-and-suspenders
            # fallback for edge cases (very long individual chunks).
            logger.warning(
                "Remote translate 524; retrying with halved segment batches (%d → 2×%d)",
                len(segments), len(segments) // 2 or 1,
            )
            mid = max(1, len(segments) // 2)
            results: List[TranslatedSegment] = []
            for chunk_segs in (segments[:mid], segments[mid:]):
                if not chunk_segs:
                    continue
                resp = call_multipart("/v1/translate", files=_build_files(chunk_segs))
                results.extend(_dict_to_tl(s) for s in resp.get("segments", []))
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
        """Match QwenVLTranslator.refine signature; forwards to worker /v1/translate/refine."""
        from app.ml.remote_client import call_json

        payload = {
            "model": self.model_name,
            "first_pass": first_pass,
            "source_text": source_text,
            "prev_text": prev_chunk,
            "next_text": next_chunk,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "start": start,
            "end": end,
            "context": _ctx_to_dict(context) if context else None,
            "glossary_text": glossary_text,
        }
        resp = call_json("/v1/translate/refine", json_data=payload)
        return resp.get("text", first_pass)

    def _call_model(
        self,
        source_text: str,
        system_prompt: str,
        history_text: str,
        frames=None,
    ) -> str:
        """Forward a raw text-generation call to the worker's /v1/generate endpoint."""
        from app.ml.remote_client import call_json

        payload = {
            "model": self.model_name,
            "source_text": source_text,
            "system_prompt": system_prompt,
            "history_text": history_text or "",
        }
        resp = call_json("/v1/generate", json_data=payload)
        return resp.get("text", "")


# ------------------------------------------------------------------ #
# Remote denoiser proxy                                               #
# ------------------------------------------------------------------ #

class RemoteDenoiser:
    """Proxy for Demucs + noisereduce + VAD denoising."""

    def denoise(self, raw_wav: str, output_dir: str, job_id: str) -> tuple:
        """Upload raw_wav, run denoising on the worker, return (clean_wav_path, speech_intervals)."""
        import base64
        from pathlib import Path as _Path
        from app.ml.remote_client import call_multipart_binary, _get_client
        from app.config import settings as _settings

        import httpx

        # Upload raw wav and get back clean wav + speech intervals
        with open(raw_wav, "rb") as f:
            client = _get_client()
            headers = {}
            if _settings.REMOTE_GPU_TOKEN:
                headers["Authorization"] = f"Bearer {_settings.REMOTE_GPU_TOKEN}"
            resp = client.post(
                "/v1/denoise",
                files={"audio": f},
                data={"job_id": job_id},
                timeout=_settings.REMOTE_GPU_TIMEOUT_S,
            )
            resp.raise_for_status()

        # Save clean wav
        clean_wav_path = str(_Path(output_dir) / "clean.wav")
        with open(clean_wav_path, "wb") as out:
            out.write(resp.content)

        # Parse speech intervals from response header (base64-encoded JSON array)
        intervals_b64 = resp.headers.get("X-Speech-Intervals", "")
        speech_intervals: list = []
        if intervals_b64:
            try:
                speech_intervals = json.loads(base64.b64decode(intervals_b64).decode())
            except Exception:
                pass

        return clean_wav_path, speech_intervals
