"""Detect language of an audio chunk.

Primary: Qwen3-ASR-0.6B LID (when available).
Fallback: faster-whisper detect_language (always available, reliable for en/hi/mr).
Last resort: user hint or 'en'.
"""
import logging
from app.ml.base import ModelNotReady
from app.config import settings

logger = logging.getLogger(__name__)


def detect(audio_path: str, hint: str | None = None) -> tuple[str, float]:
    """Returns (iso_language_code, confidence)."""
    # 1. Try Qwen LID first
    try:
        from app.ml.registry import get
        model = get("qwen_lid")
        lang, conf = model.detect_language(audio_path)
        if conf >= settings.LID_CONFIDENCE_THRESHOLD:
            if hint == "mr" and lang == "hi":
                return "mr", conf
            return lang, conf
        # low confidence — fall through to whisper LID
    except ModelNotReady:
        pass
    except Exception as e:
        logger.warning(f"Qwen LID failed ({e}), falling back to Whisper LID")

    # 2. Faster-whisper language detection (always available)
    try:
        from app.ml.registry import get
        whisper = get("whisper_turbo")
        if whisper._model is not None:
            # transcribe with no language arg — whisper detects lang from first 30s
            # language is determined before yielding any segments, so we don't consume the iter
            _, info = whisper._model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                vad_filter=False,
                beam_size=1,
            )
            lang = _normalize_lang(info.language)
            conf = info.language_probability
            if hint == "mr" and lang == "hi":
                return "mr", conf
            return lang, conf
    except Exception as e:
        logger.warning(f"Whisper LID failed ({e})")

    # 3. Last resort: honour user hint, default to English
    return hint or "en", 0.0


def _normalize_lang(code: str) -> str:
    """Normalize language codes — faster-whisper returns ISO 639-1 already."""
    _MAP = {"english": "en", "hindi": "hi", "marathi": "mr"}  # guard against full names
    code = code.lower()
    return _MAP.get(code, code)
