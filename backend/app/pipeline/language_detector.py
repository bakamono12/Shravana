"""Detect language of an audio chunk using Qwen3-ASR-1.7B (built-in LID)."""
import logging
from app.ml.base import ModelNotReady
from app.config import settings

logger = logging.getLogger(__name__)


def detect(audio_path: str, hint: str | None = None) -> tuple[str, float]:
    """Returns (iso_language_code, confidence)."""
    try:
        from app.ml.registry import get
        model = get("qwen3_asr")
        lang, conf = model.detect_language(audio_path)
        # Marathi and Hindi share close phonetics — honour user hint when confident
        if hint == "mr" and lang == "hi":
            return "mr", conf
        if conf >= settings.LID_CONFIDENCE_THRESHOLD:
            return lang, conf
    except ModelNotReady:
        raise
    except Exception as e:
        logger.warning(f"Qwen3-ASR LID failed ({e})")

    # Honour user hint, default to English
    return hint or "en", 0.0
