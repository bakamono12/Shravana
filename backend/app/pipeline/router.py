"""Route each chunk to the appropriate STT model based on detected language."""
from app.config import settings

LANGUAGE_MODEL_MAP: dict[str, str] = {
    "en": "parakeet",
    "hi": "qwen3_asr",
    "mr": "whisper_turbo",
}
DEFAULT_MODEL = "whisper_turbo"


def route(language: str, confidence: float) -> str:
    """
    Return the model name for the given language code.
    Routes Hindi+English code-switching to qwen3_asr.
    """
    # Code-switching: low confidence + one of the langs is English → use qwen3_asr
    if language == "en" and confidence < settings.CODE_SWITCH_THRESHOLD:
        return "qwen3_asr"
    return LANGUAGE_MODEL_MAP.get(language, DEFAULT_MODEL)
