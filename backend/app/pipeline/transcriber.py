"""Run STT on a single chunk using the model selected by the router."""
import logging
from typing import List
from app.ml.base import TranscriptSegment, ModelNotReady
from app.ml import registry

logger = logging.getLogger(__name__)

_FALLBACK_MODEL = "whisper_turbo"


def transcribe_chunk(chunk_path: str, model_name: str, language: str) -> List[TranscriptSegment]:
    """
    Transcribe a single chunk WAV file.
    Falls back to whisper_turbo if the primary model fails.
    Raises ModelNotReady only if whisper_turbo itself isn't ready.
    """
    if model_name != _FALLBACK_MODEL:
        try:
            model = registry.get(model_name)
            return model.transcribe(chunk_path)
        except ModelNotReady:
            raise
        except Exception as exc:
            logger.warning(f"{model_name} failed ({exc}), falling back to {_FALLBACK_MODEL}")

    # Whisper path (primary or fallback)
    model = registry.get(_FALLBACK_MODEL)
    return model.transcribe(chunk_path, language=language)
