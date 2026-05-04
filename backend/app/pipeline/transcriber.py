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
    Falls back to local whisper_turbo if the primary model fails.
    When the remote worker is unreachable, uses get_local() so the fallback
    doesn't also try the dead remote (REMOTE_GPU_MODELS=ALL would otherwise
    route whisper_turbo remotely too).
    Raises ModelNotReady only if whisper_turbo itself isn't ready locally.
    """
    from app.ml.remote_client import RemoteCallError
    from app.config import settings as _settings

    def _local_fallback() -> List[TranscriptSegment]:
        logger.warning(f"Remote worker unreachable — running {_FALLBACK_MODEL} locally")
        model = registry.get_local(_FALLBACK_MODEL)
        return model.transcribe(chunk_path, language=language)

    if model_name != _FALLBACK_MODEL:
        try:
            model = registry.get(model_name)
            return model.transcribe(chunk_path)
        except ModelNotReady:
            raise
        except RemoteCallError as exc:
            if not _settings.REMOTE_GPU_FALLBACK_LOCAL:
                raise
            return _local_fallback()
        except Exception as exc:
            logger.warning(f"{model_name} failed ({exc}), falling back to {_FALLBACK_MODEL}")

    # whisper_turbo path (primary or non-remote-error fallback)
    try:
        model = registry.get(_FALLBACK_MODEL)
        return model.transcribe(chunk_path, language=language)
    except RemoteCallError as exc:
        if not _settings.REMOTE_GPU_FALLBACK_LOCAL:
            raise
        return _local_fallback()
