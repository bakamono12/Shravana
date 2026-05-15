"""Run STT on a single audio chunk using Qwen3-ASR-1.7B."""
import logging
from typing import List
from app.ml.base import TranscriptSegment, ModelNotReady
from app.ml import registry

logger = logging.getLogger(__name__)


def transcribe_chunk(chunk_path: str, language: str) -> List[TranscriptSegment]:
    """Transcribe a single chunk WAV. Raises ModelNotReady if model isn't loaded yet."""
    model = registry.get("qwen3_asr")
    return model.transcribe(chunk_path)
