from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class WordTimestamp:
    word: str
    start: float  # seconds from chunk start
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)
    language: str = "en"


class ModelNotReady(Exception):
    """Raised when a model has not finished downloading yet."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' is not ready yet")


class BaseSTTModel(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at startup."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe a 16kHz mono WAV file. Returns segments with word timestamps."""
