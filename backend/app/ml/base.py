from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


# ------------------------------------------------------------------ #
# Translation types                                                    #
# ------------------------------------------------------------------ #

@dataclass
class SpeakerProfile:
    speaker_id: str
    dominant_language: str = "en"
    register: str = "conversational"      # formal|conversational|colloquial|technical
    code_switch_tendency: str = "none"    # heavy|moderate|none


@dataclass
class ContextBundle:
    domain: str = "casual"
    format: str = "monologue"
    source_language: str = "en"
    target_language: str = "en"
    named_entities: List[str] = field(default_factory=list)
    idioms_detected: List[str] = field(default_factory=list)
    speakers: dict = field(default_factory=dict)    # speaker_id -> SpeakerProfile
    scene_description: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class TranslatedSegment:
    text: str                   # translated text
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)  # timings from source, word replaced
    source_text: str = ""
    source_language: str = "en"
    target_language: str = "en"


class BaseTranslator(ABC):
    name: str = ""
    supported_targets: set = field(default_factory=set)

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
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
        """Translate a list of transcript segments. Returns one TranslatedSegment per input."""


# ------------------------------------------------------------------ #
# STT                                                                  #
# ------------------------------------------------------------------ #

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
