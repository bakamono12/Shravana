"""Registry of supported translation target languages.

All translation is handled by Qwen2.5-VL. Add a new language by appending
an entry — the model handles any language it knows.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LangInfo:
    name: str


SUPPORTED_LANGUAGES: dict[str, LangInfo] = {
    "en": LangInfo("English"),
    "hi": LangInfo("Hindi"),
    "mr": LangInfo("Marathi"),
    "ta": LangInfo("Tamil"),
    "te": LangInfo("Telugu"),
    "bn": LangInfo("Bengali"),
    "gu": LangInfo("Gujarati"),
    "kn": LangInfo("Kannada"),
    "ml": LangInfo("Malayalam"),
    "pa": LangInfo("Punjabi"),
    "ur": LangInfo("Urdu"),
    "fr": LangInfo("French"),
    "de": LangInfo("German"),
    "es": LangInfo("Spanish"),
    "ja": LangInfo("Japanese"),
    "zh": LangInfo("Chinese"),
    "ar": LangInfo("Arabic"),
    "ru": LangInfo("Russian"),
    "pt": LangInfo("Portuguese"),
    "ko": LangInfo("Korean"),
}


def is_supported(code: str) -> bool:
    return code in SUPPORTED_LANGUAGES
