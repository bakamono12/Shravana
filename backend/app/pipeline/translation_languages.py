"""Registry of supported translation target languages.

To add a new language: append one entry to SUPPORTED_LANGUAGES.
  key   — BCP-47 / ISO-639-1 code used throughout the pipeline
  name  — human-readable display name for the frontend dropdown
  seamless — supported by SeamlessM4T v2 as a target
  vlm   — supported as a text target by Qwen2.5-VL (all languages the LLM knows)

SeamlessM4T v2 supported targets use the model's internal language codes;
the mapping from BCP-47 → Seamless code is in SEAMLESS_LANG_MAP below.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LangInfo:
    name: str
    seamless: bool = True
    vlm: bool = True
    # Seamless internal language code (e.g. "eng", "hin"). Empty = not supported by Seamless.
    seamless_code: str = ""


SUPPORTED_LANGUAGES: dict[str, LangInfo] = {
    "en":  LangInfo("English",    seamless=True,  vlm=True, seamless_code="eng"),
    "hi":  LangInfo("Hindi",      seamless=True,  vlm=True, seamless_code="hin"),
    "mr":  LangInfo("Marathi",    seamless=True,  vlm=True, seamless_code="mar"),
    "ta":  LangInfo("Tamil",      seamless=True,  vlm=True, seamless_code="tam"),
    "te":  LangInfo("Telugu",     seamless=True,  vlm=True, seamless_code="tel"),
    "bn":  LangInfo("Bengali",    seamless=True,  vlm=True, seamless_code="ben"),
    "gu":  LangInfo("Gujarati",   seamless=True,  vlm=True, seamless_code="guj"),
    "kn":  LangInfo("Kannada",    seamless=True,  vlm=True, seamless_code="kan"),
    "ml":  LangInfo("Malayalam",  seamless=True,  vlm=True, seamless_code="mal"),
    "pa":  LangInfo("Punjabi",    seamless=True,  vlm=True, seamless_code="pan"),
    "ur":  LangInfo("Urdu",       seamless=True,  vlm=True, seamless_code="urd"),
    "fr":  LangInfo("French",     seamless=True,  vlm=True, seamless_code="fra"),
    "de":  LangInfo("German",     seamless=True,  vlm=True, seamless_code="deu"),
    "es":  LangInfo("Spanish",    seamless=True,  vlm=True, seamless_code="spa"),
    "ja":  LangInfo("Japanese",   seamless=True,  vlm=True, seamless_code="jpn"),
    "zh":  LangInfo("Chinese",    seamless=True,  vlm=True, seamless_code="cmn"),
    "ar":  LangInfo("Arabic",     seamless=True,  vlm=True, seamless_code="arb"),
    "ru":  LangInfo("Russian",    seamless=True,  vlm=True, seamless_code="rus"),
    "pt":  LangInfo("Portuguese", seamless=True,  vlm=True, seamless_code="por"),
    "ko":  LangInfo("Korean",     seamless=True,  vlm=True, seamless_code="kor"),
}


def is_supported(code: str, backend: str = "any") -> bool:
    info = SUPPORTED_LANGUAGES.get(code)
    if info is None:
        return False
    if backend == "seamless":
        return info.seamless
    if backend == "vlm":
        return info.vlm
    return True


def seamless_code(bcp47: str) -> str:
    """Return the Seamless internal language code for a BCP-47 key."""
    info = SUPPORTED_LANGUAGES.get(bcp47)
    if info and info.seamless_code:
        return info.seamless_code
    raise ValueError(f"No Seamless code for language '{bcp47}'")
