"""Route each chunk to the appropriate STT model."""


def route(language: str, confidence: float) -> str:
    """All languages are handled by Qwen3-ASR-1.7B (52-language multilingual model)."""
    return "qwen3_asr"
