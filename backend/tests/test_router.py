"""Test model routing logic — no models required."""
from app.pipeline.router import route


def test_english_routes_to_parakeet():
    assert route("en", 0.95) == "parakeet"


def test_hindi_routes_to_qwen():
    assert route("hi", 0.90) == "qwen3_asr"


def test_marathi_routes_to_whisper():
    assert route("mr", 0.88) == "whisper_turbo"


def test_code_switch_routes_to_qwen():
    # Low-confidence English → likely code-switched Hindi-English
    assert route("en", 0.70) == "qwen3_asr"


def test_unknown_lang_routes_to_whisper():
    assert route("fr", 0.95) == "whisper_turbo"
