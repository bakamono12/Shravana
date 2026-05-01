"""Basic format tests for the subtitle generator — no models required."""
import pytest
from app.ml.base import TranscriptSegment, WordTimestamp
from app.pipeline.subtitle_generator import generate_srt, generate_vtt
import tempfile, os


def _sample_segments():
    return [
        TranscriptSegment(
            text="Hello world",
            start=0.0,
            end=2.5,
            words=[
                WordTimestamp("Hello", 0.0, 1.0),
                WordTimestamp("world", 1.1, 2.5),
            ],
            language="en",
        ),
        TranscriptSegment(
            text="This is a test",
            start=3.0,
            end=5.5,
            words=[
                WordTimestamp("This", 3.0, 3.5),
                WordTimestamp("is", 3.6, 3.9),
                WordTimestamp("a", 4.0, 4.2),
                WordTimestamp("test", 4.3, 5.5),
            ],
            language="en",
        ),
    ]


def test_srt_format():
    segs = _sample_segments()
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
        path = f.name
    try:
        generate_srt(segs, path)
        content = open(path).read()
        assert "00:00:00," in content
        assert "-->" in content
        assert "Hello world" in content or "Hello" in content
    finally:
        os.unlink(path)


def test_vtt_format():
    segs = _sample_segments()
    with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as f:
        path = f.name
    try:
        generate_vtt(segs, path)
        content = open(path).read()
        assert content.startswith("WEBVTT")
        assert "00:00:00." in content
    finally:
        os.unlink(path)
