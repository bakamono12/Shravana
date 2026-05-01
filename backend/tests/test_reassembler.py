"""Test overlap deduplication logic — no models required."""
from app.ml.base import TranscriptSegment, WordTimestamp
from app.pipeline.reassembler import deduplicate_overlap, reassemble, serialize_segments


def _seg(text, start, end):
    return TranscriptSegment(text=text, start=start, end=end, words=[
        WordTimestamp(text, start, end)
    ], language="en")


def test_deduplicate_removes_overlap():
    a = [_seg("one", 0, 3), _seg("two", 3, 5), _seg("three", 5, 7)]
    b = [_seg("three", 5, 7), _seg("four", 7, 9), _seg("five", 9, 11)]
    a2, b2 = deduplicate_overlap(a, b, overlap=4.0)  # half = 2s
    # a should cut at end_of_a - 2 = 7 - 2 = 5 → only segs with start < 5
    assert all(s.start < 5.0 for s in a2), f"a2 starts: {[s.start for s in a2]}"
    # b should start after first_of_b + 2 = 5 + 2 = 7 → only segs with end > 7
    assert all(s.end > 7.0 for s in b2), f"b2 ends: {[s.end for s in b2]}"


def test_reassemble_empty():
    result = reassemble([])
    assert result == []


def test_reassemble_single_chunk():
    segs = [_seg("hello", 0.5, 2.0), _seg("world", 2.1, 3.5)]
    chunk_data = [{"sequence": 0, "start_time": 0.0, "transcript_json": serialize_segments(segs)}]
    result = reassemble(chunk_data)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert abs(result[0].start - 0.5) < 0.01
