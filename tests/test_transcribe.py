"""Tests for the pure helpers in the transcription pipeline."""

import transcribe


def test_format_srt_timestamp():
    assert transcribe._format_srt_timestamp(0) == "00:00:00,000"
    assert transcribe._format_srt_timestamp(1.78) == "00:00:01,780"
    assert transcribe._format_srt_timestamp(61.5) == "00:01:01,500"
    assert transcribe._format_srt_timestamp(3661.25) == "01:01:01,250"


def test_format_srt_timestamp_clamps_negatives():
    assert transcribe._format_srt_timestamp(-5) == "00:00:00,000"


def test_build_srt_numbers_and_formats_entries():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 2.0, "end": 5.0, "text": "world"},
    ]
    srt = transcribe.build_srt(entries)
    blocks = srt.strip().split("\n\n")

    assert len(blocks) == 2
    assert blocks[0] == "1\n00:00:00,000 --> 00:00:02,000\nHello"
    assert blocks[1] == "2\n00:00:02,000 --> 00:00:05,000\nworld"


def test_build_srt_preserves_offset_timeline():
    # Simulates entries already offset to the second chunk (start at 1 minute).
    entries = [{"start": 60.0, "end": 63.0, "text": "second chunk"}]
    srt = transcribe.build_srt(entries)
    assert "00:01:00,000 --> 00:01:03,000" in srt


def test_build_srt_empty():
    assert transcribe.build_srt([]) == ""
