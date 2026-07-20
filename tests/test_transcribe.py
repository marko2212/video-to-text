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


def test_format_clock():
    assert transcribe._format_clock(0) == "0:00"
    assert transcribe._format_clock(74) == "1:14"
    assert transcribe._format_clock(3675) == "1:01:15"


def test_build_transcript_prefixes_timestamps():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "Hello there."},
        {"start": 2.0, "end": 4.0, "text": "How are you?"},
    ]
    assert (
        transcribe.build_transcript(entries)
        == "(0:00) Hello there. (0:02) How are you?"
    )


def test_build_transcript_breaks_paragraph_on_pause():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "First part."},
        {"start": 10.0, "end": 12.0, "text": "After a long pause."},
    ]
    out = transcribe.build_transcript(entries, paragraph_gap=2.0)
    assert out == "(0:00) First part.\n\n(0:10) After a long pause."


def test_build_transcript_skips_empty_segments():
    entries = [
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": 1.0, "end": 2.0, "text": "Real text."},
    ]
    assert transcribe.build_transcript(entries) == "(0:01) Real text."


def test_split_into_paragraphs_groups_sentences():
    text = "One. Two. Three. Four. Five. Six."
    out = transcribe.split_into_paragraphs(text, sentences_per_paragraph=2)
    assert out == "One. Two.\n\nThree. Four.\n\nFive. Six."
