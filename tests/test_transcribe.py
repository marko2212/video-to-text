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


def test_build_transcript_interleaves_visual_notes_by_time():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "Here are the results."},
        {"start": 10.0, "end": 12.0, "text": "You can see the growth."},
    ]
    notes = [{"time": 5.0, "description": "Slide: Q3 Revenue"}]
    out = transcribe.build_transcript(entries, visual_notes=notes)

    assert out.split("\n\n") == [
        "(0:00) Here are the results.",
        "🖥️ (0:05) Slide: Q3 Revenue",
        "(0:10) You can see the growth.",
    ]


def test_build_transcript_orders_unsorted_notes():
    entries = [{"start": 30.0, "end": 32.0, "text": "Wrapping up."}]
    notes = [
        {"time": 20.0, "description": "Second slide"},
        {"time": 10.0, "description": "First slide"},
    ]
    out = transcribe.build_transcript(entries, visual_notes=notes)

    assert out.split("\n\n") == [
        "🖥️ (0:10) First slide",
        "🖥️ (0:20) Second slide",
        "(0:30) Wrapping up.",
    ]


def test_build_transcript_note_precedes_speech_sharing_its_time():
    entries = [{"start": 5.0, "end": 7.0, "text": "Look at this."}]
    notes = [{"time": 5.0, "description": "Slide shown"}]
    out = transcribe.build_transcript(entries, visual_notes=notes)

    assert out == "🖥️ (0:05) Slide shown\n\n(0:05) Look at this."


def test_build_transcript_note_splits_the_paragraph_it_lands_in():
    entries = [
        {"start": 12.0, "end": 14.0, "text": "Moving on to next year."},
        {"start": 16.0, "end": 18.0, "text": "And who owns each workstream."},
    ]
    notes = [{"time": 15.0, "description": "Next Steps slide"}]
    out = transcribe.build_transcript(entries, visual_notes=notes)

    assert out.split("\n\n") == [
        "(0:12) Moving on to next year.",
        "🖥️ (0:15) Next Steps slide",
        "(0:16) And who owns each workstream.",
    ]


def test_build_transcript_keeps_notes_after_the_last_segment():
    entries = [{"start": 0.0, "end": 1.0, "text": "Start."}]
    notes = [{"time": 90.0, "description": "Closing slide"}]
    out = transcribe.build_transcript(entries, visual_notes=notes)

    assert out.endswith("🖥️ (1:30) Closing slide")


def test_build_transcript_without_notes_is_unchanged():
    entries = [{"start": 0.0, "end": 2.0, "text": "Only speech."}]
    out = transcribe.build_transcript(entries, visual_notes=[])
    assert out == "(0:00) Only speech."


def test_split_into_paragraphs_groups_sentences():
    text = "One. Two. Three. Four. Five. Six."
    out = transcribe.split_into_paragraphs(text, sentences_per_paragraph=2)
    assert out == "One. Two.\n\nThree. Four.\n\nFive. Six."
