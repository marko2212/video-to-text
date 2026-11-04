"""The cross-check of an API transcript against a local Whisper pass.

Small synthetic transcripts: the API side has estimated paragraph times that
bend around a missing stretch, the local side has the recording's own times.
"""

import align
import transcribe

# What was said, as the local pass heard it: one segment per sentence, timed.
_SPEECH = [
    (0.0, 4.0, "Good morning everyone and welcome to the quarterly review."),
    (4.0, 8.0, "Today we look at the numbers for the third quarter."),
    (8.0, 12.0, "Revenue grew by twelve percent against the plan."),
    (12.0, 16.0, "Costs stayed flat thanks to the new supplier contract."),
    (60.0, 64.0, "Let us move on to the product roadmap for next year."),
    (64.0, 68.0, "The migration to the new platform starts in January."),
    (68.0, 72.0, "Every team gets a dedicated contact for the transition."),
    (72.0, 76.0, "Questions about the timeline go to the programme office."),
]


def _segments(speech=_SPEECH):
    return [{"start": s, "end": e, "text": t} for s, e, t in speech]


def test_words_are_compared_without_case_punctuation_or_script_games():
    assert align.normalise("Hello, World! It's 'fine'.") == [
        "hello",
        "world",
        "it's",
        "fine",
    ]
    assert align.normalise("Данас причамо о буџету.") == [
        "данас",
        "причамо",
        "о",
        "буџету",
    ]
    assert align.normalise("don’t stop") == ["don't", "stop"]


def test_paragraphs_take_the_local_pass_s_times():
    # The API answered one chunk of 80 s; its paragraphs are stamped by
    # character share, which puts the second one at ~0:40 although it was
    # spoken at 1:00 (the 44 s pause in the middle is invisible to a share).
    paragraphs = [
        {
            "start": 0.0,
            "text": "Good morning everyone and welcome to the quarterly review. "
            "Today we look at the numbers for the third quarter. Revenue grew by "
            "twelve percent against the plan. Costs stayed flat thanks to the "
            "new supplier contract.",
        },
        {
            "start": 40.0,
            "text": "Let us move on to the product roadmap for next year. The "
            "migration to the new platform starts in January. Every team gets a "
            "dedicated contact for the transition. Questions about the timeline "
            "go to the programme office.",
        },
    ]

    checked = align.cross_check(paragraphs, _segments())

    first, second = checked["paragraphs"]
    assert first["exact"] and second["exact"]
    assert first["start"] == 0.0
    assert second["start"] == 60.0
    assert checked["gaps"] == []
    stats = checked["stats"]
    assert (stats["paragraphs"], stats["anchored"], stats["gaps"]) == (2, 2, 0)
    assert stats["matched_share"] == 1.0


def test_a_paragraph_the_local_pass_does_not_know_moves_with_its_neighbour():
    paragraphs = [
        # Estimated 10 s late: spoken at 0:00.
        {"start": 10.0, "text": "Good morning everyone and welcome to the review."},
        # Nothing the local pass heard: it keeps its estimate, shifted by the
        # drift of the nearest re-timed paragraph — the earlier one on a tie.
        {"start": 20.0, "text": "Mumbled words nobody else transcribed at all."},
        # Estimated 20 s early: spoken at 1:00.
        {"start": 40.0, "text": "Let us move on to the product roadmap for next year."},
        {"start": 70.0, "text": "More mumbling that only the API heard."},
    ]

    checked = align.cross_check(paragraphs, _segments())

    first, shifted, third, last = checked["paragraphs"]
    assert first["exact"] and first["start"] == 0.0
    assert third["exact"] and third["start"] == 60.0
    assert not shifted["exact"] and shifted["start"] == 10.0  # 20 - 10
    assert not last["exact"] and last["start"] == 90.0  # 70 + 20
    assert checked["stats"]["shifted"] == 2


def test_a_stretch_the_api_lacks_is_added_with_the_local_times():
    # The API answer skipped the whole roadmap part (four sentences, 38 words).
    paragraphs = [
        {
            "start": 0.0,
            "text": "Good morning everyone and welcome to the quarterly review. "
            "Today we look at the numbers for the third quarter.",
        },
        {
            "start": 30.0,
            "text": "Revenue grew by twelve percent against the plan. Costs stayed "
            "flat thanks to the new supplier contract.",
        },
    ]

    checked = align.cross_check(paragraphs, _segments())

    [gap] = checked["gaps"]
    assert gap["start"] == 60.0
    assert gap["text"].startswith("Let us move on to the product roadmap")
    assert gap["text"].endswith("go to the programme office.")
    assert gap["words"] == 38
    stats = checked["stats"]
    assert (stats["gaps"], stats["added_words"], stats["api_only_words"]) == (1, 38, 0)
    # Reading order: the added stretch comes after the paragraph before it.
    kinds = [
        item["kind"] for item in align.merge(checked["paragraphs"], checked["gaps"])
    ]
    assert kinds == ["paragraph", "paragraph", "gap"]


def test_a_few_api_words_for_a_long_local_stretch_still_count_as_a_gap():
    # The API answer resumed mid-sentence with a few words of its own where
    # the local pass has four sentences: a `replace`, not a clean `insert`.
    paragraphs = [
        {
            "start": 0.0,
            "text": "Good morning everyone and welcome to the quarterly review.",
        },
        {"start": 10.0, "text": "Um so yes."},
        {
            "start": 20.0,
            "text": "Questions about the timeline go to the programme office.",
        },
    ]

    checked = align.cross_check(paragraphs, _segments())

    assert checked["stats"]["gaps"] == 1
    assert checked["gaps"][0]["text"].startswith("Today we look at the numbers")


def test_short_differences_are_wording_not_gaps():
    # The API has every sentence, worded a little differently here and there.
    paragraphs = [
        {
            "start": 0.0,
            "text": "Good morning everyone, welcome to the quarterly review. Today we "
            "look at the figures for the third quarter. Revenue grew by twelve "
            "percent against plan. Costs stayed flat thanks to the new supplier "
            "contract.",
        },
        {
            "start": 40.0,
            "text": "Let's move on to the product roadmap for next year. The migration "
            "to the new platform starts in January. Each team gets a dedicated "
            "contact for the transition. Questions about the timeline go to the "
            "programme office.",
        },
    ]

    checked = align.cross_check(paragraphs, _segments())

    assert checked["gaps"] == []
    assert checked["stats"]["gaps"] == 0
    assert 0.8 < checked["stats"]["matched_share"] < 1.0


def test_an_added_stretch_is_broken_into_paragraphs_at_sentence_ends():
    long_speech = [
        (float(10 * i), float(10 * i + 9), f"Sentence number {i} of the lost part.")
        for i in range(12)
    ]
    paragraphs = [
        {"start": 0.0, "text": "Nothing of this was in the API answer at all."}
    ]

    checked = align.cross_check(paragraphs, _segments(long_speech))

    gaps = checked["gaps"]
    assert checked["stats"]["gaps"] == 1 and len(gaps) == 3  # 120 s in ~40 s pieces
    assert [gap["start"] for gap in gaps] == [0.0, 40.0, 80.0]
    assert gaps[0]["text"].endswith("of the lost part.")
    assert sum(gap["words"] for gap in gaps) == checked["stats"]["added_words"]


def test_an_added_stretch_breaks_where_a_segment_starts_a_sentence_without_a_stop():
    # faster-whisper `base` left every sentence end unpunctuated on a
    # synthetic recording (2026-11-03): the capital letter that starts the
    # next segment is the only sign of a sentence boundary.
    unpunctuated = [
        (float(10 * i), float(10 * i + 9), f"Sentence number {i} of the lost part")
        for i in range(12)
    ]
    paragraphs = [{"start": 0.0, "text": "Nothing of this was in the API answer."}]

    checked = align.cross_check(paragraphs, _segments(unpunctuated))

    assert [gap["start"] for gap in checked["gaps"]] == [0.0, 40.0, 80.0]
    assert checked["gaps"][0]["text"].startswith("Sentence number 0 of the lost part")


def test_merge_puts_added_paragraphs_where_their_time_falls():
    paragraphs = [
        {"start": 10.0, "text": "a", "exact": True},
        {"start": 30.0, "text": "b", "exact": True},
    ]
    gaps = [
        {"start": 5.0, "text": "before", "words": 1},
        {"start": 20.0, "text": "between", "words": 1},
        {"start": 40.0, "text": "after", "words": 1},
    ]

    assert [item["text"] for item in align.merge(paragraphs, gaps)] == [
        "before",
        "a",
        "between",
        "b",
        "after",
    ]


def test_paragraph_order_is_kept_even_when_times_disagree():
    paragraphs = [
        {"start": 50.0, "text": "Let us move on to the product roadmap for next year."},
        # Ten words the local pass lacks, then a matched sentence: its start
        # is moved ahead of the matched words, which would put it before the
        # paragraph above — so it is held at that one's time.
        {
            "start": 55.0,
            "text": "um er so uh well right okay then now erm The migration to the new "
            "platform starts in January.",
        },
    ]

    checked = align.cross_check(paragraphs, _segments())

    first, second = checked["paragraphs"]
    assert first["start"] == 60.0
    assert second["exact"] and second["start"] == 60.0
    assert checked["stats"]["clamped"] == 1


def test_no_local_words_leaves_the_estimates_alone():
    paragraphs = [{"start": 12.0, "text": "Whatever was said."}]

    checked = align.cross_check(paragraphs, [])

    assert checked["paragraphs"][0]["start"] == 12.0
    assert not checked["paragraphs"][0]["exact"]
    assert checked["stats"]["matched_share"] == 0.0


def test_the_verified_transcript_carries_exact_stamps_and_the_added_part():
    pieces = [
        {
            "start": 0.0,
            "duration": 80.0,
            "text": "Good morning everyone and welcome to the quarterly review. Today "
            "we look at the numbers for the third quarter. Revenue grew by twelve "
            "percent against the plan. Costs stayed flat thanks to the new supplier "
            "contract. Questions about the timeline go to the programme office.",
        }
    ]
    notes = [{"time": 62.0, "description": "Roadmap slide"}]

    text, stats = transcribe.build_verified_transcript(
        pieces, _segments(), "base", visual_notes=notes
    )

    lines = text.split("\n\n")
    assert lines[0].startswith("🔍 Checked against a local Whisper pass (base): ")
    assert "1 stretch(es) with 29 words" in lines[0]
    # Four sentences in the first paragraph, the fifth — spoken after the
    # gap, at 1:12 — in a paragraph of its own; the added stretch and the
    # on-screen note fall between them, in time order.
    assert lines[1].startswith("(0:00) Good morning everyone")
    assert lines[2].startswith("🧩 (1:00) Let us move on to the product roadmap")
    assert lines[3] == "🖥️ (1:02) Roadmap slide"
    assert lines[4] == "(1:12) Questions about the timeline go to the programme office."
    assert "(~" not in text
    assert stats["model"] == "base" and stats["added_words"] == 29
