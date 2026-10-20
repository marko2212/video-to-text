"""The slider under the player, headless: what it shows and what the run gets."""

from datetime import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import checkpoints
import config
import db
import transcribe
import waveform
from exceptions import AudioProcessingError

_APP = """
import app
app.main()
"""
_TIMEOUT = 30
_WHOLE = (time(0, 0, 0), time(0, 0, 6))


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the OpenAI pipeline; records the keyword arguments of each call."""
    calls = []

    def fake(input_file, output_file, api_key, **kwargs):
        calls.append(kwargs)
        Path(output_file).write_text("Transcript", encoding="utf-8")
        kwargs["progress_callback"]({"status": "complete", "message": "done"})
        return {
            "requests": 1,
            "seconds": 3.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.01,
            "estimated": False,
            "audio_seconds": 3.0,
        }

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    return calls


def _page(wav_file, name: str = "talk.wav") -> AppTest:
    """The page with a six-second recording uploaded: 4 s of speech, 2 s of quiet."""
    talk = wav_file(name, (4, 0.8), (2, 0.0))
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.file_uploader[0].set_value((name, talk.read_bytes(), "audio/wav")).run()
    return test


def _slider(test: AppTest):
    return next(s for s in test.slider if s.label == "Transcribe from – to")


def _start(test: AppTest):
    return next(b for b in test.button if b.label == "Start Transcription")


def _whole_button(test: AppTest):
    return next((b for b in test.button if b.label == "Whole recording"), None)


def _captions(test: AppTest) -> list[str]:
    return [c.value for c in test.caption]


def test_a_recording_gets_the_slider_over_its_whole_length(wav_file):
    test = _page(wav_file)

    assert tuple(_slider(test).value) == _WHOLE
    assert "The whole recording, **0:06**, is sent." in _captions(test)
    assert _whole_button(test) is None


def test_the_run_gets_only_the_chosen_stretch(wav_file, fake_openai):
    test = _page(wav_file)
    _slider(test).set_range(time(0, 0, 1), time(0, 0, 4)).run()

    assert any(
        "**0:01 – 0:04** is sent: 0:03 of 0:06; 0:03 left out — not sent, not paid" in c
        for c in _captions(test)
    )
    _start(test).click().run()

    assert fake_openai[-1]["span"] == (1, 4)
    assert any("0:01–0:04 of the recording transcribed" in c for c in _captions(test))
    [row] = db.list_transcriptions()
    assert '"span": [1, 4]' in row["usage_json"]


def test_the_whole_recording_button_puts_the_slider_back(wav_file):
    test = _page(wav_file)
    _slider(test).set_range(time(0, 0, 1), time(0, 0, 4)).run()
    assert _whole_button(test) is not None

    _whole_button(test).click().run()

    assert tuple(_slider(test).value) == _WHOLE
    assert "The whole recording, **0:06**, is sent." in _captions(test)
    assert _whole_button(test) is None


def test_a_new_recording_starts_with_all_of_it(wav_file):
    test = _page(wav_file)
    _slider(test).set_range(time(0, 0, 1), time(0, 0, 4)).run()

    other = wav_file("other.wav", (2, 0.5), (4, 0.0))
    test.file_uploader[0].set_value(("other.wav", other.read_bytes(), "audio/wav"))
    test.run()

    assert tuple(_slider(test).value) == _WHOLE
    assert "The whole recording, **0:06**, is sent." in _captions(test)


def test_ends_together_leave_nothing_and_hold_start_back(wav_file):
    test = _page(wav_file)
    _slider(test).set_range(time(0, 0, 2), time(0, 0, 2)).run()

    assert any("Nothing is left to transcribe" in w.value for w in test.warning)
    assert _start(test).disabled


def test_a_recording_whose_sound_cannot_be_read_has_no_slider(fake_openai, monkeypatch):
    def unreadable(*args, **kwargs):
        raise AudioProcessingError("ffmpeg: Invalid data found")

    monkeypatch.setattr(waveform, "to_wav", unreadable)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.file_uploader[0].set_value(("talk.wav", b"not audio", "audio/wav")).run()

    assert not any(s.label == "Transcribe from – to" for s in test.slider)
    assert not any("is sent" in c for c in _captions(test))
    assert not test.exception
    _start(test).click().run()
    assert fake_openai[-1]["span"] is None


def test_saved_parts_are_offered_for_the_stretch_they_fit(wav_file):
    test = _page(wav_file)
    digest = checkpoints.file_digest(test.session_state.audio_path)
    whole = transcribe.checkpoint_dir(digest, config.DEFAULT_MODEL)
    stretch = transcribe.checkpoint_dir(digest, config.DEFAULT_MODEL, (1, 4))

    # A part of the whole recording's unfinished run, 0:00–0:05: it lies
    # outside 0:01–0:04, so it is not offered there — but counted exactly for
    # the whole recording.
    checkpoints.save(whole, "chunk_000", {"bounds": [0, 5000], "pieces": []})
    _slider(test).set_range(time(0, 0, 1), time(0, 0, 4)).run()
    assert not any("unfinished run" in c for c in _captions(test))
    _whole_button(test).click().run()
    assert any(
        "1 part(s) of an unfinished run of this recording are saved" in c
        for c in _captions(test)
    )

    # A part within the stretch may match its plan; the hint promises no more.
    checkpoints.save(whole, "chunk_001", {"bounds": [1000, 3000], "pieces": []})
    _slider(test).set_range(time(0, 0, 1), time(0, 0, 4)).run()
    assert any(
        "reuses those whose boundaries match this stretch" in c for c in _captions(test)
    )

    # The stretch's own parts are counted exactly.
    checkpoints.save(stretch, "chunk_000", {"bounds": [1000, 4000], "pieces": []})
    test.run()
    assert any(
        "1 part(s) of an unfinished run of this stretch of the recording" in c
        for c in _captions(test)
    )


def test_a_fraction_of_a_second_at_the_end_is_nothing_to_transcribe(wav_file):
    # 5.5 s: the slider ends at 0:06, but 0:05–0:06 holds half a second.
    short = wav_file("short.wav", (3, 0.8), (2.5, 0.0))
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.file_uploader[0].set_value(("short.wav", short.read_bytes(), "audio/wav"))
    test.run()

    _slider(test).set_range(time(0, 0, 5), time(0, 0, 6)).run()
    assert any("Nothing is left to transcribe" in w.value for w in test.warning)
    assert _start(test).disabled

    _slider(test).set_range(time(0, 0, 4), time(0, 0, 6)).run()
    assert not _start(test).disabled
    assert any("**0:04 – 0:06** is sent: 0:02 of 0:06" in c for c in _captions(test))


_AUDIO_SCRIPT = """
import streamlit as st
from pathlib import Path
import app

app.init_session_state()
st.session_state.upload_id = "day"
st.session_state.result = app.render_audio(
    None, Path(st.session_state.audio), disabled=False
)
"""


def test_a_recording_a_hair_under_a_day_long_gets_no_slider(wav_file, monkeypatch):
    # ceil(86399.5) is a day, which no clock time can show: no slider rather
    # than a ValueError on the page.
    import app

    recording = wav_file("day.wav", (1, 0.5))
    monkeypatch.setattr(
        app, "_audio_loudness", lambda path, size, mtime_ns: ([0.5] * 10, 86399.5)
    )
    test = AppTest.from_string(_AUDIO_SCRIPT)
    test.session_state["audio"] = str(recording)
    test.run()

    assert not test.exception
    assert not test.slider
    assert test.session_state.result == (None, True)
