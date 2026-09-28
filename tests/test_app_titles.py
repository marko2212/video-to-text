"""The AI title in the page: settings, runs, History and file names (no network)."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
import config
import db
import titles
import transcribe
from exceptions import TitleError

_TIMEOUT = 30
_APP = """
import app
app.main()
"""
_RUN_SCRIPT = """
import streamlit as st
import app
import db

db.init_db()
app.init_session_state()
st.session_state.original_filename = "Meeting Recording.mp4"
st.session_state.audio_path = st.session_state.audio
app.run_transcription(st.session_state.provider, st.session_state.model, False, "audio")
app.render_run_notices()
app.render_results()
"""
_TITLE_RECORD = {
    "kind": "title",
    "model": "gpt-5.4-nano",
    "input_tokens": 3_000,
    "output_tokens": 5,
    "cost_usd": 0.001,
    "estimated": False,
}


def _prefer(mode: str) -> None:
    db.init_db()
    db.set_preference("title_mode", mode)


@pytest.fixture
def engines(monkeypatch):
    """Fake both engines; each writes a transcript. OpenAI reports its cost."""

    def fake_openai(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("We went over the budget.", encoding="utf-8")
        return {
            "requests": 1,
            "seconds": 60.0,
            "input_tokens": 1_000,
            "output_tokens": 200,
            "cost_usd": 0.06,
            "estimated": False,
        }

    def fake_local(input_file, output_file, whisper_model, **kwargs):
        Path(output_file).write_text("Local talk about the budget.", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_openai", fake_openai)
    monkeypatch.setattr(transcribe, "transcribe_local", fake_local)
    monkeypatch.setattr(app, "load_whisper_model", lambda *a: object())
    monkeypatch.setattr(app, "_model_is_cached", lambda name: True)


@pytest.fixture
def title_requests(monkeypatch):
    """Answer every title request with "Budget review"; list what was sent."""
    sent = []

    def fake(transcript, api_key, model):
        sent.append((transcript, model))
        return "Budget review", dict(_TITLE_RECORD, model=model)

    monkeypatch.setattr(titles, "make_title", fake)
    return sent


def _run(tmp_path, provider, model) -> AppTest:
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"audio")
    test = AppTest.from_string(_RUN_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["audio"] = audio
    test.session_state["provider"] = provider
    test.session_state["model"] = model
    return test.run()


def test_an_openai_run_is_named_and_the_title_counted(
    tmp_path, engines, title_requests
):
    _prefer(config.TITLE_MODE_APPEND)

    test = _run(tmp_path, config.PROVIDER_OPENAI, "whisper-1")

    assert title_requests == [("We went over the budget.", config.DEFAULT_TITLE_MODEL)]
    [row] = db.list_transcriptions()
    assert row["title"] == "Budget review"
    assert row["filename"] == "Meeting Recording.mp4"
    assert row["cost_usd"] == pytest.approx(0.061)
    saved = json.loads(row["usage_json"])
    assert saved["title"]["input_tokens"] == 3_000
    assert any("🏷️ AI title: Budget review" in c.value for c in test.caption)
    assert any("title 3,000 → 5 tokens" in c.value for c in test.caption)


def test_a_local_run_is_named_too(tmp_path, engines, title_requests):
    _prefer(config.TITLE_MODE_REPLACE)

    _run(tmp_path, config.PROVIDER_LOCAL, "base")

    assert title_requests[0][0] == "Local talk about the budget."
    [row] = db.list_transcriptions()
    assert row["title"] == "Budget review"
    assert row["cost_usd"] == pytest.approx(0.001)


def test_with_titles_off_nothing_is_sent(tmp_path, engines, title_requests):
    # Off is the default: no preference stored.
    _run(tmp_path, config.PROVIDER_OPENAI, "whisper-1")

    assert title_requests == []
    [row] = db.list_transcriptions()
    assert row["title"] is None
    assert json.loads(row["usage_json"])["title"] is None


def test_a_title_that_fails_only_warns_and_its_cost_still_counts(
    tmp_path, engines, monkeypatch
):
    def empty_answer(transcript, api_key, model):
        error = TitleError("OpenAI returned no title")
        error.usage_record = _TITLE_RECORD
        raise error

    monkeypatch.setattr(titles, "make_title", empty_answer)
    _prefer(config.TITLE_MODE_APPEND)

    test = _run(tmp_path, config.PROVIDER_OPENAI, "whisper-1")

    assert any("No AI title: OpenAI returned no title" in w.value for w in test.warning)
    assert not test.error
    [row] = db.list_transcriptions()
    assert row["title"] is None
    assert row["cost_usd"] == pytest.approx(0.061)


def test_without_a_key_a_local_run_is_saved_untitled(
    tmp_path, engines, title_requests, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config.get_settings.cache_clear()
    _prefer(config.TITLE_MODE_APPEND)

    test = _run(tmp_path, config.PROVIDER_LOCAL, "base")

    assert title_requests == []
    assert any("needs an OpenAI API key" in i.value for i in test.info)
    [row] = db.list_transcriptions()
    assert row["title"] is None


def test_the_title_settings_are_kept_for_the_next_session():
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    [mode] = test.sidebar.radio
    [model] = test.sidebar.selectbox
    assert mode.value == config.TITLE_MODE_OFF
    assert model.disabled  # nothing to choose while titles are off

    mode.set_value(config.TITLE_MODE_REPLACE).run()
    test.sidebar.selectbox[0].set_value("gpt-5.4-mini").run()

    assert db.get_preferences() == {
        "title_mode": config.TITLE_MODE_REPLACE,
        "title_model": "gpt-5.4-mini",
    }
    later = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    assert later.sidebar.radio[0].value == config.TITLE_MODE_REPLACE
    assert later.sidebar.selectbox[0].value == "gpt-5.4-mini"
    assert not later.sidebar.selectbox[0].disabled


def test_a_stored_model_no_longer_offered_falls_back_to_the_default():
    db.init_db()
    db.set_preference("title_model", "gpt-3-retired")

    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()

    assert test.sidebar.selectbox[0].value == config.DEFAULT_TITLE_MODEL


def _history_labels() -> list[str]:
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    return [expander.label for expander in test.expander]


@pytest.mark.parametrize(
    ("mode", "shown"),
    [
        (config.TITLE_MODE_OFF, "Meeting Recording.mp4"),
        (config.TITLE_MODE_REPLACE, "Budget review"),
        (config.TITLE_MODE_APPEND, "Meeting Recording - Budget review"),
    ],
)
def test_history_shows_the_title_as_the_mode_says(mode, shown):
    _prefer(mode)
    db.add_transcription(
        "Meeting Recording.mp4", "video", "whisper-1", False, "t", title="Budget review"
    )
    db.add_transcription("untitled.wav", "audio", "whisper-1", False, "t")

    labels = _history_labels()

    assert labels[0].startswith("untitled.wav · ")  # newest first, never titled
    assert labels[1].startswith(f"{shown} · ")


_DOWNLOAD_SCRIPT = """
import streamlit as st
import app

st.session_state.stems = [
    app._download_stem("Meeting Recording.mp4", "Budget review"),
    app._download_stem("Meeting Recording.mp4", None),
]
"""


@pytest.mark.parametrize(
    ("mode", "titled"),
    [
        (config.TITLE_MODE_OFF, "transcript_Meeting Recording"),
        (config.TITLE_MODE_REPLACE, "Budget review"),
        (config.TITLE_MODE_APPEND, "Meeting Recording - Budget review"),
    ],
)
def test_downloaded_files_are_named_as_the_mode_says(mode, titled):
    _prefer(mode)

    test = AppTest.from_string(_DOWNLOAD_SCRIPT, default_timeout=_TIMEOUT).run()

    assert test.session_state.stems == [titled, "transcript_Meeting Recording"]
