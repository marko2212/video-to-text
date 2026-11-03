"""A missing ffmpeg: what the code raises and what the page shows (no ffmpeg needed)."""

import pytest
from streamlit.testing.v1 import AppTest

import audio
import config
from audio import missing_tools as real_missing_tools  # conftest patches the name
from exceptions import FfmpegMissingError

_APP = """
import app
app.main()
"""


def _not_installed(*args, **kwargs):
    # What subprocess raises on Windows when the program is not on PATH.
    raise FileNotFoundError(2, "The system cannot find the file specified")


def test_missing_tools_names_only_the_programs_not_on_path(monkeypatch):
    found = {"ffmpeg": "C:/ffmpeg/bin/ffmpeg.exe"}
    monkeypatch.setattr(audio.shutil, "which", found.get)

    assert real_missing_tools() == ["ffprobe"]


def test_missing_tools_is_empty_when_both_are_found(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"C:/bin/{name}.exe")

    assert real_missing_tools() == []


def test_the_message_names_the_programs_the_command_and_the_restart():
    message = audio.missing_tools_message(["ffmpeg", "ffprobe"])

    assert message.startswith("**ffmpeg** and **ffprobe** were not found")
    assert f"`{config.FFMPEG_WINDOWS_INSTALL}`" in message
    assert "close the app and start it again" in message


def test_to_wav_without_ffmpeg_says_ffmpeg_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.ffmpeg, "run", _not_installed)

    with pytest.raises(FfmpegMissingError, match=r"^\*\*ffmpeg\*\* was not found"):
        audio.to_wav(tmp_path / "talk.mp4", tmp_path / "talk.wav")


def test_to_preview_without_ffmpeg_says_ffmpeg_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.ffmpeg, "run", _not_installed)

    with pytest.raises(FfmpegMissingError, match=r"^\*\*ffmpeg\*\* was not found"):
        audio.to_preview(tmp_path / "talk.amr", tmp_path / "talk.mp3")


def test_the_page_says_at_the_top_that_ffmpeg_is_missing(monkeypatch):
    monkeypatch.setattr(audio, "missing_tools", lambda: ["ffmpeg", "ffprobe"])

    test = AppTest.from_string(_APP, default_timeout=30).run()

    [error] = test.error
    assert error.value.startswith("**ffmpeg** and **ffprobe** were not found")


def test_the_page_shows_no_warning_when_both_are_installed():
    test = AppTest.from_string(_APP, default_timeout=30).run()

    assert not test.error
