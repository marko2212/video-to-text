"""Recordings picked from a folder on this computer (no Streamlit)."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import recordings
from exceptions import FilePickerError


def _file(folder, name, content=b"x", mtime=None):
    path = folder / name
    path.write_bytes(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_a_folder_is_listed_as_it_is(tmp_path):
    assert recordings.parse_location(str(tmp_path)) == (tmp_path, None)


def test_a_pasted_recording_path_means_its_folder_with_that_file(tmp_path):
    video = _file(tmp_path, "Call.mkv")

    # Explorer's "Copy as path" wraps the path in double quotes.
    assert recordings.parse_location(f'  "{video}"  ') == (tmp_path, video)


def test_text_that_names_no_folder_or_recording_is_nothing(tmp_path):
    notes = _file(tmp_path, "notes.txt")

    assert recordings.parse_location("") == (None, None)
    assert recordings.parse_location('""') == (None, None)
    assert recordings.parse_location(str(tmp_path / "missing")) == (None, None)
    assert recordings.parse_location(str(notes)) == (None, None)


def test_only_video_and_audio_files_are_listed_newest_first(tmp_path, monkeypatch):
    # Ordered by creation time; the test uses the modification time, which it
    # can set (os.utime cannot set a creation time).
    monkeypatch.setattr(recordings, "_created", lambda stat: stat.st_mtime)
    old = _file(tmp_path, "old.MP4", mtime=1_000)
    new = _file(tmp_path, "new.m4a", mtime=3_000)
    middle = _file(tmp_path, "middle.wav", mtime=2_000)
    _file(tmp_path, "notes.txt", mtime=4_000)
    (tmp_path / "sub.mkv").mkdir()  # a folder, even with a media extension

    assert recordings.list_recordings(tmp_path) == [new, middle, old]


def test_a_folder_that_cannot_be_read_is_an_error_not_an_empty_folder(tmp_path):
    # "No recordings here" would be wrong for a network drive that is gone.
    with pytest.raises(OSError):
        recordings.list_recordings(tmp_path / "missing")


def test_only_a_full_path_is_accepted(tmp_path, monkeypatch):
    # A relative path (or a bare drive) is read against the app's own folder,
    # whose temp/ and uploads/ hold its working copies.
    (tmp_path / "Recordings").mkdir()
    monkeypatch.chdir(tmp_path)

    assert recordings.parse_location("Recordings") == (None, None)
    assert recordings.parse_location(".") == (None, None)


def test_a_path_typed_in_another_letter_case_is_found_in_the_listing(tmp_path):
    listed = [str(tmp_path / "Call.MKV")]

    found = recordings.find(listed, str(tmp_path / "call.mkv"))

    if os.path.normcase("A") == "a":  # Windows: names ignore case
        assert found == listed[0]
    assert recordings.find(listed, str(tmp_path / "other.mkv")) is None


def test_the_creation_time_orders_the_list_where_the_system_has_one():
    # A recording still being written keeps its creation time, so it does
    # not jump around the list; systems without one fall back to mtime.
    assert recordings._created(SimpleNamespace(st_birthtime=5, st_mtime=9)) == 5
    assert recordings._created(SimpleNamespace(st_mtime=9)) == 9


def test_a_growing_file_keeps_its_entry_in_the_list(tmp_path, monkeypatch):
    # The list keeps the choice by its text: a size in it would lose the choice
    # whenever another recording in the folder grew.
    monkeypatch.setattr(recordings, "_created", lambda stat: 1_000)
    path = _file(tmp_path, "Call.mkv", b"a")
    before = recordings.label(path)

    _file(tmp_path, "Call.mkv", b"a" * 5_000, mtime=9_000)

    assert recordings.label(path) == before
    assert before.startswith("Call.mkv · ")


def test_the_summary_tells_the_size_and_the_last_save(tmp_path):
    path = _file(tmp_path, "Call.mkv", b"a" * (3 * 1024 * 1024))

    assert recordings.summary(path).startswith("3.0 MB · saved ")
    assert recordings.summary(tmp_path / "gone.mkv") == ""


def test_a_recording_that_changes_on_disk_has_a_new_version(tmp_path):
    path = _file(tmp_path, "Call.mkv", b"first", mtime=1_000)
    first = recordings.version(path)

    _file(tmp_path, "Call.mkv", b"second take", mtime=2_000)

    assert recordings.version(path) != first
    assert recordings.version(path)[0] == len(b"second take")


# --- the computer's own file window ---------------------------------------------


class _FakeRun:
    """Stands in for ``subprocess.run``: records the call, answers as told."""

    def __init__(self, stdout="", returncode=0, stderr="", raises=None):
        self.calls = []
        self._answer = SimpleNamespace(
            stdout=stdout, returncode=returncode, stderr=stderr
        )
        self._raises = raises

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self._raises:
            raise self._raises
        return self._answer


def test_the_file_window_returns_the_chosen_recording(tmp_path, monkeypatch):
    run = _FakeRun(stdout="C:/Recordings/Call š.mkv")
    monkeypatch.setattr(recordings.subprocess, "run", run)

    chosen = recordings.ask_for_recording(tmp_path)

    assert chosen == Path("C:/Recordings/Call š.mkv")
    [(args, kwargs)] = run.calls
    # Its own process: Tk must not run in Streamlit's script threads.
    assert args[:2] == [sys.executable, "-c"]
    assert args[4] == str(tmp_path)  # opens in the last folder used
    assert "*.mkv" in args[5] and "*.amr" in args[5]
    assert kwargs["env"]["PYTHONUTF8"] == "1"  # č, ć, š survive the pipe
    assert kwargs["timeout"] > 0


def test_a_cancelled_file_window_chooses_nothing(monkeypatch):
    monkeypatch.setattr(recordings.subprocess, "run", _FakeRun(stdout=""))

    assert recordings.ask_for_recording(None) is None


def test_a_file_window_that_cannot_open_is_an_error(monkeypatch):
    failed = _FakeRun(returncode=1, stderr="Traceback …\n_tkinter.TclError: no display")
    monkeypatch.setattr(recordings.subprocess, "run", failed)

    with pytest.raises(FilePickerError, match="no display"):
        recordings.ask_for_recording(None)


def test_a_file_window_left_open_too_long_is_closed(monkeypatch):
    expired = subprocess.TimeoutExpired(cmd="python", timeout=600)
    monkeypatch.setattr(recordings.subprocess, "run", _FakeRun(raises=expired))

    with pytest.raises(FilePickerError, match="open too long"):
        recordings.ask_for_recording(None)
    # The next click can open a window again.
    monkeypatch.setattr(recordings.subprocess, "run", _FakeRun(stdout=""))
    assert recordings.ask_for_recording(None) is None


def test_only_one_file_window_is_open_at_a_time(monkeypatch):
    recordings._picker_open.acquire()
    try:
        with pytest.raises(FilePickerError, match="already open"):
            recordings.ask_for_recording(None)
    finally:
        recordings._picker_open.release()


# --- a recording's length ---------------------------------------------------------


@pytest.fixture
def probe(monkeypatch):
    """Answer ffprobe calls as told; the cache starts empty."""
    recordings._probe_duration.cache_clear()
    run = _FakeRun(stdout="2832.48\n")
    monkeypatch.setattr(recordings.subprocess, "run", run)
    yield run
    recordings._probe_duration.cache_clear()


def test_the_length_is_read_once_until_the_file_changes(tmp_path, probe):
    path = _file(tmp_path, "Call.mkv", b"first", mtime=1_000)

    assert recordings.duration(path) == 2832.48
    assert recordings.duration(path) == 2832.48
    assert len(probe.calls) == 1  # the list is drawn on every click
    [(args, kwargs)] = probe.calls
    assert args[0] == "ffprobe" and args[-1] == str(path)
    assert kwargs["timeout"] > 0

    _file(tmp_path, "Call.mkv", b"finished take", mtime=2_000)
    recordings.duration(path)

    assert len(probe.calls) == 2


def test_a_file_that_does_not_say_has_no_length(tmp_path, probe, monkeypatch):
    path = _file(tmp_path, "Call.mkv")
    # An MKV still being recorded: ffprobe prints N/A until it is finished.
    monkeypatch.setattr(recordings.subprocess, "run", _FakeRun(stdout="N/A\n"))
    assert recordings.duration(path) is None

    recordings._probe_duration.cache_clear()
    broken = _FakeRun(stdout="", returncode=1, stderr="moov atom not found")
    monkeypatch.setattr(recordings.subprocess, "run", broken)
    assert recordings.duration(path) is None

    assert recordings.duration(tmp_path / "gone.mkv") is None


def test_no_ffprobe_or_a_hanging_drive_gives_no_length(tmp_path, probe, monkeypatch):
    path = _file(tmp_path, "Call.mkv")
    missing = _FakeRun(raises=FileNotFoundError(2, "ffprobe not found"))
    monkeypatch.setattr(recordings.subprocess, "run", missing)
    assert recordings.duration(path) is None

    recordings._probe_duration.cache_clear()
    hung = _FakeRun(raises=subprocess.TimeoutExpired(cmd="ffprobe", timeout=15))
    monkeypatch.setattr(recordings.subprocess, "run", hung)
    assert recordings.duration(path) is None


def test_several_lengths_are_read_together(tmp_path, probe):
    paths = [_file(tmp_path, f"Call {n}.mkv", b"x" * n) for n in range(1, 4)]

    assert recordings.durations(paths) == dict.fromkeys(paths, 2832.48)
    assert recordings.durations([]) == {}
