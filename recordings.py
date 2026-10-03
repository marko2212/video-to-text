"""Recordings already on this computer, picked from a folder instead of uploaded.

A browser upload passes through memory several times over: Streamlit receives
the whole request body, and Tornado copies it while parsing the form. On a
machine with little memory to spare, a 372.6 MB video failed that way with a
``MemoryError`` (2026-09-29). A file read straight from its folder needs no copy:
ffmpeg reads it where it lies, and nothing is duplicated into ``uploads/``.
A recording can also be found in the computer's own file window
(:func:`ask_for_recording`), since the app runs on the machine that has it.
This module is UI-agnostic — it never imports Streamlit.
"""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from config import (
    AUDIO_FORMATS,
    FILE_PICKER_TIMEOUT_SECONDS,
    RECORDING_PROBE_TIMEOUT_SECONDS,
    RECORDING_PROBE_WORKERS,
    VIDEO_FORMATS,
)
from exceptions import FilePickerError

_MEDIA = {f".{extension}" for extension in VIDEO_FORMATS + AUDIO_FORMATS}
# The file window runs in a process of its own, in that process's main thread:
# Tk must not be used from Streamlit's script threads (a Tk object collected in
# another thread aborts the whole process). Arguments: title, start folder,
# file patterns. It prints the chosen path, or nothing when cancelled.
_PICKER = """
import sys
import tkinter
from tkinter import filedialog

root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    parent=root,
    title=sys.argv[1],
    initialdir=sys.argv[2] or None,
    filetypes=[("Video and audio", sys.argv[3]), ("All files", "*.*")],
)
root.destroy()
sys.stdout.write(path or "")
"""
_PICKER_TITLE = "Choose a recording"
# One file window at a time for the whole app: a second click (or a second
# tab) while one is open would stack windows the owner may not even see.
_picker_open = threading.Lock()
# Characters that come around a pasted path: Explorer's "Copy as path" adds
# double quotes, and a path copied from a terminal may carry single ones.
_WRAPPING = "\"' \t"


def is_media(path: Path) -> bool:
    """Return True for a file name with a supported video or audio extension.

    Args:
        path: Any path.

    Returns:
        Whether the app accepts files of this type.
    """
    return path.suffix.lower() in _MEDIA


def parse_location(text: str) -> tuple[Path | None, Path | None]:
    """Read what was typed or pasted as the recordings folder.

    A path to one recording counts too: its folder is the one listed, with that
    recording chosen, so "Copy as path" on a file in Explorer is enough. Only
    a full path is accepted: a relative one (or a bare ``C:``) would be read
    against the app's own folder, whose ``temp/`` and ``uploads/`` hold its
    working copies.

    Args:
        text: The folder field's text.

    Returns:
        The folder and the pasted recording (``None`` when a folder was given);
        ``(None, None)`` when the text names no existing folder or recording.
    """
    cleaned = text.strip(_WRAPPING)
    if not cleaned:
        return None, None
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        return None, None
    try:
        if path.is_dir():
            return path, None
        if path.is_file() and is_media(path):
            return path.parent, path
    except OSError:
        pass
    return None, None


def _created(stat: os.stat_result) -> float:
    """Return when a file was created (Windows, macOS), else last modified."""
    return getattr(stat, "st_birthtime", stat.st_mtime)


def list_recordings(folder: Path) -> list[Path]:
    """Return the video and audio files directly in a folder, newest first.

    Ordered by creation time, which stays put while a file is still being
    written (a recording in progress), so the list does not reshuffle.

    Args:
        folder: The recordings folder.

    Returns:
        The files, most recently created first. Subfolders are not searched.

    Raises:
        OSError: If the folder cannot be read (no permission, a network drive
            that is gone).
    """
    found: list[tuple[float, Path]] = []
    for path in folder.iterdir():
        if not is_media(path):
            continue
        try:
            if path.is_file():
                found.append((_created(path.stat()), path))
        except OSError:
            continue  # removed or locked while listing
    return [path for _, path in sorted(found, key=lambda item: item[0], reverse=True)]


def find(listed: list[str], wanted: str) -> str | None:
    """Return a listed path as the listing spells it, ignoring case on Windows.

    A path typed or pasted by hand may differ in letter case from the one the
    folder listing returns, and the list keeps its choice by exact text.

    Args:
        listed: Paths from :func:`list_recordings`, as strings.
        wanted: The path to look for.

    Returns:
        The matching entry of ``listed``, or ``None``.
    """
    key = os.path.normcase(wanted)
    return next((path for path in listed if os.path.normcase(path) == key), None)


def label(path: Path) -> str:
    """Return a recording's entry in the list to pick from: name and date.

    Nothing that changes while a file is written (size, modification time) is
    in it: the list keeps the choice by its text. The page puts the length
    (:func:`duration`) in front, which appears only once a recording is
    finished, and handles that one change itself.

    Args:
        path: A recording.

    Returns:
        E.g. ``Call.mkv · 2026-09-29 16:05``; just the name when the file can
        no longer be read.
    """
    try:
        created = _created(path.stat())
    except OSError:
        return path.name
    return f"{path.name} · {datetime.fromtimestamp(created):%Y-%m-%d %H:%M}"


def summary(path: Path) -> str:
    """Return a chosen recording's size and when it was last saved.

    Args:
        path: A recording.

    Returns:
        E.g. ``372.6 MB · saved 2026-09-29 16:31``; empty when unreadable.
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    saved = datetime.fromtimestamp(stat.st_mtime)
    return f"{stat.st_size / (1024 * 1024):,.1f} MB · saved {saved:%Y-%m-%d %H:%M}"


def duration(path: Path) -> float | None:
    """Return how long a recording plays, as its file says.

    Read once per version of the file (size and modification time), so the
    list costs nothing on later reruns, and a file that changes is read again.

    Args:
        path: A recording.

    Returns:
        The length in seconds; ``None`` when the file does not say. An MKV that
        is still being recorded has no length until the recorder finishes it
        (ffprobe prints ``N/A``), an MP4 has none before its index is written,
        and nothing is known without ffprobe or for a file it cannot read.
    """
    try:
        size, modified = version(path)
    except OSError:
        return None
    return _probe_duration(str(path), size, modified)


@lru_cache(maxsize=1024)
def _probe_duration(path: str, size: int, modified: int) -> float | None:
    """Ask ffprobe for a file's length (``size`` and ``modified`` key the cache)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RECORDING_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        seconds = float(result.stdout.strip())
    except ValueError:
        return None  # "N/A", or nothing for a file ffprobe cannot read
    return seconds if seconds > 0 else None


def durations(paths: list[Path]) -> dict[Path, float | None]:
    """Return the lengths of several recordings, reading a few at a time.

    Args:
        paths: Recordings, e.g. from :func:`list_recordings`.

    Returns:
        Each path's :func:`duration`.
    """
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=RECORDING_PROBE_WORKERS) as pool:
        return dict(zip(paths, pool.map(duration, paths), strict=True))


def version(path: Path) -> tuple[int, int]:
    """Return what changes when a recording on disk changes: size and mtime.

    Kept apart from the file's identity (its path): a recording that is still
    being written must not silently throw away what was made from it, but the
    page should notice and offer to load it again.

    Args:
        path: A recording.

    Returns:
        ``(size in bytes, modification time in ns)``.

    Raises:
        OSError: If the file can no longer be read.
    """
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def ask_for_recording(start: Path | None) -> Path | None:
    """Let the owner choose a recording in the computer's own file window.

    Only meaningful where the app runs on the owner's own computer: the window
    opens on the machine that runs the app. It shows video and audio files
    and starts in ``start``; the page waits while it is open.

    Args:
        start: The folder to open in, if any.

    Returns:
        The chosen file, or ``None`` when the window was cancelled.

    Raises:
        FilePickerError: If a file window is already open, it cannot be shown
            (no desktop, no Tk), or nothing was chosen within
            ``FILE_PICKER_TIMEOUT_SECONDS``.
    """
    if not _picker_open.acquire(blocking=False):
        raise FilePickerError(
            "A file window is already open — it may be behind the browser."
        )
    patterns = " ".join(f"*.{extension}" for extension in VIDEO_FORMATS + AUDIO_FORMATS)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PICKER, _PICKER_TITLE, str(start or ""), patterns],
            capture_output=True,
            text=True,
            encoding="utf-8",
            # Paths with č, ć, š must survive the pipe on Windows.
            env={**os.environ, "PYTHONUTF8": "1"},
            timeout=FILE_PICKER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FilePickerError(
            "The file window was open too long and was closed — try again."
        ) from exc
    except OSError as exc:
        raise FilePickerError(f"The file window could not be opened: {exc}") from exc
    finally:
        _picker_open.release()
    if result.returncode != 0:
        last = (result.stderr or "").strip().splitlines()[-1:] or ["no details"]
        raise FilePickerError(f"The file window could not be opened: {last[0]}")
    chosen = result.stdout.strip()
    return Path(chosen) if chosen else None
