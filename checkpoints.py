"""On-disk checkpoints for paid work: transcribed chunks and described frames.

A run used to keep every result in memory, so one failed chunk — or a closed
tab — threw away everything already transcribed and billed, and the next
attempt paid for all of it again. Each result is now written to disk as soon as
it arrives, under a directory named after the content of the input file and the
settings that shaped the result, so a rerun of the same file picks up where the
last one stopped. This module is UI-agnostic — it never imports Streamlit.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import WORKING_COPY_TEMP_SUFFIXES, get_settings
from logger import get_logger

logger = get_logger(__name__)

_CHECKPOINT_DIRNAME = "checkpoints"
_READ_BLOCK_BYTES = 1024 * 1024
# Enough of the SHA-256 to make an accidental collision irrelevant, short enough
# to keep directory names readable.
_DIGEST_CHARS = 16
# Per-run scratch folders (chunk MP3s, screenshots) and video scans.
_SCRATCH_PATTERNS = ("segments-*", "frames-*", "scan-*")

# Runs in progress in this process, across every browser tab. Kept here, in an
# imported module, because Streamlit re-executes app.py from scratch each run.
_active_lock = threading.Lock()
_active_runs = 0


@contextmanager
def active_run() -> Iterator[None]:
    """Mark a transcription as running for the duration of the block.

    Yields:
        Nothing; the run counts as active until the block exits, however it
        exits (a toolbar Stop included).
    """
    global _active_runs
    with _active_lock:
        _active_runs += 1
    try:
        yield
    finally:
        with _active_lock:
            _active_runs -= 1


def any_active_run() -> bool:
    """Return True while any transcription is running in this process.

    "Clean temporary files" in one tab used to delete the checkpoints and
    scratch files of a run in another tab, which then paid for them again.
    """
    return _active_runs > 0


def file_digest(path: Path) -> str:
    """Return a short content hash of a file (so a renamed copy still matches).

    Args:
        path: File to hash.

    Returns:
        The first characters of the file's SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()[:_DIGEST_CHARS]


def checkpoint_root() -> Path:
    """Return the directory that holds every checkpoint (inside ``temp_dir``)."""
    return get_settings().temp_dir / _CHECKPOINT_DIRNAME


def run_dir(*parts: object) -> Path:
    """Return the checkpoint directory for one combination of inputs.

    Args:
        *parts: Everything the cached results depend on — the input's digest
            first, then e.g. the model and chunk length.

    Returns:
        The directory path (not created until something is saved).
    """
    name = "_".join(re.sub(r"[^A-Za-z0-9.-]+", "-", str(part)) for part in parts)
    return checkpoint_root() / name


def load(directory: Path, name: str) -> Any | None:
    """Return a saved result, or None when there is none (or it is unreadable).

    Args:
        directory: Checkpoint directory from :func:`run_dir`.
        name: Result name, e.g. ``"chunk_003"``.

    Returns:
        The decoded JSON value, or ``None``.
    """
    path = directory / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        # A half-written or corrupt file only costs a repeat of that one request.
        logger.warning("Ignoring unreadable checkpoint %s: %s", path, exc)
        return None


def save(directory: Path, name: str, data: Any) -> None:
    """Persist one result atomically, so a crash never leaves half a file.

    A failure to save (full disk, a locked file) is logged and otherwise
    ignored: the result is still in memory and the run can finish; only the
    ability to resume that one step is lost.

    Args:
        directory: Checkpoint directory from :func:`run_dir`.
        name: Result name, e.g. ``"chunk_003"``.
        data: A JSON-serialisable value.
    """
    target = directory / f"{name}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # A unique temporary name, so two runs saving the same result at once
        # cannot rename each other's file away.
        handle, partial = tempfile.mkstemp(
            prefix=f".{name}-", suffix=".tmp", dir=directory
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
        Path(partial).replace(target)
    except OSError as exc:
        logger.warning("Could not save checkpoint %s: %s", target, exc)


def count(directory: Path, prefix: str = "") -> int:
    """Return how many results are saved in a checkpoint directory.

    Args:
        directory: Checkpoint directory from :func:`run_dir`.
        prefix: Only count results whose name starts with this.

    Returns:
        The number of saved results (0 when the directory does not exist).
    """
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(f"{prefix}*.json"))


def prune(max_age_days: float) -> int:
    """Delete checkpoint directories untouched for longer than ``max_age_days``.

    They hold transcript text and screen descriptions of unfinished runs, which
    should not outlive their usefulness just because nobody pressed "Clean
    temporary files".

    Args:
        max_age_days: Age limit, by the newest file's modification time.

    Returns:
        How many directories were deleted.
    """
    root = checkpoint_root()
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86_400
    removed = 0
    for directory in root.iterdir():
        try:
            newest = max(
                (entry.stat().st_mtime for entry in directory.iterdir()),
                default=directory.stat().st_mtime,
            )
        except OSError:
            continue
        if newest < cutoff:
            discard(directory)
            removed += 1
    return removed


def prune_scratch(max_age_hours: float) -> int:
    """Delete per-run scratch folders left behind by a run that was killed.

    A run removes its own ``segments-*`` / ``frames-*`` folder when it ends, but
    not if the process dies (a closed console window, a restart). They hold
    audio chunks and screenshots, so they must not stay forever. Nothing is
    touched while a run is active, and the age limit is generous: a live frames
    folder is not written to while its screenshots are being described.

    Args:
        max_age_hours: Age limit, by the newest file's modification time.

    Returns:
        How many folders were deleted.
    """
    temp_dir = get_settings().temp_dir
    if any_active_run() or not temp_dir.is_dir():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for pattern in _SCRATCH_PATTERNS:
        for directory in temp_dir.glob(pattern):
            try:
                if not directory.is_dir():
                    continue
                newest = max(
                    (entry.stat().st_mtime for entry in directory.iterdir()),
                    default=directory.stat().st_mtime,
                )
            except OSError:
                continue
            if newest < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
    return removed


def prune_working_copies(max_age_hours: float) -> int:
    """Delete working copies of recordings that are older than ``max_age_hours``.

    An upload is stored in ``uploads/``, and a video's audio is extracted into
    ``temp/`` as a WAV, with a small MP3 for the player. Only the "Clean
    temporary files" button removed them. Deleted here: files directly in
    ``uploads/``, and WAV and MP3 files directly in ``temp/``. Left alone:
    transcripts, folders (saved parts of unfinished runs have their own
    14-day rule, scratch folders theirs), anything inside the history or model
    folder, and a file another program holds open. A page that still shows a
    deleted copy prepares it again from its source. Nothing is touched while a
    run is active in this process.

    Args:
        max_age_hours: Age limit, by the file's modification time.

    Returns:
        How many files were deleted.
    """
    if any_active_run():
        return 0
    settings = get_settings()
    cutoff = time.time() - max_age_hours * 3600
    keep = [settings.data_dir.resolve(), settings.whisper_model_dir.resolve()]
    removed = 0
    for folder, suffixes in (
        (settings.upload_dir, None),
        (settings.temp_dir, WORKING_COPY_TEMP_SUFFIXES),
    ):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                if not path.is_file():
                    continue
                if suffixes is not None and path.suffix.lower() not in suffixes:
                    continue
                resolved = path.resolve()
                if any(resolved.is_relative_to(kept) for kept in keep):
                    continue
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError as exc:
                # Held open elsewhere (a player, another app instance): next time.
                logger.warning("Could not remove %s: %s", path, exc)
    if removed:
        logger.info("Removed %d working copies older than %g h", removed, max_age_hours)
    return removed


def working_files_size() -> int:
    """Return how much space "Clean temporary files" would free.

    Everything in ``temp/`` and ``uploads/`` counts (uploads, extracted audio,
    transcripts, scans, saved parts of unfinished runs), except what the button
    leaves alone too: an entry that is, holds or lies inside the history or
    model folder. A file is counted once even if both settings name one folder.

    Returns:
        The size in bytes.
    """
    settings = get_settings()
    keep = [settings.data_dir.resolve(), settings.whisper_model_dir.resolve()]
    sizes: dict[Path, int] = {}
    for folder in (settings.temp_dir, settings.upload_dir):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                resolved = entry.resolve()
                if any(
                    resolved.is_relative_to(kept) or kept.is_relative_to(resolved)
                    for kept in keep
                ):
                    continue
                inside = list(resolved.rglob("*")) if resolved.is_dir() else [resolved]
            except OSError:
                continue
            for path in inside:
                try:
                    if path.is_file():
                        sizes[path] = path.stat().st_size
                except OSError:
                    continue  # removed or locked while counting
    return sum(sizes.values())


def discard(directory: Path) -> None:
    """Delete a checkpoint directory once its run has finished.

    Args:
        directory: Checkpoint directory from :func:`run_dir`.
    """
    shutil.rmtree(directory, ignore_errors=True)
