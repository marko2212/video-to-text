"""Audio preparation helpers.

Handles persisting uploads and converting any media file to the mono 16 kHz WAV
that the transcription pipeline expects. Consolidating the ffmpeg call here means
both the UI (audio preview/extraction) and the pipeline (normalisation before
splitting) share one implementation. This module is UI-agnostic — it never
imports Streamlit.
"""

import re
import shutil
from pathlib import Path
from typing import Any

import ffmpeg

from config import (
    FFMPEG_TOOLS,
    FFMPEG_WINDOWS_INSTALL,
    PREVIEW_BITRATE,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    get_settings,
)
from exceptions import AudioProcessingError, FfmpegMissingError
from logger import get_logger

logger = get_logger(__name__)


def missing_tools() -> list[str]:
    """Return the ffmpeg programs this process cannot find on its PATH.

    A process keeps the PATH it started with, so a program installed while the
    app runs stays missing here until the app is started again.

    Returns:
        The names from :data:`config.FFMPEG_TOOLS` that are not found, in order.
    """
    return [tool for tool in FFMPEG_TOOLS if shutil.which(tool) is None]


def missing_tools_message(missing: list[str]) -> str:
    """Say which programs are missing and how to install them, for the page.

    Args:
        missing: Program names, as :func:`missing_tools` returns them.

    Returns:
        A Markdown sentence or three naming the programs, the Windows install
        command and the restart the app needs afterwards.
    """
    names = " and ".join(f"**{name}**" for name in missing)
    verb = "was" if len(missing) == 1 else "were"
    return (
        f"{names} {verb} not found on this computer, and the app needs ffmpeg "
        f"and ffprobe to read recordings. Install them — on Windows: "
        f"`{FFMPEG_WINDOWS_INSTALL}` — then close the app and start it again "
        "(README → Installation → Install ffmpeg)."
    )


def _run(stream: Any) -> None:
    """Run an ffmpeg command, saying so plainly when ffmpeg is not installed.

    Args:
        stream: The ffmpeg-python output stream to run.

    Raises:
        FfmpegMissingError: If the ffmpeg program cannot be started at all.
        ffmpeg.Error: If ffmpeg ran and failed (its stderr is captured).
    """
    try:
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
    except FileNotFoundError as exc:
        # Raised by subprocess before ffmpeg starts; a missing input file is
        # reported by ffmpeg itself, as ffmpeg.Error.
        logger.error("ffmpeg could not be started: %s", exc)
        raise FfmpegMissingError(missing_tools_message(["ffmpeg"])) from exc


def safe_name(filename: str) -> str:
    """Return a file name that cannot point outside the folder it is joined to.

    The basename alone is not enough on Windows: ``x/D:evil.mp4`` has the
    basename ``D:evil.mp4``, which a join resolves against drive D:.

    Args:
        filename: The (possibly attacker-controlled) uploaded file name.

    Returns:
        The basename without drive or path separators (``"upload"`` if empty).
    """
    name = re.split(r"[\\/]", filename)[-1]
    name = name.replace(":", "_").strip(" .")
    return name or "upload"


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file to the configured uploads directory.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (anything exposing a ``name``
            attribute and a ``getbuffer()`` method).

    Returns:
        Path to the stored file.
    """
    destination = get_settings().upload_dir / safe_name(uploaded_file.name)
    destination.write_bytes(uploaded_file.getbuffer())
    logger.info("Saved upload to %s", destination)
    return destination


def to_wav(
    input_path: Path,
    output_path: Path | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Convert any media file to a mono 16 kHz PCM WAV using ffmpeg.

    Works for both video (extracts the audio track) and audio inputs, since
    ffmpeg reads any supported container.

    Args:
        input_path: Path to the source video/audio file.
        output_path: Destination WAV path. Defaults to ``<temp_dir>/<stem>.wav``.
        start: Seconds to skip at the beginning, if only part of the file is
            wanted (exact in a WAV, within a frame or so in an MP3).
        duration: How many seconds to keep from there; the rest of the file
            when ``None``.

    Returns:
        Path to the written WAV file.

    Raises:
        FfmpegMissingError: If ffmpeg is not installed.
        AudioProcessingError: If ffmpeg fails to decode or convert the input.
    """
    if output_path is None:
        output_path = get_settings().temp_dir / f"{input_path.stem}.wav"

    stretch: dict[str, str] = {}
    if start:
        stretch["ss"] = f"{start:.3f}"
    if duration is not None:
        stretch["t"] = f"{duration:.3f}"
    try:
        stream = ffmpeg.input(str(input_path), **stretch)
        stream = ffmpeg.output(
            stream,
            str(output_path),
            acodec="pcm_s16le",
            ac=TARGET_CHANNELS,
            ar=TARGET_SAMPLE_RATE,
        )
        _run(stream)
    except ffmpeg.Error as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        logger.error("ffmpeg failed for %s: %s", input_path, detail)
        raise AudioProcessingError(f"Failed to extract audio: {detail}") from exc

    logger.info("Converted %s -> %s", input_path, output_path)
    return output_path


def to_preview(input_path: Path, output_path: Path) -> Path:
    """Encode a small mono MP3 for the in-browser player.

    The player used to get the raw upload or the full WAV: browsers cannot play
    AMR (every phone call), and a 90-minute WAV (~160 MB) was re-read and
    re-hashed on every rerun, costing about half a second per click.

    Args:
        input_path: Any ffmpeg-readable audio or video file.
        output_path: Destination ``.mp3`` path.

    Returns:
        Path to the written MP3.

    Raises:
        FfmpegMissingError: If ffmpeg is not installed.
        AudioProcessingError: If ffmpeg fails to decode or encode the input.
    """
    try:
        stream = ffmpeg.output(
            ffmpeg.input(str(input_path)),
            str(output_path),
            acodec="libmp3lame",
            audio_bitrate=PREVIEW_BITRATE,
            # LAME's fastest mode: ~5 s instead of ~9 s for 90 minutes, and the
            # quality it trades away does not matter for checking speech.
            compression_level=9,
            ac=TARGET_CHANNELS,
            ar=TARGET_SAMPLE_RATE,
            vn=None,
        )
        _run(stream)
    except ffmpeg.Error as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        logger.error("ffmpeg preview failed for %s: %s", input_path, detail)
        raise AudioProcessingError(
            f"Failed to make the audio preview: {detail}"
        ) from exc
    return output_path
