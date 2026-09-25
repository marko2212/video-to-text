"""Audio preparation helpers.

Handles persisting uploads and converting any media file to the mono 16 kHz WAV
that the transcription pipeline expects. Consolidating the ffmpeg call here means
both the UI (audio preview/extraction) and the pipeline (normalisation before
splitting) share one implementation. This module is UI-agnostic — it never
imports Streamlit.
"""

import re
from pathlib import Path
from typing import Any

import ffmpeg

from config import PREVIEW_BITRATE, TARGET_CHANNELS, TARGET_SAMPLE_RATE, get_settings
from exceptions import AudioProcessingError
from logger import get_logger

logger = get_logger(__name__)


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


def to_wav(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert any media file to a mono 16 kHz PCM WAV using ffmpeg.

    Works for both video (extracts the audio track) and audio inputs, since
    ffmpeg reads any supported container.

    Args:
        input_path: Path to the source video/audio file.
        output_path: Destination WAV path. Defaults to ``<temp_dir>/<stem>.wav``.

    Returns:
        Path to the written WAV file.

    Raises:
        AudioProcessingError: If ffmpeg fails to decode or convert the input.
    """
    if output_path is None:
        output_path = get_settings().temp_dir / f"{input_path.stem}.wav"

    try:
        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(
            stream,
            str(output_path),
            acodec="pcm_s16le",
            ac=TARGET_CHANNELS,
            ar=TARGET_SAMPLE_RATE,
        )
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
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
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        logger.error("ffmpeg preview failed for %s: %s", input_path, detail)
        raise AudioProcessingError(
            f"Failed to make the audio preview: {detail}"
        ) from exc
    return output_path
