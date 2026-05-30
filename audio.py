"""Audio preparation helpers.

Handles persisting uploads and converting any media file to the mono 16 kHz WAV
that the transcription pipeline expects. Consolidating the ffmpeg call here means
both the UI (audio preview/extraction) and the pipeline (normalisation before
splitting) share one implementation. This module is UI-agnostic — it never
imports Streamlit.
"""

from pathlib import Path
from typing import Any

import ffmpeg

from config import TARGET_CHANNELS, TARGET_SAMPLE_RATE, get_settings
from exceptions import AudioProcessingError
from logger import get_logger

logger = get_logger(__name__)


def _safe_name(filename: str) -> str:
    """Return just the file-name component, guarding against path traversal.

    Args:
        filename: The (possibly attacker-controlled) uploaded file name.

    Returns:
        The basename with any directory components stripped.
    """
    return Path(filename).name


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file to the configured uploads directory.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (anything exposing a ``name``
            attribute and a ``getbuffer()`` method).

    Returns:
        Path to the stored file.
    """
    destination = get_settings().upload_dir / _safe_name(uploaded_file.name)
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
