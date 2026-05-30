"""Transcription pipeline.

Splits an audio file into chunks, transcribes each chunk with the OpenAI audio
API, and writes a plain-text transcript plus an optional SRT subtitle file. This
module is UI-agnostic: progress is reported through an optional callback.
"""

import math
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydub import AudioSegment

from audio import to_wav
from config import (
    DEFAULT_MODEL,
    MAX_SEGMENT_SIZE_MB,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TIMESTAMP_MODELS,
    get_settings,
)
from exceptions import TranscriptionError
from logger import get_logger

logger = get_logger(__name__)

# Tuning constants (previously magic numbers scattered through the module).
_SEGMENT_BITRATE = "192k"
_MIN_SEGMENT_SIZE_MB = 0.1
_MIN_TINY_SEGMENT_SIZE_MB = 0.01
_MIN_DURATION_SECONDS = 5
_MAX_RETRIES = 3

ProgressCallback = Callable[[dict[str, Any]], None]


def _format_srt_timestamp(seconds: float) -> str:
    """Format a number of seconds as an SRT timestamp (``HH:MM:SS,mmm``).

    Args:
        seconds: Time offset in seconds.

    Returns:
        The SRT-formatted timestamp string.
    """
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(entries: list[dict[str, Any]]) -> str:
    """Build SRT subtitle text from timed entries.

    Args:
        entries: List of dicts with ``start`` and ``end`` (seconds) and ``text``.

    Returns:
        The full SRT document as a string.
    """
    blocks = []
    for index, entry in enumerate(entries, start=1):
        start = _format_srt_timestamp(entry["start"])
        end = _format_srt_timestamp(entry["end"])
        blocks.append(f"{index}\n{start} --> {end}\n{entry['text']}\n")
    return "\n".join(blocks)


def get_audio_info(file_path: Path, segment_duration_ms: int) -> dict[str, Any]:
    """Return basic information about an audio file.

    Args:
        file_path: Path to the audio file.
        segment_duration_ms: Chunk length in milliseconds (for the segment count).

    Returns:
        Dict with ``duration_minutes``, ``size_mb`` and ``total_segments``.
    """
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    return {
        "duration_minutes": duration_ms / 1000 / 60,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
        "total_segments": max(1, math.ceil(duration_ms / segment_duration_ms)),
    }


def _save_segment(segment: AudioSegment, index: int, temp_folder: Path) -> Path:
    """Export one chunk straight to a mono 16 kHz MP3 ready for the API.

    Args:
        segment: The pydub audio chunk.
        index: Zero-based chunk index (used for the file name).
        temp_folder: Directory to write the MP3 into.

    Returns:
        Path to the exported MP3.

    Raises:
        TranscriptionError: If the exported chunk is implausibly small.
    """
    mp3_path = temp_folder / f"segment_{index}.mp3"
    logger.info("Saving segment %d", index + 1)
    segment.export(
        mp3_path,
        format="mp3",
        bitrate=_SEGMENT_BITRATE,
        parameters=["-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE)],
    )

    size_mb = mp3_path.stat().st_size / (1024 * 1024)
    is_long_enough = len(segment) >= segment.frame_rate * _MIN_DURATION_SECONDS
    if is_long_enough and size_mb < _MIN_SEGMENT_SIZE_MB:
        raise TranscriptionError(f"Generated segment is too small: {size_mb:.2f} MB")
    return mp3_path


def _split_audio(
    input_file: Path, temp_folder: Path, segment_duration_ms: int
) -> list[AudioSegment]:
    """Normalise the input to WAV and slice it into fixed-length chunks.

    Args:
        input_file: Source audio/video file.
        temp_folder: Scratch directory for the intermediate WAV.
        segment_duration_ms: Chunk length in milliseconds.

    Returns:
        List of pydub audio chunks.
    """
    temp_wav = temp_folder / "temp_full.wav"
    try:
        to_wav(input_file, temp_wav)
        audio = AudioSegment.from_wav(temp_wav)
        logger.info(
            "Loaded audio: %d ch, %d Hz, %.1f s",
            audio.channels,
            audio.frame_rate,
            len(audio) / 1000,
        )
        return [
            audio[start : start + segment_duration_ms]
            for start in range(0, len(audio), segment_duration_ms)
        ]
    finally:
        temp_wav.unlink(missing_ok=True)


def _transcribe_segment(
    file_path: Path,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    with_timestamps: bool = False,
    retry_count: int = _MAX_RETRIES,
) -> Any:
    """Transcribe a single chunk, retrying transient failures.

    Args:
        file_path: Path to the chunk MP3.
        client: Configured OpenAI client.
        model: Transcription model name.
        with_timestamps: If True, request ``verbose_json`` (per-segment times).
        retry_count: Number of attempts before giving up.

    Returns:
        The transcript text, or — when ``with_timestamps`` is set — the full
        verbose response object (exposing ``.text`` and ``.segments``).

    Raises:
        TranscriptionError: If the file is missing or exceeds the size limit.
    """
    for attempt in range(retry_count):
        try:
            if not file_path.exists():
                raise TranscriptionError(f"File does not exist: {file_path}")

            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_SEGMENT_SIZE_MB:
                raise TranscriptionError(
                    f"File is too large ({size_mb:.2f} MB). "
                    f"Maximum is {MAX_SEGMENT_SIZE_MB:.0f} MB"
                )
            too_small = (
                duration_seconds >= _MIN_DURATION_SECONDS
                and size_mb < _MIN_SEGMENT_SIZE_MB
            ) or (
                duration_seconds < _MIN_DURATION_SECONDS
                and size_mb < _MIN_TINY_SEGMENT_SIZE_MB
            )
            if too_small:
                raise TranscriptionError(f"File is too small ({size_mb:.2f} MB)")

            logger.info("Transcribing segment (attempt %d)", attempt + 1)
            with file_path.open("rb") as audio_file:
                if with_timestamps:
                    return client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="verbose_json",
                    )
                return client.audio.transcriptions.create(
                    model=model, file=audio_file
                ).text
        except Exception as exc:  # retried below; re-raised on the final attempt
            logger.warning("Transcription attempt %d failed: %s", attempt + 1, exc)
            if attempt == retry_count - 1:
                raise TranscriptionError(str(exc)) from exc
            time.sleep(2**attempt)
    # Unreachable: the loop either returns or raises.
    raise TranscriptionError("Transcription failed unexpectedly")


def _report(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    """Invoke the progress callback if one was provided.

    Args:
        progress_callback: Optional callback receiving a status payload.
        **payload: Status fields (e.g. ``status``, ``message``, ``progress``).
    """
    if progress_callback:
        progress_callback(payload)


def _transcribe_all(
    segments: list[AudioSegment],
    client: OpenAI,
    model: str,
    with_timestamps: bool,
    temp_folder: Path,
    output_file: Path,
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    """Transcribe every chunk, appending text to the output and timing the SRT.

    Args:
        segments: Audio chunks to transcribe.
        client: Configured OpenAI client.
        model: Transcription model name.
        with_timestamps: Whether per-segment timestamps were requested.
        temp_folder: Scratch directory for chunk MP3s.
        output_file: Transcript file to append each chunk's text to.
        progress_callback: Optional progress callback.

    Returns:
        SRT entries (``start``/``end``/``text``) with timeline offsets applied.
    """
    srt_entries: list[dict[str, Any]] = []
    offset_seconds = 0.0
    total = len(segments)

    for index, segment in enumerate(segments):
        _report(
            progress_callback,
            status="progress",
            message=f"Processing segment {index + 1}/{total}",
            progress=(index + 1) / total,
        )
        segment_path = _save_segment(segment, index, temp_folder)
        try:
            result = _transcribe_segment(
                segment_path, client, model=model, with_timestamps=with_timestamps
            )
            if with_timestamps:
                text = result.text
                for seg in getattr(result, "segments", None) or []:
                    srt_entries.append(
                        {
                            "start": seg.start + offset_seconds,
                            "end": seg.end + offset_seconds,
                            "text": seg.text.strip(),
                        }
                    )
            else:
                text = result

            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n--- Segment {index + 1} ---\n{text}\n")
        finally:
            segment_path.unlink(missing_ok=True)

        offset_seconds += len(segment) / 1000.0

    return srt_entries


def process_audio(
    input_file: str | Path,
    output_file: str | Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    with_timestamps: bool = False,
    srt_output_file: str | Path | None = None,
    segment_duration_minutes: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Transcribe an audio/video file to text (and optionally subtitles).

    Args:
        input_file: Source audio or video file.
        output_file: Destination ``.txt`` transcript path.
        api_key: OpenAI API key.
        model: Transcription model name.
        with_timestamps: Request per-segment timestamps (only for capable models).
        srt_output_file: Destination ``.srt`` path; written only with timestamps.
        segment_duration_minutes: Length of each audio chunk in minutes.
        progress_callback: Optional callback receiving status payloads.

    Raises:
        TranscriptionError: If transcription fails at any stage.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    segment_duration_ms = segment_duration_minutes * 60 * 1000
    temp_folder = get_settings().temp_dir / "segments"
    client = OpenAI(api_key=api_key)

    # Timestamps/subtitles require a model that returns verbose_json.
    if with_timestamps and model not in TIMESTAMP_MODELS:
        with_timestamps = False

    try:
        temp_folder.mkdir(parents=True, exist_ok=True)

        info = get_audio_info(input_file, segment_duration_ms)
        _report(
            progress_callback,
            status="info",
            message=(
                f"Audio information:\n"
                f"Duration: {info['duration_minutes']:.2f} minutes\n"
                f"Size: {info['size_mb']:.2f} MB\n"
                f"Number of segments: {info['total_segments']}"
            ),
        )

        segments = _split_audio(input_file, temp_folder, segment_duration_ms)

        output_file.write_text(
            f"Transcription started: {datetime.now()}\n\n", encoding="utf-8"
        )
        _report(
            progress_callback,
            status="start",
            message=f"Transcription started at {datetime.now()}",
        )

        srt_entries = _transcribe_all(
            segments,
            client,
            model,
            with_timestamps,
            temp_folder,
            output_file,
            progress_callback,
        )

        if with_timestamps and srt_output_file and srt_entries:
            Path(srt_output_file).write_text(build_srt(srt_entries), encoding="utf-8")

        completion_time = datetime.now()
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\nTranscription completed: {completion_time}")

        _report(
            progress_callback,
            status="complete",
            message=f"Transcription completed at {completion_time}",
        )
        logger.info("Transcription saved to %s", output_file)

    except Exception as exc:
        _report(progress_callback, status="error", message=f"An error occurred: {exc}")
        logger.exception("Transcription failed")
        if isinstance(exc, TranscriptionError):
            raise
        raise TranscriptionError(str(exc)) from exc
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)
