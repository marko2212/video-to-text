"""Transcription pipeline.

Transcribes audio with the OpenAI audio API (chunked, because of the 25 MB
request limit) or a local faster-whisper model. When the engine returns timed
segments, the transcript is rendered as timestamped paragraphs; otherwise it
falls back to sentence-grouped paragraphs. This module is UI-agnostic:
progress is reported through an optional callback.
"""

import math
import re
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

# Readability tuning for the rendered transcript: start a new paragraph after a
# pause this long, or once a paragraph grows past this many characters.
_PARAGRAPH_GAP_SECONDS = 2.0
_PARAGRAPH_MAX_CHARS = 350
_SENTENCES_PER_PARAGRAPH = 4

# Prefix marking a line that describes what was on screen rather than spoken.
_VISUAL_NOTE_MARKER = "🖥️"
_VISUAL_SECTION_HEADING = "On-screen context"

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


def _format_clock(seconds: float) -> str:
    """Format seconds as a short clock stamp: ``M:SS`` (or ``H:MM:SS``).

    Args:
        seconds: Time offset in seconds.

    Returns:
        The stamp shown inline in the transcript.
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_visual_note(note: dict[str, Any]) -> str:
    """Render one on-screen observation as a transcript line.

    Args:
        note: Dict with ``time`` (seconds) and ``description``.

    Returns:
        The note prefixed with a screen marker and its ``(M:SS)`` stamp.
    """
    description = str(note["description"]).strip()
    return f"{_VISUAL_NOTE_MARKER} ({_format_clock(note['time'])}) {description}"


def build_transcript(
    entries: list[dict[str, Any]],
    paragraph_gap: float = _PARAGRAPH_GAP_SECONDS,
    paragraph_chars: int = _PARAGRAPH_MAX_CHARS,
    visual_notes: list[dict[str, Any]] | None = None,
) -> str:
    """Render timed segments as a readable, timestamped transcript.

    Each segment is prefixed with its ``(M:SS)`` start time, and segments are
    grouped into paragraphs: a new paragraph begins after a pause longer than
    ``paragraph_gap`` or once the paragraph grows past ``paragraph_chars``.
    On-screen observations, when supplied, are interleaved by timestamp so a
    slide is described right after the speech it accompanies.

    Args:
        entries: Segments with ``start``/``end`` (seconds) and ``text``.
        paragraph_gap: Pause (seconds) that forces a paragraph break.
        paragraph_chars: Soft maximum characters per paragraph.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.

    Returns:
        The transcript as blank-line separated paragraphs.
    """
    notes = sorted(visual_notes or [], key=lambda note: note["time"])
    note_index = 0
    paragraphs: list[str] = []
    current: list[str] = []
    current_chars = 0
    previous_end: float | None = None

    for entry in entries:
        text = str(entry["text"]).strip()
        if not text:
            continue

        # Anything that appeared on screen before this line was spoken belongs
        # above it. The paragraph is closed first so the note stands alone —
        # otherwise a note landing mid-paragraph would be pushed past all of it.
        while note_index < len(notes) and notes[note_index]["time"] <= entry["start"]:
            if current:
                paragraphs.append(" ".join(current))
                current, current_chars = [], 0
            paragraphs.append(format_visual_note(notes[note_index]))
            note_index += 1

        gap = entry["start"] - previous_end if previous_end is not None else 0.0
        if current and (gap > paragraph_gap or current_chars >= paragraph_chars):
            paragraphs.append(" ".join(current))
            current, current_chars = [], 0
        current.append(f"({_format_clock(entry['start'])}) {text}")
        current_chars += len(text)
        previous_end = entry["end"]

    if current:
        paragraphs.append(" ".join(current))
    paragraphs += [format_visual_note(note) for note in notes[note_index:]]

    return "\n\n".join(paragraphs)


def split_into_paragraphs(
    text: str, sentences_per_paragraph: int = _SENTENCES_PER_PARAGRAPH
) -> str:
    """Group a flat transcript into paragraphs (used when there are no segments).

    Models such as ``gpt-4o-transcribe`` return one continuous string, so the
    text is split on sentence endings and regrouped to stay readable.

    Args:
        text: The flat transcript text.
        sentences_per_paragraph: How many sentences to keep per paragraph.

    Returns:
        The transcript as blank-line separated paragraphs.
    """
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    paragraphs = [
        " ".join(sentences[index : index + sentences_per_paragraph])
        for index in range(0, len(sentences), sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


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
    want_segments: bool = False,
    retry_count: int = _MAX_RETRIES,
) -> Any:
    """Transcribe a single chunk, retrying transient failures.

    Args:
        file_path: Path to the chunk MP3.
        client: Configured OpenAI client.
        model: Transcription model name.
        want_segments: If True, request ``verbose_json`` (per-segment times).
        retry_count: Number of attempts before giving up.

    Returns:
        The transcript text, or — when ``want_segments`` is set — the full
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
                if want_segments:
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


def _write_outputs(
    output_file: Path,
    texts: list[str],
    timed_entries: list[dict[str, Any]],
    srt_output_file: str | Path | None,
    visual_notes: list[dict[str, Any]] | None = None,
) -> None:
    """Write the rendered transcript and, when requested, the SRT file.

    Timed segments give a timestamped, paragraphed transcript; without them the
    flat text is at least regrouped into paragraphs so it stays readable.

    Args:
        output_file: Destination ``.txt`` path.
        texts: Raw per-chunk transcript texts (fallback source).
        timed_entries: Timed segments, if the engine returned any.
        srt_output_file: Destination ``.srt`` path, or None to skip subtitles.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.
    """
    if timed_entries:
        document = build_transcript(timed_entries, visual_notes=visual_notes)
    else:
        document = split_into_paragraphs(" ".join(text.strip() for text in texts))
        if visual_notes:
            # Without timed speech there is nothing to interleave against, so
            # the observations are listed as their own section instead.
            lines = [format_visual_note(note) for note in visual_notes]
            document += f"\n\n{_VISUAL_SECTION_HEADING}\n\n" + "\n\n".join(lines)
    output_file.write_text(document + "\n", encoding="utf-8")

    if srt_output_file and timed_entries:
        Path(srt_output_file).write_text(build_srt(timed_entries), encoding="utf-8")


def _transcribe_all(
    segments: list[AudioSegment],
    client: OpenAI,
    model: str,
    want_segments: bool,
    temp_folder: Path,
    progress_callback: ProgressCallback | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Transcribe every chunk and collect its text and timed segments.

    Args:
        segments: Audio chunks to transcribe.
        client: Configured OpenAI client.
        model: Transcription model name.
        want_segments: Whether the model returns per-segment timestamps.
        temp_folder: Scratch directory for chunk MP3s.
        progress_callback: Optional progress callback.

    Returns:
        Per-chunk texts and timed entries with timeline offsets applied.
    """
    texts: list[str] = []
    timed_entries: list[dict[str, Any]] = []
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
                segment_path, client, model=model, want_segments=want_segments
            )
            if want_segments:
                texts.append(result.text)
                # Offset each chunk's timestamps by its position in the original
                # audio so the timeline stays continuous across chunks.
                for seg in getattr(result, "segments", None) or []:
                    timed_entries.append(
                        {
                            "start": seg.start + offset_seconds,
                            "end": seg.end + offset_seconds,
                            "text": seg.text.strip(),
                        }
                    )
            else:
                texts.append(result)
        finally:
            segment_path.unlink(missing_ok=True)

        offset_seconds += len(segment) / 1000.0

    return texts, timed_entries


def transcribe_openai(
    input_file: str | Path,
    output_file: str | Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    with_timestamps: bool = False,
    srt_output_file: str | Path | None = None,
    segment_duration_minutes: int = 10,
    progress_callback: ProgressCallback | None = None,
    visual_notes: list[dict[str, Any]] | None = None,
) -> None:
    """Transcribe an audio/video file with the OpenAI API (segmented upload).

    Args:
        input_file: Source audio or video file.
        output_file: Destination ``.txt`` transcript path.
        api_key: OpenAI API key.
        model: Transcription model name.
        with_timestamps: Request per-segment timestamps (only for capable models).
        srt_output_file: Destination ``.srt`` path; written only with timestamps.
        segment_duration_minutes: Length of each audio chunk in minutes.
        progress_callback: Optional callback receiving status payloads.
        visual_notes: Optional on-screen notes to interleave into the transcript.

    Raises:
        TranscriptionError: If transcription fails at any stage.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    segment_duration_ms = segment_duration_minutes * 60 * 1000
    temp_folder = get_settings().temp_dir / "segments"
    client = OpenAI(api_key=api_key)

    # Only some models return verbose_json. Ask for segments whenever the model
    # supports them (free) so the transcript can be rendered with timestamps;
    # the SRT file itself is still written only when the user asked for it.
    want_segments = model in TIMESTAMP_MODELS
    if with_timestamps and not want_segments:
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

        _report(
            progress_callback,
            status="start",
            message=f"Transcription started at {datetime.now()}",
        )

        texts, timed_entries = _transcribe_all(
            segments,
            client,
            model,
            want_segments,
            temp_folder,
            progress_callback,
        )

        _write_outputs(
            output_file,
            texts,
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
        )

        _report(
            progress_callback,
            status="complete",
            message=f"Transcription completed at {datetime.now()}",
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


def transcribe_local(
    input_file: str | Path,
    output_file: str | Path,
    whisper_model: Any,
    with_timestamps: bool = False,
    srt_output_file: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    visual_notes: list[dict[str, Any]] | None = None,
) -> None:
    """Transcribe an audio/video file with a local faster-whisper model.

    Runs fully offline and needs no API key. Unlike the OpenAI path there is no
    25 MB limit, so the whole file is transcribed at once and per-segment
    timestamps come back natively (available for every local model).

    Args:
        input_file: Source audio or video file (ffmpeg-readable).
        output_file: Destination ``.txt`` transcript path.
        whisper_model: A loaded ``faster_whisper.WhisperModel`` instance.
        with_timestamps: Whether to also write an SRT subtitle file.
        srt_output_file: Destination ``.srt`` path; written only with timestamps.
        progress_callback: Optional callback receiving status payloads.
        visual_notes: Optional on-screen notes to interleave into the transcript.

    Raises:
        TranscriptionError: If transcription fails.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    try:
        _report(
            progress_callback,
            status="start",
            message=f"Local transcription started at {datetime.now()}",
        )
        # transcribe() returns a lazy generator; iterating it does the work.
        segments, info = whisper_model.transcribe(str(input_file), vad_filter=True)

        # faster-whisper always returns timed segments, so collect them even when
        # no SRT was requested — they drive the readable transcript.
        timed_entries: list[dict[str, Any]] = []
        texts: list[str] = []
        duration = getattr(info, "duration", 0) or 0
        for segment in segments:
            texts.append(segment.text)
            timed_entries.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )
            if progress_callback and duration:
                _report(
                    progress_callback,
                    status="progress",
                    message=f"Transcribing… {segment.end:.0f}/{duration:.0f}s",
                    progress=min(1.0, segment.end / duration),
                )

        _write_outputs(
            output_file,
            texts,
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
        )

        _report(
            progress_callback,
            status="complete",
            message=f"Transcription completed at {datetime.now()}",
        )
        logger.info("Local transcription saved to %s", output_file)
    except Exception as exc:
        _report(progress_callback, status="error", message=f"An error occurred: {exc}")
        logger.exception("Local transcription failed")
        raise TranscriptionError(str(exc)) from exc
