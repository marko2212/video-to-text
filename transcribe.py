"""Transcription pipeline.

Transcribes audio with the OpenAI audio API (in chunks, because of the 25 MB
request limit and the gpt-4o output cap) or a local faster-whisper model. When
the engine returns timed segments, the transcript is rendered as timestamped
paragraphs; otherwise paragraphs get approximate times derived from where each
chunk sits in the recording. Every finished chunk is checkpointed on disk, so a
failed or interrupted run resumes instead of paying again. This module is
UI-agnostic: progress is reported through an optional callback.
"""

import re
import shutil
import tempfile
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from pydub import AudioSegment

import checkpoints
import openai_api
import serbian
import usage
from audio import to_wav
from config import (
    CUT_SEARCH_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_SEGMENT_MINUTES,
    MAX_SEGMENT_SIZE_MB,
    MIN_SPLIT_SECONDS,
    MIN_TAIL_SECONDS,
    OUTPUT_TOKEN_CAP_GUARD,
    SEGMENT_MINUTES_BY_MODEL,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TIMESTAMP_MODELS,
    get_settings,
)
from exceptions import AppError, IncompleteTranscriptionError, TranscriptionError
from logger import get_logger

logger = get_logger(__name__)

# Tuning constants (previously magic numbers scattered through the module).
_SEGMENT_BITRATE = "192k"
# Loudness is compared over windows this long when looking for a pause to cut at.
_CUT_WINDOW_MS = 250
# Bump when chunk boundaries or the saved format change, so old checkpoints are
# ignored rather than stitched to differently cut neighbours.
_CHECKPOINT_VERSION = "v1"

# Readability tuning for the rendered transcript: start a new paragraph after a
# pause this long, or once a paragraph grows past this many characters.
_PARAGRAPH_GAP_SECONDS = 2.0
_PARAGRAPH_MAX_CHARS = 350
_SENTENCES_PER_PARAGRAPH = 4

# Prefix marking a line that describes what was on screen rather than spoken.
_VISUAL_NOTE_MARKER = "🖥️"
# Prefix of the lines that mark missing speech: where a partial transcript
# stops, or where an answer still ended at the output cap.
_MISSING_MARKER = "⚠️"

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


def format_missing_marker(start: float, end: float, reason: str) -> str:
    """Render the line that marks where a partial transcript stops.

    Args:
        start: Where the untranscribed part begins (seconds).
        end: Where the recording ends (seconds).
        reason: Why the run stopped.

    Returns:
        A line that cannot be mistaken for speech.
    """
    return (
        f"{_MISSING_MARKER} ({_format_clock(start)}–{_format_clock(end)}) "
        f"Not transcribed: {reason} Press Start again to finish — the parts "
        "above are saved and will not be sent again."
    )


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


def _group_sentences(
    text: str, sentences_per_paragraph: int = _SENTENCES_PER_PARAGRAPH
) -> list[str]:
    """Split flat text on sentence endings and regroup it into paragraphs.

    Args:
        text: Flat transcript text.
        sentences_per_paragraph: How many sentences to keep per paragraph.

    Returns:
        The paragraphs, in order.
    """
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    return [
        " ".join(sentences[index : index + sentences_per_paragraph])
        for index in range(0, len(sentences), sentences_per_paragraph)
    ]


def split_into_paragraphs(
    text: str, sentences_per_paragraph: int = _SENTENCES_PER_PARAGRAPH
) -> str:
    """Group a flat transcript into paragraphs.

    Args:
        text: The flat transcript text.
        sentences_per_paragraph: How many sentences to keep per paragraph.

    Returns:
        The transcript as blank-line separated paragraphs.
    """
    return "\n\n".join(_group_sentences(text, sentences_per_paragraph))


def build_untimed_transcript(
    pieces: list[dict[str, Any]],
    visual_notes: list[dict[str, Any]] | None = None,
    missing: tuple[float, float, str] | None = None,
) -> str:
    """Render chunk texts without timestamps as paragraphs with approximate times.

    Models such as ``gpt-4o-transcribe`` return one string per chunk, but where
    each chunk sits in the recording is known. Each paragraph is stamped
    ``(~M:SS)`` by its share of its chunk's characters — approximate, because it
    assumes an even speaking rate — which is enough to place on-screen notes
    next to the speech they accompany instead of in a block at the end.

    Args:
        pieces: Chunks with ``start`` and ``duration`` (seconds) and ``text``.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.
        missing: ``(start, end, reason)`` of an untranscribed tail, if any.

    Returns:
        The transcript as blank-line separated paragraphs.
    """
    notes = sorted(visual_notes or [], key=lambda note: note["time"])
    lines: list[str] = []

    def notes_until(limit: float) -> None:
        while notes and notes[0]["time"] <= limit:
            lines.append(format_visual_note(notes.pop(0)))

    for piece in pieces:
        paragraphs = _group_sentences(str(piece["text"]))
        total_chars = sum(len(paragraph) for paragraph in paragraphs) or 1
        chars_before = 0
        for paragraph in paragraphs:
            share = chars_before / total_chars
            start = piece["start"] + piece["duration"] * share
            notes_until(start)
            lines.append(f"(~{_format_clock(start)}) {paragraph}")
            chars_before += len(paragraph)
        if piece.get("capped"):
            end = piece["start"] + piece["duration"]
            lines.append(
                f"{_MISSING_MARKER} (~{_format_clock(end)}) The model stopped writing "
                "at its length limit here, so the end of this part may be missing."
            )

    if missing:
        notes_until(missing[0])
        lines.append(format_missing_marker(*missing))
    lines += [format_visual_note(note) for note in notes]
    return "\n\n".join(lines)


def segment_minutes(model: str) -> int:
    """Return the chunk length used for a model.

    Args:
        model: Transcription model name.

    Returns:
        ``SEGMENT_DURATION_MINUTES`` from the environment when set, otherwise
        the per-model default from :mod:`config`.
    """
    override = get_settings().segment_duration_minutes
    if override:
        return override
    return SEGMENT_MINUTES_BY_MODEL.get(model, DEFAULT_SEGMENT_MINUTES)


def checkpoint_dir(audio_digest: str, model: str) -> Path:
    """Return where finished chunks of one recording and model are kept.

    Args:
        audio_digest: :func:`checkpoints.file_digest` of the audio file.
        model: Transcription model name.

    Returns:
        The checkpoint directory (created on first save).
    """
    minutes = segment_minutes(model)
    return checkpoints.run_dir(
        audio_digest, "transcribe", model, f"{minutes}min", _CHECKPOINT_VERSION
    )


def saved_chunk_count(audio_digest: str, model: str) -> int:
    """Return how many chunks of an unfinished run are already on disk.

    Args:
        audio_digest: :func:`checkpoints.file_digest` of the audio file.
        model: Transcription model name.

    Returns:
        The number of checkpointed chunks (0 when nothing is saved).
    """
    return checkpoints.count(checkpoint_dir(audio_digest, model), "chunk_")


def _quietest_cut(audio: AudioSegment, target_ms: int, search_ms: int) -> int:
    """Return the quietest moment near ``target_ms`` — a pause to cut at.

    Args:
        audio: The whole recording.
        target_ms: Nominal cut position in milliseconds.
        search_ms: How far either side of it to look.

    Returns:
        The centre of the quietest window, preferring the one nearest the
        target on a tie (e.g. digital silence), or ``target_ms`` itself.
    """
    low = max(0, target_ms - search_ms)
    high = min(len(audio) - _CUT_WINDOW_MS, target_ms + search_ms)
    if high <= low:
        return target_ms

    def loudness(start: int) -> tuple[int, int]:
        centre = start + _CUT_WINDOW_MS // 2
        return audio[start : start + _CUT_WINDOW_MS].rms, abs(centre - target_ms)

    best = min(range(low, high, _CUT_WINDOW_MS), key=loudness)
    return best + _CUT_WINDOW_MS // 2


def _chunk_bounds(audio: AudioSegment, chunk_ms: int) -> list[tuple[int, int]]:
    """Plan chunk boundaries: about ``chunk_ms`` long, cut in pauses.

    Args:
        audio: The whole recording.
        chunk_ms: Nominal chunk length in milliseconds.

    Returns:
        ``(start_ms, end_ms)`` pairs covering the recording without gaps. A
        final chunk shorter than ``MIN_TAIL_SECONDS`` is merged into the one
        before it.
    """
    total = len(audio)
    search_ms = int(CUT_SEARCH_SECONDS * 1000)
    cuts = [0]
    while cuts[-1] + chunk_ms < total:
        cuts.append(_quietest_cut(audio, cuts[-1] + chunk_ms, search_ms))
    cuts.append(total)
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < MIN_TAIL_SECONDS * 1000:
        del cuts[-2]
    return list(pairwise(cuts))


def _load_audio(input_file: Path, temp_folder: Path) -> AudioSegment:
    """Normalise the input to mono 16 kHz WAV and load it.

    Args:
        input_file: Source audio/video file.
        temp_folder: Scratch directory for the intermediate WAV.

    Returns:
        The decoded recording.
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
        return audio
    finally:
        temp_wav.unlink(missing_ok=True)


def _export_chunk(segment: AudioSegment, temp_folder: Path, name: str) -> Path:
    """Export one chunk to a mono 16 kHz MP3 and check it fits the API limit.

    Args:
        segment: The audio chunk.
        temp_folder: Directory to write the MP3 into.
        name: File stem, unique within the run.

    Returns:
        Path to the exported MP3.

    Raises:
        TranscriptionError: If the chunk exceeds the request size limit.
    """
    mp3_path = temp_folder / f"{name}.mp3"
    segment.export(
        mp3_path,
        format="mp3",
        bitrate=_SEGMENT_BITRATE,
        parameters=["-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE)],
    )
    size_mb = mp3_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SEGMENT_SIZE_MB:
        mp3_path.unlink(missing_ok=True)
        raise TranscriptionError(
            f"A chunk is too large ({size_mb:.1f} MB; the API accepts "
            f"{MAX_SEGMENT_SIZE_MB:.0f} MB) — lower SEGMENT_DURATION_MINUTES."
        )
    return mp3_path


def _request(file_path: Path, client: OpenAI, model: str, want_segments: bool) -> Any:
    """Send one chunk to the API (the SDK retries transient failures itself).

    Args:
        file_path: Path to the chunk MP3.
        client: Client from :func:`openai_api.make_client`.
        model: Transcription model name.
        want_segments: If True, request ``verbose_json`` (per-segment times).

    Returns:
        The API response (exposing ``.text``, and ``.segments`` when asked).

    Raises:
        OpenAIAccountError: If the key or the account is refused.
        TranscriptionError: For any other failure, with a short message.
    """
    try:
        with file_path.open("rb") as audio_file:
            if want_segments:
                return client.audio.transcriptions.create(
                    model=model, file=audio_file, response_format="verbose_json"
                )
            return client.audio.transcriptions.create(model=model, file=audio_file)
    except OpenAIError as exc:
        raise openai_api.translate_error(exc, TranscriptionError) from exc


def _output_tokens(result: Any) -> int | None:
    """Return how many tokens the model wrote, when the response says so.

    Args:
        result: A transcription response.

    Returns:
        ``usage.output_tokens``, or ``None`` if the response carries no usage.
    """
    usage = getattr(result, "usage", None)
    if isinstance(usage, dict):
        value = usage.get("output_tokens")
    else:
        value = getattr(usage, "output_tokens", None)
    return value if isinstance(value, int) else None


def _transcribe_span(
    audio: AudioSegment,
    span: tuple[int, int],
    client: OpenAI,
    model: str,
    want_segments: bool,
    work: tuple[Path, Path],
    name: str,
    may_split: bool = True,
) -> list[dict[str, Any]]:
    """Transcribe one stretch of audio, splitting it once if the answer was cut off.

    Every answer is checkpointed as soon as it arrives, including the halves of
    a split stretch, so a failure in one half never throws the other away.

    Args:
        audio: The whole recording.
        span: ``(start_ms, end_ms)`` of the stretch.
        client: Configured OpenAI client.
        model: Transcription model name.
        want_segments: Whether the model returns per-segment timestamps.
        work: Scratch directory for the chunk MP3, and the checkpoint directory.
        name: Name of this stretch, unique within the run (``"000"``, ``"000a"``).
        may_split: False for the halves of a split stretch: an answer that fills
            the cap again is a loop or noise, and splitting further only
            multiplies the cost.

    Returns:
        Pieces with ``start`` and ``duration`` (seconds) and ``text``, plus
        ``segments`` on the recording's timeline when ``want_segments`` is set,
        ``capped`` when the answer may still be cut off, and ``usage``: the
        records of the requests paid for it (see :mod:`usage`).
    """
    start_ms, end_ms = span
    temp_folder, checkpoint = work
    part = f"part_{name}"
    saved = checkpoints.load(checkpoint, part)
    if saved and saved.get("bounds") == [start_ms, end_ms]:
        if "pieces" in saved:
            return saved["pieces"]
        middle = saved["split"]
        # A marker saved before costs were recorded has no usage: the capped
        # request was still paid for, so it counts as an estimate, not as free.
        spent = saved.get("usage") or [
            usage.transcription_record(None, model, (end_ms - start_ms) / 1000)
        ]
    else:
        path = _export_chunk(audio[start_ms:end_ms], temp_folder, f"chunk_{name}")
        try:
            result = _request(path, client, model, want_segments)
        finally:
            path.unlink(missing_ok=True)

        tokens = _output_tokens(result)
        span_ms = end_ms - start_ms
        spent = [usage.transcription_record(result, model, span_ms / 1000)]
        capped = tokens is not None and tokens >= OUTPUT_TOKEN_CAP_GUARD
        if not (capped and may_split and span_ms >= 2 * MIN_SPLIT_SECONDS * 1000):
            if capped:
                logger.warning(
                    "Part %s still hit the output cap (%d tokens)", name, tokens
                )
            piece = _piece(result, start_ms, end_ms, want_segments, capped)
            piece["usage"] = spent
            checkpoints.save(
                checkpoint, part, {"bounds": [start_ms, end_ms], "pieces": [piece]}
            )
            return [piece]

        # The answer stopped at the output cap, so the end of this stretch is
        # missing from it. Transcribe the two halves instead.
        logger.warning(
            "Part %s hit the output cap (%d tokens); splitting it", name, tokens
        )
        middle = _quietest_cut(
            audio,
            (start_ms + end_ms) // 2,
            min(span_ms // 4, int(CUT_SEARCH_SECONDS * 1000)),
        )
        # The capped answer was paid for too; it is kept with the split marker
        # so a resumed run still counts it.
        checkpoints.save(
            checkpoint,
            part,
            {"bounds": [start_ms, end_ms], "split": middle, "usage": spent},
        )

    halves = ((start_ms, middle, f"{name}a"), (middle, end_ms, f"{name}b"))
    pieces = [
        piece
        for low, high, half in halves
        for piece in _transcribe_span(
            audio, (low, high), client, model, want_segments, work, half, False
        )
    ]
    if pieces:
        first = pieces[0]
        own = _usage_records([first], model)
        pieces[0] = {**first, "usage": spent + own}
    return pieces


def _piece(
    result: Any, start_ms: int, end_ms: int, want_segments: bool, capped: bool
) -> dict[str, Any]:
    """Turn one API answer into a piece placed on the recording's timeline.

    Args:
        result: The transcription response.
        start_ms: Where the transcribed stretch starts.
        end_ms: Where it ends.
        want_segments: Whether the response carries timed segments.
        capped: Whether the answer ended at the output cap.

    Returns:
        A JSON-serialisable piece (it is checkpointed as is).
    """
    offset = start_ms / 1000
    piece: dict[str, Any] = {
        "start": offset,
        "duration": (end_ms - start_ms) / 1000,
        "text": str(result.text),
    }
    if capped:
        piece["capped"] = True
    if want_segments:
        # Offset each segment by the chunk's position in the original audio so
        # the timeline stays continuous across chunks.
        piece["segments"] = [
            {
                "start": seg.start + offset,
                "end": seg.end + offset,
                "text": seg.text.strip(),
            }
            for seg in getattr(result, "segments", None) or []
        ]
    return piece


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
    pieces: list[dict[str, Any]],
    timed_entries: list[dict[str, Any]],
    srt_output_file: str | Path | None,
    visual_notes: list[dict[str, Any]] | None = None,
    missing: tuple[float, float, str] | None = None,
) -> None:
    """Write the rendered transcript and, when requested, the SRT file.

    Timed segments give a timestamped, paragraphed transcript; without them each
    chunk's text is paragraphed and stamped with approximate times. Serbian
    Cyrillic is rewritten in Latin unless ``SERBIAN_LATIN`` is off.

    Args:
        output_file: Destination ``.txt`` path.
        pieces: Per-chunk texts with their place in the recording.
        timed_entries: Timed segments, if the engine returned any.
        srt_output_file: Destination ``.srt`` path, or None to skip subtitles.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.
        missing: ``(start, end, reason)`` of an untranscribed tail, if any.
    """
    if timed_entries:
        notes = visual_notes or []
        if missing:
            # Notes from the untranscribed stretch belong below its marker.
            before = [note for note in notes if note["time"] <= missing[0]]
            after = sorted(
                (note for note in notes if note["time"] > missing[0]),
                key=lambda note: note["time"],
            )
            document = build_transcript(timed_entries, visual_notes=before)
            document = "\n\n".join(
                [document, format_missing_marker(*missing)]
                + [format_visual_note(note) for note in after]
            )
        else:
            document = build_transcript(timed_entries, visual_notes=notes)
    else:
        document = build_untimed_transcript(pieces, visual_notes, missing)

    srt = build_srt(timed_entries) if srt_output_file and timed_entries else None
    if get_settings().serbian_latin and serbian.is_serbian_cyrillic(document):
        document = serbian.cyrillic_to_latin(document)
        srt = serbian.cyrillic_to_latin(srt) if srt else srt

    output_file.write_text(document + "\n", encoding="utf-8")
    if srt_output_file and srt is not None:
        Path(srt_output_file).write_text(srt, encoding="utf-8")


def _transcribe_all(
    audio: AudioSegment,
    bounds: list[tuple[int, int]],
    client: OpenAI,
    model: str,
    want_segments: bool,
    work: tuple[Path, Path],
    progress_callback: ProgressCallback | None,
) -> tuple[list[dict[str, Any]], tuple[int, AppError] | None]:
    """Transcribe every chunk in order, reusing and saving checkpoints.

    Args:
        audio: The whole recording.
        bounds: ``(start_ms, end_ms)`` of each chunk.
        client: Configured OpenAI client.
        model: Transcription model name.
        want_segments: Whether the model returns per-segment timestamps.
        work: Scratch directory for chunk MP3s, and the checkpoint directory.
        progress_callback: Optional progress callback.

    Returns:
        The pieces transcribed so far, and ``(chunk index, error)`` if a chunk
        failed — the run stops there, since the next chunk would most likely
        fail the same way.
    """
    checkpoint = work[1]
    pieces: list[dict[str, Any]] = []
    total = len(bounds)
    for index, (start_ms, end_ms) in enumerate(bounds):
        name = f"chunk_{index:03d}"
        saved = checkpoints.load(checkpoint, name)
        if saved and saved.get("bounds") == [start_ms, end_ms]:
            pieces += saved["pieces"]
            continue
        _report(
            progress_callback,
            status="progress",
            message=f"Transcribing part {index + 1} of {total}…",
            progress=index / total,
        )
        try:
            new = _transcribe_span(
                audio,
                (start_ms, end_ms),
                client,
                model,
                want_segments,
                work,
                f"{index:03d}",
            )
        except AppError as exc:
            return pieces, (index, exc)
        except Exception as exc:
            # Anything else (a failed MP3 export, say) must still leave the
            # partial transcript and the saved chunks behind.
            logger.exception("Chunk %d failed", index + 1)
            return pieces, (index, TranscriptionError(f"Chunk {index + 1}: {exc}"))
        # One file per finished chunk, so counting them tells how far a run got.
        checkpoints.save(
            checkpoint, name, {"bounds": [start_ms, end_ms], "pieces": new}
        )
        pieces += new
    return pieces, None


def _usage_records(pieces: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Collect the usage records behind a list of pieces.

    Args:
        pieces: Transcribed pieces, some possibly restored from checkpoints.
        model: Transcription model name.

    Returns:
        Every record; a piece saved before cost recording existed gets a
        per-minute estimate, so it counts as roughly paid rather than free.
    """
    records: list[dict[str, Any]] = []
    for piece in pieces:
        if "usage" in piece:
            records += piece["usage"]
        else:
            records.append(usage.transcription_record(None, model, piece["duration"]))
    return records


def _unfinished_chunk_usage(
    checkpoint: Path, index: int, model: str
) -> list[dict[str, Any]]:
    """Collect what the failed chunk already paid for (a split and its halves).

    Args:
        checkpoint: The checkpoint directory.
        index: The chunk that failed.
        model: Transcription model name, for estimating parts saved before
            costs were recorded.

    Returns:
        Usage records of its saved split marker and finished halves.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(checkpoint.glob(f"part_{index:03d}*.json")):
        saved = checkpoints.load(checkpoint, path.stem) or {}
        if "split" in saved:
            start_ms, end_ms = saved.get("bounds", (0, 0))
            records += saved.get("usage") or [
                usage.transcription_record(None, model, (end_ms - start_ms) / 1000)
            ]
        records += _usage_records(saved.get("pieces", []), model)
    return records


def _reusable_chunks(checkpoint: Path, bounds: list[tuple[int, int]]) -> int:
    """Count saved chunks that match the planned boundaries and will be reused.

    Args:
        checkpoint: The checkpoint directory.
        bounds: ``(start_ms, end_ms)`` of each chunk.

    Returns:
        How many chunks need no request.
    """
    return sum(
        1
        for index, (start_ms, end_ms) in enumerate(bounds)
        if (saved := checkpoints.load(checkpoint, f"chunk_{index:03d}"))
        and saved.get("bounds") == [start_ms, end_ms]
    )


def transcribe_openai(
    input_file: str | Path,
    output_file: str | Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    with_timestamps: bool = False,
    srt_output_file: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    visual_notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Transcribe an audio/video file with the OpenAI API, chunk by chunk.

    Args:
        input_file: Source audio or video file.
        output_file: Destination ``.txt`` transcript path.
        api_key: OpenAI API key.
        model: Transcription model name.
        with_timestamps: Request per-segment timestamps (only for capable models).
        srt_output_file: Destination ``.srt`` path; written only with timestamps.
        progress_callback: Optional callback receiving status payloads.
        visual_notes: Optional on-screen notes to place into the transcript.

    Returns:
        What the requests used and cost (:func:`usage.summarize`), counting
        chunks paid for by an earlier, unfinished attempt too.

    Raises:
        OpenAIAccountError: If the key or account is refused before any chunk
            was transcribed.
        IncompleteTranscriptionError: If the run stopped after some chunks; a
            partial transcript has been written and the chunks are saved.
        TranscriptionError: If transcription fails at any other stage.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    started = time.monotonic()

    # Only some models return verbose_json. Ask for segments whenever the model
    # supports them (free) so the transcript can be rendered with timestamps;
    # the SRT file itself is still written only when the user asked for it.
    want_segments = model in TIMESTAMP_MODELS
    if with_timestamps and not want_segments:
        with_timestamps = False

    # A folder of its own per run: two browser tabs used to share temp/segments,
    # so one run deleted or overwrote the other's chunks.
    temp_dir = get_settings().temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_folder = Path(tempfile.mkdtemp(prefix="segments-", dir=temp_dir))
    client: OpenAI | None = None
    try:
        checkpoint = checkpoint_dir(checkpoints.file_digest(input_file), model)
        audio = _load_audio(input_file, temp_folder)
        bounds = _chunk_bounds(audio, segment_minutes(model) * 60_000)
        restored = _reusable_chunks(checkpoint, bounds)
        resume = f" — {restored} already done, resuming" if restored else ""
        _report(
            progress_callback,
            status="info",
            message=(
                f"{_format_clock(len(audio) / 1000)} of audio in "
                f"{len(bounds)} part(s){resume}"
            ),
        )

        client = openai_api.make_client(api_key)
        pieces, failure = _transcribe_all(
            audio,
            bounds,
            client,
            model,
            want_segments,
            (temp_folder, checkpoint),
            progress_callback,
        )
        timed_entries = [
            segment for piece in pieces for segment in piece.get("segments", [])
        ]

        if failure:
            index, cause = failure
            if index == 0:
                # Nothing to show, but halves of a split first chunk may have
                # been paid for; the caller says so.
                leftover = _unfinished_chunk_usage(checkpoint, 0, model)
                if leftover:
                    cause.spent = usage.summarize(leftover)
                raise cause
            spent = {
                **usage.summarize(
                    _usage_records(pieces, model)
                    + _unfinished_chunk_usage(checkpoint, index, model)
                ),
                # The length the partial transcript covers, not the audio sent
                # (a split chunk is sent more than once).
                "audio_seconds": round(bounds[index][0] / 1000, 3),
            }
            # Keep what was paid for: the finished chunks stay checkpointed, and
            # a partial transcript says plainly where it stops and why.
            _write_outputs(
                output_file,
                pieces,
                timed_entries,
                None,
                visual_notes=visual_notes,
                missing=(bounds[index][0] / 1000, len(audio) / 1000, str(cause)),
            )
            raise IncompleteTranscriptionError(
                str(cause), completed=index, total=len(bounds), spent=spent
            ) from cause

        _write_outputs(
            output_file,
            pieces,
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
        )
        # The checkpoints stay until the caller has stored the result (see
        # discard_checkpoints): a run stopped right after this point must be
        # able to finish without paying again.

        _report(
            progress_callback,
            status="complete",
            message=(
                "Transcription completed in "
                f"{_format_clock(time.monotonic() - started)}"
            ),
        )
        logger.info("Transcription saved to %s", output_file)
        # `seconds` in the summary is audio *sent*, which a split chunk exceeds;
        # the recording's own length is reported alongside it.
        return {
            **usage.summarize(_usage_records(pieces, model)),
            "audio_seconds": round(len(audio) / 1000, 3),
        }

    except AppError as exc:
        _report(progress_callback, status="error", message=str(exc))
        logger.warning("Transcription stopped: %s", exc)
        raise
    except Exception as exc:
        _report(progress_callback, status="error", message=f"An error occurred: {exc}")
        logger.exception("Transcription failed")
        raise TranscriptionError(str(exc)) from exc
    finally:
        if client is not None:
            client.close()
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
    started = time.monotonic()
    try:
        _report(
            progress_callback, status="start", message="Local transcription started…"
        )
        # transcribe() returns a lazy generator; iterating it does the work.
        segments, info = whisper_model.transcribe(str(input_file), vad_filter=True)

        # faster-whisper always returns timed segments, so collect them even when
        # no SRT was requested — they drive the readable transcript.
        timed_entries: list[dict[str, Any]] = []
        duration = getattr(info, "duration", 0) or 0
        for segment in segments:
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
            [],
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
        )

        _report(
            progress_callback,
            status="complete",
            message=(
                "Transcription completed in "
                f"{_format_clock(time.monotonic() - started)}"
            ),
        )
        logger.info("Local transcription saved to %s", output_file)
    except Exception as exc:
        _report(progress_callback, status="error", message=f"An error occurred: {exc}")
        logger.exception("Local transcription failed")
        raise TranscriptionError(str(exc)) from exc
