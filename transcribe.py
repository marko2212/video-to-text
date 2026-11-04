"""Transcription pipeline.

Transcribes audio with the OpenAI audio API (in chunks, because of the 25 MB
request limit and the gpt-4o output cap) or a local faster-whisper model. When
the engine returns timed segments, the transcript is rendered as timestamped
paragraphs; otherwise paragraphs get approximate times derived from where each
chunk sits in the recording. Every finished chunk is checkpointed on disk, so a
failed or interrupted run resumes instead of paying again. This module is
UI-agnostic: progress is reported through an optional callback.
"""

import os
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from pydub import AudioSegment

import align
import checkpoints
import openai_api
import serbian
import usage
from audio import to_wav
from config import (
    CUT_SEARCH_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_SEGMENT_MINUTES,
    LOCAL_BATCH_SIZE,
    MAX_SEGMENT_SIZE_MB,
    MIN_SPLIT_SECONDS,
    MIN_TAIL_SECONDS,
    OUTPUT_TOKEN_CAP_GUARD,
    SEGMENT_MINUTES_BY_MODEL,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    THIN_CHUNK_MIN_MEDIAN_WORDS_PER_MINUTE,
    THIN_CHUNK_SHARE,
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
# A batched local piece (up to 30 s) is split back into sentence-sized lines by
# its word times: after a word that ends a sentence, before a word that follows
# a pause this long, and before a line would run longer than this.
_BATCHED_SPLIT_PAUSE_SECONDS = 1.0
_BATCHED_MAX_LINE_SECONDS = 10.0

# Prefix marking a line that describes what was on screen rather than spoken.
_VISUAL_NOTE_MARKER = "🖥️"
# Prefix of the lines that mark missing speech: where a partial transcript
# stops, or where an answer still ended at the output cap.
_MISSING_MARKER = "⚠️"
# Prefix of the line that says only part of the recording was transcribed, on
# purpose (the start or the end was left out with the slider).
_SPAN_MARKER = "⏭️"
# Prefix of a paragraph the local Whisper pass supplied where the API answer
# lacked speech, and of the header line that says the transcript was checked.
_LOCAL_FILL_MARKER = "🧩"
_VERIFIED_MARKER = "🔍"
# A stretch shorter than this is nothing to transcribe.
_MIN_SPAN_MS = 1000

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
        note: Dict with ``time`` (seconds) and ``description`` — or, for a
            marker line placed by time among the notes, a ready ``line``.

    Returns:
        The note prefixed with a screen marker and its ``(M:SS)`` stamp.
    """
    if "line" in note:
        return str(note["line"])
    description = str(note["description"]).strip()
    return f"{_VISUAL_NOTE_MARKER} ({_format_clock(note['time'])}) {description}"


def format_thin_marker(
    start: float, end: float, rate: float, typical: float | None
) -> str:
    """Render the line that marks a part whose answer had far too few words.

    Args:
        start: Where the part begins (seconds).
        end: Where it ends (seconds).
        rate: Words per minute the answer had.
        typical: Words per minute of the run's other parts, when known.

    Returns:
        A line that cannot be mistaken for speech.
    """
    against = f" against about {typical:.0f} in the rest" if typical else ""
    return (
        f"{_MISSING_MARKER} ({_format_clock(start)}–{_format_clock(end)}) This "
        f"part came back with far fewer words than expected ({rate:.0f} a "
        f"minute{against}), so speech here may be missing."
    )


def format_capped_marker(end: float) -> str:
    """Render the line that marks a part whose answer ended at the output cap.

    Args:
        end: Where the part ends (seconds).

    Returns:
        A line that cannot be mistaken for speech.
    """
    return (
        f"{_MISSING_MARKER} (~{_format_clock(end)}) The model stopped writing "
        "at its length limit here, so the end of this part may be missing."
    )


def _flag_lines(piece: dict[str, Any]) -> list[str]:
    """Return the marker lines a piece carries (output cap, thin answer).

    Args:
        piece: A transcribed piece with ``start`` and ``duration``.

    Returns:
        Zero, one or two lines, to go right after the piece's text.
    """
    end = piece["start"] + piece["duration"]
    lines = []
    if piece.get("capped"):
        lines.append(format_capped_marker(end))
    if piece.get("thin"):
        record = piece.get("record") or {}
        lines.append(
            format_thin_marker(
                piece["start"],
                end,
                record.get("words_per_min", 0.0),
                record.get("typical_words_per_min"),
            )
        )
    return lines


def format_verify_note(stats: dict[str, Any]) -> str:
    """Render the header line of a transcript checked against a local pass.

    Args:
        stats: The cross-check's figures (see :func:`align.cross_check`), plus
            the local ``model`` name.

    Returns:
        One line for the top of the transcript.
    """
    total, exact = stats["paragraphs"], stats["anchored"]
    rest = total - exact
    kept = f"; the other {rest} keep an estimate, marked ~" if rest else ""
    added = (
        f"; {stats['gaps']} stretch(es) with {stats['added_words']} words that "
        f"the OpenAI answer lacked were added from it, marked {_LOCAL_FILL_MARKER}"
        if stats["gaps"]
        else "; nothing the OpenAI answer lacked was found"
    )
    return (
        f"{_VERIFIED_MARKER} Checked against a local Whisper pass "
        f"({stats['model']}): {exact} of {total} paragraphs carry the "
        f"recording's own times{kept}{added}."
    )


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


def format_span_note(start: float, end: float) -> str:
    """Render the line that says which part of the recording was transcribed.

    Args:
        start: Where the transcribed stretch begins (seconds).
        end: Where it ends (seconds).

    Returns:
        One line for the top of the transcript.
    """
    return (
        f"{_SPAN_MARKER} Transcribed {_format_clock(start)}–{_format_clock(end)} "
        "of the recording; the rest was left out."
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

    for index, piece in enumerate(pieces):
        for paragraph in _untimed_paragraphs([piece], index):
            notes_until(paragraph["start"])
            lines.append(f"(~{_format_clock(paragraph['start'])}) {paragraph['text']}")
        lines += _flag_lines(piece)

    if missing:
        notes_until(missing[0])
        lines.append(format_missing_marker(*missing))
    lines += [format_visual_note(note) for note in notes]
    return "\n\n".join(lines)


def _untimed_paragraphs(
    pieces: list[dict[str, Any]], first_index: int = 0
) -> list[dict[str, Any]]:
    """Paragraph the chunk texts and estimate where each paragraph starts.

    Args:
        pieces: Chunks with ``start`` and ``duration`` (seconds) and ``text``.
        first_index: The index of the first piece, for the ``piece`` field.

    Returns:
        Paragraphs with ``start`` (seconds, by character share within the
        chunk), ``text`` and ``piece`` (the index of the chunk they came from).
    """
    paragraphs: list[dict[str, Any]] = []
    for offset, piece in enumerate(pieces):
        texts = _group_sentences(str(piece["text"]))
        total_chars = sum(len(text) for text in texts) or 1
        chars_before = 0
        for text in texts:
            share = chars_before / total_chars
            paragraphs.append(
                {
                    "start": piece["start"] + piece["duration"] * share,
                    "text": text,
                    "piece": first_index + offset,
                }
            )
            chars_before += len(text)
    return paragraphs


def build_verified_transcript(
    pieces: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    model: str,
    visual_notes: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render chunk texts re-timed by, and filled in from, a local Whisper pass.

    The API's paragraphs are aligned word by word with the local pass's timed
    segments (:func:`align.cross_check`): a paragraph whose words the local
    pass also has takes the recording's own time, written ``(M:SS)``; one
    without such a run keeps its estimate, shifted like its nearest re-timed
    neighbour and still marked ``(~M:SS)``; and a stretch the local pass has
    but the API answer lacks is added where it belongs, marked as the local
    pass's text. The chunk markers (output cap, thin answer) and the on-screen
    notes are placed by time among them.

    Args:
        pieces: Chunks with ``start`` and ``duration`` (seconds) and ``text``.
        entries: The local pass's segments with ``start``, ``end`` and ``text``
            on the recording's timeline.
        model: The local model's name, for the header line.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.

    Returns:
        The transcript as blank-line separated paragraphs, and the
        cross-check's figures (``model`` added).
    """
    paragraphs = _untimed_paragraphs(pieces)
    checked = align.cross_check(paragraphs, entries)
    stats = {"model": model, **checked["stats"]}
    notes = [dict(note) for note in visual_notes or []]
    for piece in pieces:
        end = piece["start"] + piece["duration"]
        notes += [{"time": end, "line": line} for line in _flag_lines(piece)]
    notes.sort(key=lambda note: note["time"])

    lines = [format_verify_note(stats)]

    def notes_until(limit: float) -> None:
        while notes and notes[0]["time"] <= limit:
            lines.append(format_visual_note(notes.pop(0)))

    for item in align.merge(checked["paragraphs"], checked["gaps"]):
        notes_until(item["start"])
        stamp = _format_clock(item["start"])
        if item["kind"] == "gap":
            lines.append(f"{_LOCAL_FILL_MARKER} ({stamp}) {item['text']}")
        elif item["exact"]:
            lines.append(f"({stamp}) {item['text']}")
        else:
            lines.append(f"(~{stamp}) {item['text']}")
    lines += [format_visual_note(note) for note in notes]
    return "\n\n".join(lines), stats


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


def checkpoint_dir(
    audio_digest: str, model: str, span: tuple[float, float] | None = None
) -> Path:
    """Return where finished chunks of one recording and model are kept.

    Args:
        audio_digest: :func:`checkpoints.file_digest` of the audio file.
        model: Transcription model name.
        span: Where the transcribed stretch starts and ends (seconds), when
            only part of the recording is transcribed.

    Returns:
        The checkpoint directory (created on first save).
    """
    minutes = segment_minutes(model)
    parts: list[object] = [
        audio_digest,
        "transcribe",
        model,
        f"{minutes}min",
        _CHECKPOINT_VERSION,
    ]
    if span is not None:
        # A folder per stretch: chunks are named by index, so a stretch's run
        # would otherwise overwrite — and, when it finished, discard — the
        # whole recording's saved parts. Parts whose boundaries match are still
        # reused across folders (see _saved_elsewhere).
        start, end = span
        parts.append(f"{start:g}-{end:g}s")
    return checkpoints.run_dir(*parts)


def _saved_elsewhere(
    base: Path, own: Path
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Collect the chunks saved for other stretches of the same recording.

    A stretch that starts where the whole recording does plans the same
    leading chunks, so a run stopped by an empty credit and started again
    with the tail left out must not pay for them twice.

    Args:
        base: The whole recording's checkpoint directory for this model.
        own: The directory of the run being planned (skipped).

    Returns:
        The pieces of every finished chunk, by its ``(start_ms, end_ms)``.
    """
    found: dict[tuple[int, int], list[dict[str, Any]]] = {}
    if not base.parent.is_dir():
        return found
    for folder in sorted(base.parent.glob(f"{base.name}*")):
        if folder == own or not folder.is_dir():
            continue
        for path in folder.glob("chunk_*.json"):
            saved = checkpoints.load(folder, path.stem) or {}
            bounds, pieces = saved.get("bounds"), saved.get("pieces")
            if bounds and pieces is not None:
                found.setdefault((bounds[0], bounds[1]), pieces)
    return found


def saved_parts_within(audio_digest: str, model: str, span: tuple[float, float]) -> int:
    """Count parts saved for other stretches that lie within this one.

    They are reused only if the plan's boundaries match them exactly, which
    this cannot know without the audio — so the hint it serves promises no
    more than "those that fit".

    Args:
        audio_digest: :func:`checkpoints.file_digest` of the audio file.
        model: Transcription model name.
        span: The stretch chosen now, in seconds.

    Returns:
        How many saved chunks lie within the stretch.
    """
    base = checkpoint_dir(audio_digest, model)
    start_ms, end_ms = round(span[0] * 1000), round(span[1] * 1000)
    return sum(
        1
        for start, end in _saved_elsewhere(
            base, checkpoint_dir(audio_digest, model, span)
        )
        if start_ms <= start and end <= end_ms
    )


def saved_chunk_count(
    audio_digest: str, model: str, span: tuple[float, float] | None = None
) -> int:
    """Return how many chunks of an unfinished run are already on disk.

    Args:
        audio_digest: :func:`checkpoints.file_digest` of the audio file.
        model: Transcription model name.
        span: The stretch of the recording the run is for, if not all of it.

    Returns:
        The number of checkpointed chunks (0 when nothing is saved).
    """
    return checkpoints.count(checkpoint_dir(audio_digest, model, span), "chunk_")


def _span_ms(span: tuple[float, float] | None, total_ms: int) -> tuple[int, int] | None:
    """Place the chosen stretch on the recording, in milliseconds.

    Args:
        span: Where the stretch starts and ends, in seconds, or ``None`` for
            the whole recording.
        total_ms: The recording's length.

    Returns:
        ``(start_ms, end_ms)`` within the recording, or ``None``.

    Raises:
        TranscriptionError: If less than a second of the recording is left.
    """
    if span is None:
        return None
    start = min(max(0, round(span[0] * 1000)), total_ms)
    end = min(max(start, round(span[1] * 1000)), total_ms)
    if end - start < _MIN_SPAN_MS:
        raise TranscriptionError(
            f"Nothing to transcribe between {_format_clock(span[0])} and "
            f"{_format_clock(span[1])} — move the ends of the slider apart."
        )
    return start, end


def _quietest_cut(
    audio: AudioSegment, target_ms: int, search_ms: int, limit_ms: int | None = None
) -> int:
    """Return the quietest moment near ``target_ms`` — a pause to cut at.

    Args:
        audio: The whole recording.
        target_ms: Nominal cut position in milliseconds.
        search_ms: How far either side of it to look.
        limit_ms: Where the stretch being cut ends; the recording's end by
            default.

    Returns:
        The centre of the quietest window, preferring the one nearest the
        target on a tie (e.g. digital silence), or ``target_ms`` itself.
    """
    if limit_ms is None:
        limit_ms = len(audio)
    low = max(0, target_ms - search_ms)
    high = min(limit_ms - _CUT_WINDOW_MS, target_ms + search_ms)
    if high <= low:
        return target_ms

    def loudness(start: int) -> tuple[int, int]:
        centre = start + _CUT_WINDOW_MS // 2
        return audio[start : start + _CUT_WINDOW_MS].rms, abs(centre - target_ms)

    best = min(range(low, high, _CUT_WINDOW_MS), key=loudness)
    return best + _CUT_WINDOW_MS // 2


def _chunk_bounds(
    audio: AudioSegment, chunk_ms: int, span_ms: tuple[int, int] | None = None
) -> list[tuple[int, int]]:
    """Plan chunk boundaries: about ``chunk_ms`` long, cut in pauses.

    Args:
        audio: The whole recording.
        chunk_ms: Nominal chunk length in milliseconds.
        span_ms: The stretch to cover, from :func:`_span_ms`; the whole
            recording when ``None``.

    Returns:
        ``(start_ms, end_ms)`` pairs covering the stretch without gaps. A
        final chunk shorter than ``MIN_TAIL_SECONDS`` is merged into the one
        before it.
    """
    start, total = span_ms if span_ms is not None else (0, len(audio))
    search_ms = int(CUT_SEARCH_SECONDS * 1000)
    cuts = [start]
    while cuts[-1] + chunk_ms < total:
        cuts.append(_quietest_cut(audio, cuts[-1] + chunk_ms, search_ms, total))
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


def _request_id(result: Any) -> str | None:
    """Return the OpenAI request id of a response, when the SDK attached one.

    Args:
        result: A transcription response.

    Returns:
        The ``x-request-id`` the SDK stores on parsed answers, or ``None``.
    """
    value = getattr(result, "_request_id", None)
    return value if isinstance(value, str) and value else None


def words_per_minute(text: str, seconds: float) -> float:
    """Return how many words a text has per minute of the audio it came from.

    Args:
        text: The transcribed text.
        seconds: Length of the audio it transcribes.

    Returns:
        Words per minute (0 for no audio).
    """
    if seconds <= 0:
        return 0.0
    return len(text.split()) / (seconds / 60)


def _thin_floor(pieces: list[dict[str, Any]]) -> float | None:
    """Return the words-per-minute rate below which the next answer counts as thin.

    The floor is a share of the running median of the pieces answered so far
    (``THIN_CHUNK_SHARE``); pieces already flagged thin do not vote, and a run
    whose typical rate is too low to judge by (silence, music) gets no floor.

    Args:
        pieces: The pieces transcribed so far in this run.

    Returns:
        The floor, or ``None`` while there is nothing to compare with.
    """
    rates = [
        words_per_minute(str(piece["text"]), piece["duration"])
        for piece in pieces
        if not piece.get("thin")
    ]
    if not rates:
        return None
    median = statistics.median(rates)
    if median < THIN_CHUNK_MIN_MEDIAN_WORDS_PER_MINUTE:
        return None
    return THIN_CHUNK_SHARE * median


def _chunk_record(
    name: str,
    span: tuple[int, int],
    result: Any,
    tokens: int | None,
    seconds: float,
    size_bytes: int,
    thin_floor: float | None,
) -> dict[str, Any]:
    """Describe one answered chunk, for the log and the history row.

    Args:
        name: The part's name (``"003"``, ``"003a"``).
        span: ``(start_ms, end_ms)`` of the part.
        result: The transcription response.
        tokens: Output tokens the answer reported, if any.
        seconds: How long the request took.
        size_bytes: Size of the MP3 that was sent.
        thin_floor: The words-per-minute floor in force (see :func:`_thin_floor`).

    Returns:
        A JSON-serialisable record with ``part``, ``start``, ``end``,
        ``mp3_bytes``, ``seconds``, ``output_tokens``, ``chars``, ``words``,
        ``words_per_min``, ``request_id``, ``thin`` and ``capped``, plus
        ``typical_words_per_min`` when a floor was in force.
    """
    start_ms, end_ms = span
    text = str(result.text)
    rate = words_per_minute(text, (end_ms - start_ms) / 1000)
    record: dict[str, Any] = {
        "part": name,
        "start": start_ms / 1000,
        "end": end_ms / 1000,
        "mp3_bytes": size_bytes,
        "seconds": round(seconds, 2),
        "output_tokens": tokens,
        "chars": len(text),
        "words": len(text.split()),
        "words_per_min": round(rate, 1),
        "request_id": _request_id(result),
        "thin": thin_floor is not None and rate < thin_floor,
        "capped": tokens is not None and tokens >= OUTPUT_TOKEN_CAP_GUARD,
    }
    if thin_floor is not None:
        record["typical_words_per_min"] = round(thin_floor / THIN_CHUNK_SHARE)
    return record


def _log_record(record: dict[str, Any]) -> None:
    """Write the one line per answered part that the application log keeps.

    Args:
        record: A record from :func:`_chunk_record`.
    """
    flags = ""
    if record["capped"]:
        flags += ", capped"
    if record["thin"]:
        flags += f", thin (typical {record['typical_words_per_min']} words/min)"
    tokens = record["output_tokens"]
    logger.info(
        "Part %s (%s–%s, %.2f MB) answered in %.1f s: %s output tokens, %d chars, "
        "%d words, %.0f words/min%s, request %s",
        record["part"],
        _format_clock(record["start"]),
        _format_clock(record["end"]),
        record["mp3_bytes"] / (1024 * 1024),
        record["seconds"],
        f"{tokens:,}" if tokens is not None else "unknown",
        record["chars"],
        record["words"],
        record["words_per_min"],
        flags,
        record["request_id"] or "id unknown",
    )


def chunk_records(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect the per-chunk records of a run, for its history row.

    Args:
        pieces: The run's pieces, some possibly restored from checkpoints.

    Returns:
        One record per piece, in order, each with an ``index`` (the chunk's
        number in the plan). A piece saved before records existed gets one
        made from its text and usage, without a request id.
    """
    records = []
    for position, piece in enumerate(pieces):
        record = piece.get("record")
        if record is None:
            text = str(piece["text"])
            spent = piece.get("usage") or []
            tokens = [r.get("output_tokens") for r in spent]
            record = {
                "part": None,
                "start": piece["start"],
                "end": piece["start"] + piece["duration"],
                "output_tokens": sum(t for t in tokens if t) if any(tokens) else None,
                "chars": len(text),
                "words": len(text.split()),
                "words_per_min": round(words_per_minute(text, piece["duration"]), 1),
                "request_id": None,
                "thin": bool(piece.get("thin")),
                "capped": bool(piece.get("capped")),
            }
        part = record.get("part") or ""
        digits = part[:3] if part[:3].isdigit() else ""
        records.append({"index": int(digits) if digits else position, **record})
    return records


def describe_flagged(chunks: list[dict[str, Any]]) -> str | None:
    """Phrase which parts of a run may have lost speech, for a warning.

    Args:
        chunks: Records from :func:`chunk_records` (as saved with the row).

    Returns:
        One sentence naming each flagged part, or ``None`` when none is.
    """
    phrases = []
    for record in chunks:
        if not (record.get("thin") or record.get("capped")):
            continue
        where = f"{_format_clock(record['start'])}–{_format_clock(record['end'])}"
        why = []
        if record.get("thin"):
            typical = record.get("typical_words_per_min")
            against = f" against about {typical:.0f}" if typical else ""
            why.append(f"{record.get('words_per_min', 0):.0f} words/min{against}")
        if record.get("capped"):
            why.append("stopped at the output cap")
        phrases.append(f"{where} ({', '.join(why)})")
    if not phrases:
        return None
    return (
        f"Speech may be missing in {len(phrases)} part(s) — {'; '.join(phrases)}. "
        "The transcript marks them ⚠️."
    )


def _transcribe_span(
    audio: AudioSegment,
    span: tuple[int, int],
    client: OpenAI,
    model: str,
    want_segments: bool,
    work: tuple[Path, Path],
    name: str,
    may_split: bool = True,
    thin_floor: float | None = None,
) -> list[dict[str, Any]]:
    """Transcribe one stretch of audio, splitting it once if the answer is incomplete.

    An answer counts as incomplete when it ends at the output cap, or when it has far
    fewer words per minute than the run's other answers (the API has returned
    part of a chunk with no error). Every answer is checkpointed as soon as it
    arrives, including the halves of a split stretch, so a failure in one half
    never throws the other away.

    Args:
        audio: The whole recording.
        span: ``(start_ms, end_ms)`` of the stretch.
        client: Configured OpenAI client.
        model: Transcription model name.
        want_segments: Whether the model returns per-segment timestamps.
        work: Scratch directory for the chunk MP3, and the checkpoint directory.
        name: Name of this stretch, unique within the run (``"000"``, ``"000a"``).
        may_split: False for the halves of a split stretch: an answer that fills
            the cap again is a loop or noise, and one that is still thin is
            kept and marked, since splitting further only multiplies the cost.
        thin_floor: Words per minute below which the answer counts as thin
            (:func:`_thin_floor`), or ``None`` when nothing can be compared yet.

    Returns:
        Pieces with ``start`` and ``duration`` (seconds) and ``text``, plus
        ``segments`` on the recording's timeline when ``want_segments`` is set,
        ``capped`` when the answer may still be cut off, ``thin`` when it still
        has too few words, ``record`` (see :func:`_chunk_record`) and
        ``usage``: the records of the requests paid for it (see :mod:`usage`).
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
        size_bytes = path.stat().st_size
        started = time.monotonic()
        try:
            result = _request(path, client, model, want_segments)
        finally:
            path.unlink(missing_ok=True)

        tokens = _output_tokens(result)
        span_ms = end_ms - start_ms
        spent = [usage.transcription_record(result, model, span_ms / 1000)]
        taken = time.monotonic() - started
        record = _chunk_record(
            name, span, result, tokens, taken, size_bytes, thin_floor
        )
        _log_record(record)
        capped, thin = record["capped"], record["thin"]
        splittable = may_split and span_ms >= 2 * MIN_SPLIT_SECONDS * 1000
        if not ((capped or thin) and splittable):
            if capped:
                logger.warning(
                    "Part %s still hit the output cap (%d tokens)", name, tokens
                )
            if thin:
                logger.warning(
                    "Part %s is still thin (%.0f words/min, floor %.0f); kept and "
                    "marked",
                    name,
                    record["words_per_min"],
                    thin_floor,
                )
            piece = _piece(result, start_ms, end_ms, want_segments, capped, thin)
            piece["usage"] = spent
            piece["record"] = record
            checkpoints.save(
                checkpoint, part, {"bounds": [start_ms, end_ms], "pieces": [piece]}
            )
            return [piece]

        # The answer stopped at the output cap, so the end of this stretch is
        # missing from it — or it has far too few words, so some of the
        # stretch is missing from it. Transcribe the two halves instead.
        if capped:
            logger.warning(
                "Part %s hit the output cap (%d tokens); splitting it", name, tokens
            )
        else:
            logger.warning(
                "Part %s is thin (%.0f words/min, floor %.0f); splitting it",
                name,
                record["words_per_min"],
                thin_floor,
            )
        middle = _quietest_cut(
            audio,
            (start_ms + end_ms) // 2,
            min(span_ms // 4, int(CUT_SEARCH_SECONDS * 1000)),
        )
        # The incomplete answer was paid for too; it is kept with the split
        # marker so a resumed run still counts it.
        checkpoints.save(
            checkpoint,
            part,
            {
                "bounds": [start_ms, end_ms],
                "split": middle,
                "usage": spent,
                "reason": "capped" if capped else "thin",
                "record": record,
            },
        )

    halves = ((start_ms, middle, f"{name}a"), (middle, end_ms, f"{name}b"))
    pieces = [
        piece
        for low, high, half in halves
        for piece in _transcribe_span(
            audio,
            (low, high),
            client,
            model,
            want_segments,
            work,
            half,
            False,
            thin_floor,
        )
    ]
    if pieces:
        first = pieces[0]
        own = _usage_records([first], model)
        pieces[0] = {**first, "usage": spent + own}
    return pieces


def _piece(
    result: Any,
    start_ms: int,
    end_ms: int,
    want_segments: bool,
    capped: bool,
    thin: bool = False,
) -> dict[str, Any]:
    """Turn one API answer into a piece placed on the recording's timeline.

    Args:
        result: The transcription response.
        start_ms: Where the transcribed stretch starts.
        end_ms: Where it ends.
        want_segments: Whether the response carries timed segments.
        capped: Whether the answer ended at the output cap.
        thin: Whether the answer had far too few words for its length.

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
    if thin:
        piece["thin"] = True
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
    span: tuple[float, float] | None = None,
    verified: tuple[list[dict[str, Any]], str] | None = None,
) -> dict[str, Any] | None:
    """Write the rendered transcript and, when requested, the SRT file.

    Timed segments give a timestamped, paragraphed transcript; without them each
    chunk's text is paragraphed and stamped with approximate times — or, when
    a local pass is given, re-timed by it and filled in from it. Serbian
    Cyrillic is rewritten in Latin unless ``SERBIAN_LATIN`` is off.

    Args:
        output_file: Destination ``.txt`` path.
        pieces: Per-chunk texts with their place in the recording.
        timed_entries: Timed segments, if the engine returned any.
        srt_output_file: Destination ``.srt`` path, or None to skip subtitles.
        visual_notes: Optional on-screen notes with ``time`` and ``description``.
        missing: ``(start, end, reason)`` of an untranscribed tail, if any.
        span: ``(start, end)`` of the stretch transcribed on purpose, if not
            the whole recording; the transcript says so in its first line.
        verified: The local pass's timed segments and the local model's name,
            to check untimed pieces against (:func:`build_verified_transcript`).

    Returns:
        The cross-check's figures when a local pass was used, else ``None``.
    """
    stats = None
    if timed_entries:
        notes = list(visual_notes or [])
        for piece in pieces:
            end = piece["start"] + piece["duration"]
            notes += [{"time": end, "line": line} for line in _flag_lines(piece)]
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
    elif verified is not None and not missing:
        entries, local_model = verified
        document, stats = build_verified_transcript(
            pieces, entries, local_model, visual_notes
        )
    else:
        document = build_untimed_transcript(pieces, visual_notes, missing)
    if span is not None:
        document = f"{format_span_note(*span)}\n\n{document}"

    srt = build_srt(timed_entries) if srt_output_file and timed_entries else None
    if get_settings().serbian_latin and serbian.is_serbian_cyrillic(document):
        document = serbian.cyrillic_to_latin(document)
        srt = serbian.cyrillic_to_latin(srt) if srt else srt

    output_file.write_text(document + "\n", encoding="utf-8")
    if srt_output_file and srt is not None:
        Path(srt_output_file).write_text(srt, encoding="utf-8")
    return stats


def _transcribe_all(
    audio: AudioSegment,
    bounds: list[tuple[int, int]],
    client: OpenAI,
    model: str,
    want_segments: bool,
    work: tuple[Path, Path],
    progress_callback: ProgressCallback | None,
    borrowed: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
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
        borrowed: Chunks saved for other stretches of the recording, by
            bounds (:func:`_saved_elsewhere`); one with the same bounds is
            taken instead of a request.

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
            logger.info("Part %03d reused from the saved parts of this run", index)
            pieces += saved["pieces"]
            continue
        lent = (borrowed or {}).get((start_ms, end_ms))
        if lent is not None:
            # Saved as this run's own, so its count and clean-up stay simple.
            logger.info("Part %03d reused from another stretch's saved parts", index)
            checkpoints.save(
                checkpoint, name, {"bounds": [start_ms, end_ms], "pieces": lent}
            )
            pieces += lent
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
                thin_floor=_thin_floor(pieces),
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


def _reusable_chunks(
    checkpoint: Path,
    bounds: list[tuple[int, int]],
    borrowed: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> int:
    """Count saved chunks that match the planned boundaries and will be reused.

    Args:
        checkpoint: The checkpoint directory.
        bounds: ``(start_ms, end_ms)`` of each chunk.
        borrowed: Chunks saved for other stretches, by bounds.

    Returns:
        How many chunks need no request.
    """
    return sum(
        1
        for index, (start_ms, end_ms) in enumerate(bounds)
        if (
            (saved := checkpoints.load(checkpoint, f"chunk_{index:03d}"))
            and saved.get("bounds") == [start_ms, end_ms]
        )
        or (start_ms, end_ms) in (borrowed or {})
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
    span: tuple[float, float] | None = None,
    verify: tuple[Any, str] | None = None,
    verify_batched: bool = False,
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
        span: ``(start, end)`` in seconds to transcribe only that stretch of
            the recording; its times in the transcript stay the recording's.
        verify: A loaded ``faster_whisper.WhisperModel`` and its name: after
            the API run, the same audio is transcribed locally and the
            transcript is re-timed by, and filled in from, that pass
            (:func:`build_verified_transcript`). Only for models without
            timestamps; ignored when the run stops part-way.
        verify_batched: Run that local pass in batches (:func:`_local_entries`).

    Returns:
        What the requests used and cost (:func:`usage.summarize`), counting
        chunks paid for by an earlier, unfinished attempt too, with
        ``audio_seconds`` (the length transcribed), ``chunks`` (one record per
        answered part, :func:`chunk_records`) and, after a cross-check,
        ``verified`` (its figures and how long the local pass took).

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
        digest = checkpoints.file_digest(input_file)
        checkpoint = checkpoint_dir(digest, model, span)
        borrowed = _saved_elsewhere(checkpoint_dir(digest, model), checkpoint)
        audio = _load_audio(input_file, temp_folder)
        bounds = _chunk_bounds(
            audio, segment_minutes(model) * 60_000, _span_ms(span, len(audio))
        )
        logger.info(
            "Chunk plan for %s with %s: %d part(s): %s",
            input_file.name,
            model,
            len(bounds),
            ", ".join(
                f"{_format_clock(start / 1000)}–{_format_clock(end / 1000)}"
                for start, end in bounds
            ),
        )
        # The stretch being transcribed: the whole recording, or the chosen
        # part of it, placed on the recording's own timeline.
        first, last = bounds[0][0] / 1000, bounds[-1][1] / 1000
        covered = (first, last) if span is not None else None
        restored = _reusable_chunks(checkpoint, bounds, borrowed)
        resume = f" — {restored} already done, resuming" if restored else ""
        stretch = ""
        if covered:
            stretch = (
                f" ({_format_clock(first)}–{_format_clock(last)} of the "
                f"{_format_clock(len(audio) / 1000)} recording)"
            )
        _report(
            progress_callback,
            status="info",
            message=(
                f"{_format_clock(last - first)} of audio{stretch} in "
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
            borrowed,
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
                "audio_seconds": round(bounds[index][0] / 1000 - first, 3),
            }
            # Keep what was paid for: the finished chunks stay checkpointed, and
            # a partial transcript says plainly where it stops and why.
            _write_outputs(
                output_file,
                pieces,
                timed_entries,
                None,
                visual_notes=visual_notes,
                missing=(bounds[index][0] / 1000, last, str(cause)),
                span=covered,
            )
            raise IncompleteTranscriptionError(
                str(cause), completed=index, total=len(bounds), spent=spent
            ) from cause

        local_pass = None
        checked: dict[str, Any] | None = None
        local_seconds = 0.0
        if verify is not None and not timed_entries:
            local_model, local_name = verify
            local_started = time.monotonic()
            try:
                entries, _ = _local_entries(
                    input_file,
                    local_model,
                    covered,
                    progress_callback,
                    f"Checking with local Whisper ({local_name})",
                    model_name=local_name,
                    batched=verify_batched,
                )
                local_pass = (entries, local_name)
            except Exception as exc:
                # The paid transcript is written as it is: the check is a
                # bonus, and its failure must not cost the run.
                logger.exception("Cross-check with local %s failed", local_name)
                checked = {"model": local_name, "error": str(exc)}
            local_seconds = time.monotonic() - local_started
        stats = _write_outputs(
            output_file,
            pieces,
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
            span=covered,
            verified=local_pass,
        )
        if stats is not None:
            checked = {**stats, "seconds": round(local_seconds, 1)}
            logger.info(
                "Cross-check with local %s in %.0f s: %d of %d paragraphs re-timed, "
                "%d gap(s) with %d words added, %d API-only words, %.0f%% of API "
                "words matched",
                checked["model"],
                checked["seconds"],
                checked["anchored"],
                checked["paragraphs"],
                checked["gaps"],
                checked["added_words"],
                checked["api_only_words"],
                100 * checked["matched_share"],
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
        # the length transcribed is reported alongside it.
        summary = {
            **usage.summarize(_usage_records(pieces, model)),
            "audio_seconds": round(last - first, 3),
            "chunks": chunk_records(pieces),
        }
        if checked is not None:
            summary["verified"] = checked
        return summary

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
    span: tuple[float, float] | None = None,
    model_name: str | None = None,
    batched: bool = False,
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
        span: ``(start, end)`` in seconds to transcribe only that stretch of
            the recording; its times in the transcript stay the recording's.
        model_name: The model's name, for the log line.
        batched: Decode in batches (:func:`_local_entries`).

    Raises:
        TranscriptionError: If transcription fails.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    started = time.monotonic()
    try:
        timed_entries, _ = _local_entries(
            input_file,
            whisper_model,
            span,
            progress_callback,
            model_name=model_name,
            batched=batched,
        )
        _write_outputs(
            output_file,
            [],
            timed_entries,
            srt_output_file if with_timestamps else None,
            visual_notes=visual_notes,
            span=span,
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


def _local_entries(
    input_file: Path,
    whisper_model: Any,
    span: tuple[float, float] | None,
    progress_callback: ProgressCallback | None,
    label: str = "Transcribing",
    model_name: str | None = None,
    batched: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Run a local faster-whisper model over the recording, or a stretch of it.

    Args:
        input_file: Source audio or video file (ffmpeg-readable).
        whisper_model: A loaded ``faster_whisper.WhisperModel`` instance.
        span: ``(start, end)`` in seconds to transcribe only that stretch; the
            times returned stay the recording's own.
        progress_callback: Optional callback receiving status payloads.
        label: What the progress messages call the work.
        model_name: The model's name, for the log line.
        batched: Cut the recording at pauses into pieces of up to 30 s and
            write ``LOCAL_BATCH_SIZE`` of them at once — faster, but each
            piece without the text before it and without a second try. The
            pieces come back split into sentences (:func:`_split_by_words`).

    Returns:
        Timed segments with ``start``, ``end`` and ``text`` on the
        recording's timeline, and the audio length the model reported.

    Raises:
        TranscriptionError: If the stretch is too short to transcribe.
    """
    source = input_file
    offset = 0.0
    cut: Path | None = None
    started = time.monotonic()
    try:
        if span is not None:
            # The stretch is cut out into a WAV of its own: faster-whisper's own
            # clip option would switch its silence filter off.
            start, end = span
            if (end - start) * 1000 < _MIN_SPAN_MS:
                raise TranscriptionError(
                    f"Nothing to transcribe between {_format_clock(start)} and "
                    f"{_format_clock(end)} — move the ends of the slider apart."
                )
            temp_dir = get_settings().temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(
                prefix="stretch-", suffix=".wav", dir=temp_dir
            )
            os.close(handle)
            cut = Path(name)
            _report(
                progress_callback,
                status="start",
                message=(
                    f"Cutting out {_format_clock(start)}–{_format_clock(end)} "
                    "of the recording…"
                ),
            )
            to_wav(input_file, cut, start=start, duration=end - start)
            source, offset = cut, start
        _report(progress_callback, status="start", message=f"{label} started…")
        # transcribe() returns a lazy generator; iterating it does the work.
        if batched:
            # Not with the pieces' own timestamps: with them faster-whisper
            # 1.2.1 left out a piece's last sentence. Word times instead, which
            # also split each piece into sentences.
            segments, info = _batched_pipeline(whisper_model).transcribe(
                str(source),
                vad_filter=True,
                batch_size=LOCAL_BATCH_SIZE,
                word_timestamps=True,
            )
        else:
            segments, info = whisper_model.transcribe(str(source), vad_filter=True)

        # faster-whisper always returns timed segments, so collect them even when
        # no SRT was requested — they drive the readable transcript. Times are
        # placed on the recording's own timeline when a stretch was cut out.
        timed_entries: list[dict[str, Any]] = []
        duration = getattr(info, "duration", 0) or 0
        for segment in segments:
            parts = (
                _split_by_words(segment)
                if batched
                else [(segment.start, segment.end, segment.text)]
            )
            timed_entries += [
                {"start": start + offset, "end": end + offset, "text": text.strip()}
                for start, end, text in parts
            ]
            if progress_callback and duration:
                _report(
                    progress_callback,
                    status="progress",
                    message=(
                        f"{label}… {segment.end + offset:.0f}/{duration + offset:.0f}s"
                    ),
                    progress=min(1.0, segment.end / duration),
                )
        text = " ".join(entry["text"] for entry in timed_entries)
        logger.info(
            "Local pass over %s (%s, %s%s) in %.0f s: %d segment(s), %d words, "
            "%.0f words/min",
            input_file.name,
            model_name or "local model",
            f"{_format_clock(span[0])}–{_format_clock(span[1])}" if span else "whole",
            ", batched" if batched else "",
            time.monotonic() - started,
            len(timed_entries),
            len(text.split()),
            words_per_minute(text, duration),
        )
        return timed_entries, float(duration)
    finally:
        if cut is not None:
            cut.unlink(missing_ok=True)


def _batched_pipeline(whisper_model: Any) -> Any:
    """Wrap a loaded model for batched decoding.

    Args:
        whisper_model: A loaded ``faster_whisper.WhisperModel`` instance.

    Returns:
        A ``faster_whisper.BatchedInferencePipeline`` around the model.
    """
    # Imported here: faster-whisper is an optional dependency.
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model=whisper_model)


def _split_by_words(segment: Any) -> list[tuple[float, float, str]]:
    """Split a batched piece into sentence-sized lines by its word times.

    A batched piece runs up to 30 s: as one subtitle it would stay on screen
    that long, and the transcript could break paragraphs only between pieces.
    A line ends after a word that ends a sentence, before a word that follows
    a pause, and before it would grow past ``_BATCHED_MAX_LINE_SECONDS``.

    Args:
        segment: A faster-whisper segment with ``words``.

    Returns:
        ``(start, end, text)`` per line; the segment as it is when it carries
        no word times.
    """
    words = list(getattr(segment, "words", None) or [])
    if not words:
        return [(segment.start, segment.end, segment.text)]
    lines: list[list[Any]] = []
    current: list[Any] = []
    for word in words:
        if current and (
            word.start - current[-1].end >= _BATCHED_SPLIT_PAUSE_SECONDS
            or word.end - current[0].start > _BATCHED_MAX_LINE_SECONDS
        ):
            lines.append(current)
            current = []
        current.append(word)
        if word.word.strip().endswith((".", "!", "?", "…")):
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return [
        (line[0].start, line[-1].end, "".join(word.word for word in line))
        for line in lines
    ]
