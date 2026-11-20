"""One run, from a prepared recording to a saved history row.

The order of a run — on-screen context, the engine, the AI title, the cost, the
history row, and only then the discard of the saved parts — used to live in the
page (``app.run_transcription``), tied to Streamlit's session state. It lives
here so that the page and the command line (:mod:`cli`) do exactly the same
thing. This module never imports Streamlit.

A run leaves its results in a *sink*: any object whose attributes can be set
(``transcript_path``, ``srt_path``, ``elapsed_seconds``, ``cost``, ``partial``,
``title``, ``row_id``, ``notices``). The page's sink writes each one to the
session state, and that is deliberate: a Streamlit Stop takes effect at the
next Streamlit call, so each result set here is a point where the page can
stop the run — and there is none between writing the history row and
discarding the saved parts, where a Stop would let the next Start save the
same transcript a second time. The command line's sink is a plain object.
"""

import json
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import audio
import checkpoints
import db
import frames
import titles
import transcribe
import usage
import vision
from config import (
    AUDIO_FORMATS,
    DEFAULT_TITLE_MODEL,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    TITLE_MODE_OFF,
    TRANSCRIPT_PREFIX,
    TRANSCRIPTION_FALLBACK_PRICE_PER_MINUTE,
    TRANSCRIPTION_PRICE_PER_MINUTE,
    VIDEO_FORMATS,
    get_settings,
)
from exceptions import (
    AppError,
    AudioProcessingError,
    IncompleteTranscriptionError,
    OpenAIAccountError,
)
from logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]
Notice = tuple[str, str]
Span = tuple[int, int]

NO_KEY_NOTICE = (
    "Add your OpenAI API key in the sidebar, or set `OPENAI_API_KEY` "
    "in a `.env` file. (The Local engine needs no key.)"
)
_LOCAL_HINT = " The Local engine needs no key or credit."


@dataclass(frozen=True)
class RunRequest:
    """What one run works on and how.

    Attributes:
        filename: The recording's name, as History shows it.
        source_type: ``"audio"`` or ``"video"``.
        audio_path: The audio to transcribe (a video's extracted WAV).
        transcript_path: Where the transcript is written.
        srt_path: Where subtitles are written when ``with_timestamps``.
        provider: The engine (``PROVIDER_OPENAI`` or ``PROVIDER_LOCAL``).
        model: The API model name, or the local model size.
        video_path: The video itself, for on-screen context.
        with_timestamps: Whether subtitles are wanted.
        visual: On-screen context settings (``model``, ``detail``,
            ``interval``), or ``None`` to skip that step.
        span: ``(start, end)`` in seconds when only that stretch is wanted.
        verify: The local model to check an OpenAI transcript against.
        batched: Decode the local pass (the run's, or the check's) in batches.
        api_key: The OpenAI key, if there is one.
        local_available: Whether the offline engine is installed, so that a
            refused account can be pointed to it, and the parts an OpenAI run
            still marks incomplete can be filled by it (:func:`fill_model`).
    """

    filename: str
    source_type: str
    audio_path: Path | None
    transcript_path: Path
    srt_path: Path
    provider: str
    model: str
    video_path: Path | None = None
    with_timestamps: bool = False
    visual: dict[str, Any] | None = None
    span: Span | None = None
    verify: str | None = None
    batched: bool = False
    api_key: str | None = None
    local_available: bool = False


def _log_progress(info: dict[str, Any]) -> None:
    """Write a progress payload to the log (the default progress hook)."""
    logger.info("%s", info.get("message", info.get("status")))


@contextmanager
def _log_stage(message: str) -> Iterator[None]:
    """Log what a long step is doing (the default stage hook)."""
    logger.info("%s", message)
    yield


def load_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load a local faster-whisper model (downloads on first use).

    Args:
        model_name: Model size (e.g. ``"base"``).
        device: Compute device ("auto", "cpu" or "cuda").
        compute_type: Quantization (e.g. ``"int8"``).

    Returns:
        A loaded ``faster_whisper.WhisperModel`` instance.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(get_settings().whisper_model_dir),
    )


def load_local_model(model: str) -> Any:
    """Load a local model with the device and quantization from the settings.

    Args:
        model: The local model size.

    Returns:
        The loaded ``faster_whisper.WhisperModel``.
    """
    settings = get_settings()
    return load_whisper_model(model, settings.local_device, settings.local_compute_type)


def saved_parts_hint(
    audio_path: Path | None,
    model: str,
    total: int | None = None,
    span: Span | None = None,
    digest: Callable[[Path], str] = checkpoints.file_digest,
    retry: str = "run it again",
    retry_verb: str = "A new run",
) -> str:
    """Say how much of an unfinished run is really saved.

    Counted on disk rather than promised: the saved parts can be gone (a
    clean-up, a disk error), and then the next run would pay for them again.

    Args:
        audio_path: The audio the run was for.
        model: The transcription model of the run.
        total: Parts in the recording, when known.
        span: The stretch of the recording the run was for, if not all of it.
        digest: Returns a file's content digest.
        retry: How the reader starts the run again, e.g. ``press Start``.
        retry_verb: What then transcribes it, e.g. ``Start``.

    Returns:
        One sentence for the notice.
    """
    saved = 0
    if audio_path and audio_path.exists():
        saved = transcribe.saved_chunk_count(digest(audio_path), model, span)
    if not saved:
        return (
            f"No finished parts are saved — {retry_verb} transcribes it from "
            "the beginning."
        )
    of_total = f" of {total}" if total else ""
    return (
        f"{saved}{of_total} part(s) are saved — {retry} to transcribe only "
        "the rest; saved parts are not sent or paid for again."
    )


def _no_title() -> tuple[str, str]:
    """Return the AI title setting of a run that asks for none (the default)."""
    return TITLE_MODE_OFF, DEFAULT_TITLE_MODEL


def _default_parts_hint(
    request: RunRequest, total: int | None, digest: Callable[[Path], str]
) -> str:
    """Phrase :func:`saved_parts_hint` for a run with no page (the default)."""
    return saved_parts_hint(
        request.audio_path, request.model, total, request.span, digest
    )


@dataclass(frozen=True)
class RunHooks:
    """How a run talks to whatever started it.

    Attributes:
        progress: Receives the engines' ``status`` / ``message`` payloads.
        stage: A context manager shown around a long step (the page's spinner).
        load_local_model: Returns a loaded local model for a model size.
        digest: Returns a file's content digest (the page caches it).
        parts_hint: Phrases how much of an unfinished run is saved, given the
            request, the number of parts and ``digest``.
        title_setting: Returns the AI title mode (a ``TITLE_MODES`` key) and
            the chat model that writes it. Asked only once a transcript is
            written, as the page's setting lives in the history database.
    """

    progress: ProgressCallback = _log_progress
    stage: Callable[[str], AbstractContextManager[Any]] = _log_stage
    load_local_model: Callable[[str], Any] = load_local_model
    digest: Callable[[Path], str] = checkpoints.file_digest
    parts_hint: Callable[[RunRequest, int | None, Callable[[Path], str]], str] = (
        _default_parts_hint
    )
    title_setting: Callable[[], tuple[str, str]] = _no_title


def safe_stem(filename: str) -> str:
    """Return a recording's name without extension, safe to build paths from.

    Args:
        filename: The file's name as the browser or the folder gave it.

    Returns:
        The stem of :func:`audio.safe_name`.
    """
    return Path(audio.safe_name(filename)).stem


def format_length(seconds: float) -> str:
    """Render a duration as ``M:SS`` or ``H:MM:SS``.

    Args:
        seconds: Duration in seconds.

    Returns:
        The duration as a clock string.
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def scan_dir(video_path: Path, digest: Callable[[Path], str]) -> Path:
    """Return where the scan of a video is kept (named after its content).

    Args:
        video_path: The video.
        digest: Returns a file's content digest.

    Returns:
        A folder under ``temp_dir``, removed by clean-up or after a day.
    """
    return get_settings().temp_dir / f"scan-{digest(video_path)}"


def partial_context_notice(
    exc: Exception, collected: list[dict[str, Any]], total: int
) -> str:
    """Phrase why on-screen context stopped, and what was kept.

    Args:
        exc: The error that stopped it.
        collected: Notes made before the stop.
        total: Screenshots there were to describe (0 if extraction failed).

    Returns:
        A one-line warning.
    """
    if not collected:
        return f"On-screen context skipped: {exc}"
    return (
        f"On-screen context incomplete: {exc} The {len(collected)} notes made "
        f"before that (of {total} screenshots) are in the transcript."
    )


def collect_visual_notes(
    video_path: Path | None,
    api_key: str | None,
    visual: dict[str, Any],
    notices: list[Notice],
    stop_on_account_error: bool,
    hooks: RunHooks,
    spent: list[dict[str, Any]] | None = None,
    used: list[str] | None = None,
    span: Span | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Extract key frames from a video and describe what they show.

    Failures are reported, and whatever was described before them is kept — it
    was paid for — because a transcript with some on-screen notes is still worth
    having. An account failure (bad key, no credit) is raised instead when the
    transcription itself would use OpenAI and be refused the same way.

    Args:
        video_path: The video, if there is one.
        api_key: The OpenAI key, if there is one.
        visual: On-screen context settings (``model``, ``detail``,
            ``interval``).
        notices: Messages for the user are appended here.
        stop_on_account_error: Raise account errors instead of warning.
        hooks: Progress, stage and digest of the run.
        spent: Usage records of the descriptions are appended here.
        used: The cache names of the selected frames are appended here, so
            the caller can mark exactly those as counted.
        span: The stretch of the video to transcribe, when not all of it: the
            whole video is scanned (the scan is kept for any stretch), and only
            the screenshots within the stretch are described.

    Returns:
        Notes with ``time`` and ``description`` (possibly empty), and whether
        every screenshot was described — only then may the cache be discarded.

    Raises:
        OpenAIAccountError: If OpenAI refuses the key or the account and
            ``stop_on_account_error`` is set.
    """
    if not video_path or not api_key:
        return [], True

    failed: list[float] = []
    collected: list[dict[str, Any]] = []
    total = 0
    try:
        with hooks.stage(
            "Looking through the video for screen changes — about a minute and a "
            "half per hour of video…"
        ):
            # The same scan and selection the caption counted, so the number
            # described is the number that was shown.
            scan = frames.scan_video(video_path, scan_dir(video_path, hooks.digest))
            if span is not None:
                scan = frames.clip_scan(scan, span)
            keyframes = frames.select_from_scan(scan, visual["interval"])
            if used is not None:
                used.extend(vision.frame_name(frame["time"]) for frame in keyframes)
            cache = vision.cache_dir(
                hooks.digest(video_path), visual["model"], visual["detail"]
            )
        if not keyframes:
            notices.append(
                ("info", "No on-screen changes were detected — nothing to describe.")
            )
            return [], True
        total = len(keyframes)
        with (
            hooks.stage(f"Describing {total} screenshots…"),
            frames.pictures(video_path) as picture,
        ):
            notes = vision.describe_keyframes(
                keyframes,
                api_key,
                model=visual["model"],
                detail=visual["detail"],
                progress_callback=hooks.progress,
                cache=cache,
                failed=failed,
                collected=collected,
                spent=spent,
                picture=picture,
            )
        if failed:
            notices.append(
                (
                    "warning",
                    f"{len(failed)} of {len(keyframes)} screenshots could not be "
                    "described and are missing from the transcript.",
                )
            )
        return notes, not failed
    except OpenAIAccountError as exc:
        if stop_on_account_error:
            raise
        notices.append(("warning", partial_context_notice(exc, collected, total)))
    except (AppError, OSError) as exc:
        notices.append(("warning", partial_context_notice(exc, collected, total)))
    return sorted(collected, key=lambda note: note["time"]), False


def describe_verified(verified: dict[str, Any]) -> str:
    """Phrase what the cross-check against a local pass did.

    Args:
        verified: The ``verified`` part of a run's transcription summary.

    Returns:
        One sentence, e.g. ``Checked against local Whisper (base) in 13:02:
        118 of 124 paragraphs re-timed; 2 stretches (312 words) the OpenAI
        answer lacked were added``.
    """
    taken = (
        f" in {format_length(verified['seconds'])}" if verified.get("seconds") else ""
    )
    text = (
        f"Checked against local Whisper ({verified['model']}){taken}: "
        f"{verified['anchored']} of {verified['paragraphs']} paragraphs re-timed"
    )
    if verified.get("gaps"):
        text += (
            f"; {verified['gaps']} stretch(es) ({verified['added_words']} words) the "
            "OpenAI answer lacked were added, marked 🧩"
        )
    else:
        text += "; nothing the OpenAI answer lacked was found"
    return text + "."


def describe_filled(filled: dict[str, Any], chunks: list[dict[str, Any]]) -> str | None:
    """Phrase what local Whisper did with the parts a run left incomplete.

    Args:
        filled: The ``filled`` part of a run's transcription summary.
        chunks: The run's chunk records, each checked one with its
            ``local_check``.

    Returns:
        One sentence, e.g. ``Local Whisper (medium) filled 2 of 2 part(s) that
        came back incomplete from OpenAI, in 3:12: 20:00–22:29 (312 words),
        25:05–27:34 (149 words), marked 🧩.``; ``None`` when no part could be
        checked (the warning on the failure says so).
    """
    checked = filled["parts"] - filled.get("failed", 0)
    if not checked:
        return None
    taken = f", in {format_length(filled['seconds'])}" if filled.get("seconds") else ""
    who = f"Local Whisper ({filled['model']})"
    of = f"part(s) that came back incomplete from OpenAI{taken}"
    if not filled["filled"]:
        return f"{who} checked the {checked} {of}, and found nothing more in them."
    where = [
        f"{format_length(record['start'])}–{format_length(record['end'])} "
        f"({record['local_check']['added_words']} words)"
        for record in chunks
        if (record.get("local_check") or {}).get("added_words")
    ]
    return (
        f"{who} filled {filled['filled']} of {filled['parts']} {of}: "
        f"{', '.join(where)}, marked 🧩."
    )


def run_warnings(spent: dict[str, Any] | None) -> list[Notice]:
    """Say what a finished OpenAI run found out about its own answers.

    Args:
        spent: The pipeline's summary (``chunks``, ``verified``, ``filled``),
            if any.

    Returns:
        Notices: a warning for parts that may have lost speech, a note on what
        local Whisper filled in, and a note or warning on the cross-check
        against a local pass.
    """
    notices: list[Notice] = []
    if not spent:
        return notices
    chunks = spent.get("chunks") or []
    flagged = transcribe.describe_flagged(chunks)
    if flagged:
        notices.append(("warning", flagged))
    filled = spent.get("filled")
    if filled:
        note = describe_filled(filled, chunks)
        if note:
            notices.append(("info", note))
        if filled.get("failed"):
            notices.append(
                (
                    "warning",
                    f"Local Whisper ({filled['model']}) could not check "
                    f"{filled['failed']} of the {filled['parts']} part(s) that came "
                    f"back incomplete: {filled.get('error')} They stay marked ⚠️.",
                )
            )
    verified = spent.get("verified")
    if verified and verified.get("error"):
        notices.append(
            (
                "warning",
                f"The check with local Whisper ({verified['model']}) failed: "
                f"{verified['error']} The transcript keeps its estimated times.",
            )
        )
    elif verified:
        notices.append(("info", describe_verified(verified)))
    return notices


def with_span(cost: dict[str, Any], span: Span | None) -> dict[str, Any]:
    """Record on a run's cost which stretch of the recording it transcribed.

    Args:
        cost: The run's cost from :func:`run_cost`.
        span: The stretch, or ``None`` for the whole recording.

    Returns:
        The cost, with ``span`` added when part of the recording was left out.
    """
    return {**cost, "span": list(span)} if span is not None else cost


def run_cost(
    provider: str,
    transcription: dict[str, Any] | None,
    vision_records: list[dict[str, Any]],
    transcription_ran: bool = True,
    title_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Put a run's transcription, screenshot and title usage together.

    Args:
        provider: The engine used (local transcription costs nothing).
        transcription: :func:`usage.summarize` of the transcription requests,
            or ``None`` when the pipeline did not report it.
        vision_records: Usage records of the screenshot descriptions.
        transcription_ran: False when the run stopped before transcribing.
        title_records: Usage records of the AI title request, if one was made.

    Returns:
        ``transcription``, ``vision`` and ``title`` summaries (``None`` when
        unused), the total ``cost_usd`` and whether any part of it is
        ``estimated``.
    """
    described = usage.summarize(vision_records) if vision_records else None
    title = usage.summarize(title_records) if title_records else None
    unknown = (
        provider == PROVIDER_OPENAI and transcription_ran and transcription is None
    )
    parts = [part for part in (transcription, described, title) if part]
    return {
        "provider": provider,
        "transcription": transcription,
        "vision": described,
        "title": title,
        "cost_usd": sum(part["cost_usd"] for part in parts),
        "estimated": unknown or any(part["estimated"] for part in parts),
    }


def describe_cost(cost: dict[str, Any]) -> str:
    """Phrase what a run cost and what it used, for a caption.

    Args:
        cost: The run's cost from :func:`run_cost`.

    Returns:
        E.g. ``:material/payments: $0.0071 — transcription 1 request, 0:56 of
        audio, 1,210 → 240 tokens; screenshots 4 described, 2,580 → 190
        tokens``.
    """
    parts = []
    transcription = cost.get("transcription")
    if transcription:
        tokens = ""
        if transcription["input_tokens"] or transcription["output_tokens"]:
            tokens = (
                f", {transcription['input_tokens']:,} → "
                f"{transcription['output_tokens']:,} tokens"
            )
        audio_seconds = transcription.get("audio_seconds", transcription["seconds"])
        parts.append(
            f"transcription {transcription['requests']} request(s), "
            f"{format_length(audio_seconds)} of audio{tokens}"
        )
    elif cost.get("provider") == PROVIDER_LOCAL:
        parts.append("transcription free (local)")
    described = cost.get("vision")
    if described:
        parts.append(
            f"screenshots {described['requests']} described, "
            f"{described['input_tokens']:,} → {described['output_tokens']:,} tokens"
        )
    title = cost.get("title")
    if title:
        parts.append(
            f"title {title['input_tokens']:,} → {title['output_tokens']:,} tokens"
        )
    verified = (transcription or {}).get("verified")
    if verified and not verified.get("error"):
        parts.append(f"checked with local Whisper ({verified['model']}, free)")
    filled = (transcription or {}).get("filled")
    if filled and filled["parts"] > filled.get("failed", 0):
        parts.append(
            f"{filled['parts']} incomplete part(s) checked with local Whisper "
            f"({filled['model']}, free)"
        )
    span = cost.get("span")
    if span:
        parts.append(
            f"{format_length(span[0])}–{format_length(span[1])} of the recording "
            "transcribed, the rest left out"
        )
    total = usage.format_usd(cost["cost_usd"], cost.get("estimated", False))
    icon = ":material/payments:"
    text = f"{icon} {total} — " + "; ".join(parts) if parts else f"{icon} {total}"
    if cost.get("partial"):
        text += " — spent so far, not saved to history."
        if transcription:
            text += (
                " Start again with the same model to transcribe only the rest; "
                "that run counts this in its total."
            )
        if described:
            text += (
                " The screenshot descriptions are kept: the next run with the same "
                "screenshot model and detail reuses them and counts them in its total."
            )
    return text


def _note_spent_before_failure(
    provider: str,
    vision_spent: list[dict[str, Any]],
    notices: list[Notice],
    transcription_spent: dict[str, Any] | None = None,
) -> None:
    """Say what a failed run already paid for (nothing is saved).

    Args:
        provider: The engine of the run.
        vision_spent: Usage records of the descriptions made before the failure.
        notices: Messages for the user are appended here.
        transcription_spent: Usage summary of transcription requests paid before
            the failure (halves of a split first chunk), if any.
    """
    if vision_spent or transcription_spent:
        cost = run_cost(
            provider,
            transcription_spent,
            vision_spent,
            transcription_ran=transcription_spent is not None,
        )
        notices.append(("info", describe_cost({**cost, "partial": True})))


def _digest_if_any(digest: Callable[[Path], str], path: Path | None) -> str | None:
    """Return a file's digest, or ``None`` when there is no readable file."""
    try:
        return digest(path) if path else None
    except OSError:
        return None


def _remembering(digest: Callable[[Path], str]) -> Callable[[Path], str]:
    """Return ``digest`` hashing each file once for the length of a run.

    A recording from a folder that is still being written would hash
    differently at the end, and what is discarded then must be what this run
    saved; and a large video is not read three times over.
    """
    known: dict[Path, str] = {}

    def remembered(path: Path) -> str:
        if path not in known:
            known[path] = digest(path)
        return known[path]

    return remembered


def fill_model(local_available: bool, verify: str | None = None) -> str | None:
    """Return the local model that fills what an OpenAI run left incomplete.

    Decided here, for the page and the command line alike: on unless
    ``LOCAL_FILL`` is off, and only with the offline engine installed. A run
    that verifies the whole recording against local Whisper needs no second
    check of some of its parts.

    Args:
        local_available: Whether the offline engine is installed.
        verify: The local model of the run's whole-recording check, if any.

    Returns:
        ``LOCAL_FILL_MODEL``, or ``None`` when the parts stay marked ``⚠️``.
    """
    settings = get_settings()
    if not settings.local_fill or not local_available or verify:
        return None
    return settings.local_fill_model


def _run_engine(
    request: RunRequest,
    hooks: RunHooks,
    visual_notes: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Run the chosen engine.

    Args:
        request: The run.
        hooks: Progress, stage and model loading of the run.
        visual_notes: On-screen notes to place into the transcript.

    Returns:
        What the OpenAI requests used (``None`` for the local engine).
    """
    srt_path = request.srt_path if request.with_timestamps else None
    if request.provider == PROVIDER_LOCAL:
        whisper_model = hooks.load_local_model(request.model)
        with hooks.stage("Transcribing locally… This may take a while on CPU."):
            transcribe.transcribe_local(
                request.audio_path,
                request.transcript_path,
                whisper_model,
                with_timestamps=request.with_timestamps,
                srt_output_file=srt_path,
                progress_callback=hooks.progress,
                visual_notes=visual_notes,
                span=request.span,
                model_name=request.model,
                batched=request.batched,
            )
        return None

    # Loaded before the paid run, so a model that cannot be loaded (no disk,
    # no download) fails before anything is spent.
    local = (
        (hooks.load_local_model(request.verify), request.verify)
        if request.verify
        else None
    )
    # Loaded only if a part is still incomplete after the run: most runs
    # need no local pass, and a first use downloads the model.
    filler = fill_model(request.local_available, request.verify)
    fill = (lambda: hooks.load_local_model(filler), filler) if filler else None
    with hooks.stage("Transcribing with the OpenAI API…"):
        return transcribe.transcribe_openai(
            request.audio_path,
            request.transcript_path,
            request.api_key,
            model=request.model,
            with_timestamps=request.with_timestamps,
            srt_output_file=srt_path,
            progress_callback=hooks.progress,
            visual_notes=visual_notes,
            span=request.span,
            verify=local,
            verify_batched=request.batched,
            fill=fill,
        )


def _make_title(
    request: RunRequest, hooks: RunHooks, notices: list[Notice]
) -> tuple[str | None, list[dict[str, Any]]]:
    """Name a finished transcript when the AI title setting asks for it.

    Runs after either engine: a Local run sends its transcript too. A title
    that cannot be made costs the run only a warning — the transcript is
    already paid for and is saved without one.

    Args:
        request: The run.
        hooks: Progress of the run.
        notices: Messages for the user are appended here.

    Returns:
        The title (``None`` without one) and the usage records of its request
        (empty when none was sent).
    """
    mode, model = hooks.title_setting()
    if mode == TITLE_MODE_OFF:
        return None, []
    if not request.api_key:
        notices.append(
            ("info", "No AI title: it needs an OpenAI API key (sidebar or `.env`).")
        )
        return None, []
    hooks.progress({"status": "info", "message": "Writing a title…"})
    try:
        title, records = titles.make_title(
            request.transcript_path.read_text(encoding="utf-8"),
            request.api_key,
            model,
        )
    except AppError as exc:
        notices.append(("warning", f"No AI title: {exc}"))
        return None, list(getattr(exc, "usage_records", None) or [])
    return title, records


def _save_row(
    request: RunRequest,
    srt_path: Path | None,
    elapsed_seconds: float,
    cost: dict[str, Any],
    title: str | None,
) -> int | None:
    """Write the finished run into the SQLite history.

    Args:
        request: The run.
        srt_path: The subtitles it wrote, if any.
        elapsed_seconds: How long the run took.
        cost: What it used and cost.
        title: Its AI title, if any.

    Returns:
        The new row's id, or ``None`` when there was no transcript to save.
    """
    if not request.transcript_path.exists():
        return None
    audio_path = request.audio_path
    file_size_mb = None
    if audio_path and audio_path.exists():
        file_size_mb = round(audio_path.stat().st_size / (1024 * 1024), 2)
    return db.add_transcription(
        filename=request.filename,
        source_type=request.source_type,
        model=request.model,
        provider=request.provider,
        with_timestamps=request.with_timestamps,
        transcript=request.transcript_path.read_text(encoding="utf-8"),
        srt=srt_path.read_text(encoding="utf-8") if srt_path else None,
        audio_path=str(audio_path) if audio_path else None,
        file_size_mb=file_size_mb,
        elapsed_seconds=elapsed_seconds,
        cost_usd=cost["cost_usd"],
        usage_json=json.dumps(cost),
        title=title,
    )


def run(request: RunRequest, results: Any, hooks: RunHooks | None = None) -> None:
    """Do one run and leave what it produced in ``results``.

    Messages for the user end up in ``results.notices`` rather than being
    raised: a run that fails is reported, not crashed. A finished run has a
    ``row_id``; one that stopped part-way has ``partial`` (``"3 of 11"``) and
    its partial transcript; one that failed has neither.

    Args:
        request: What to transcribe and how.
        results: The sink: an object whose attributes can be set (see the
            module docstring for which, and why they are set one by one).
        hooks: How to report progress, load models and identify files.
    """
    hooks = hooks or RunHooks()
    hooks = replace(hooks, digest=_remembering(hooks.digest))
    notices: list[Notice] = []
    # Drop the previous run's result state before starting. Without this, a run
    # that fails or returns early leaves the old figure on screen, so the result
    # panel claims "Finished in …" for a run that never finished.
    results.elapsed_seconds = None
    results.cost = None
    results.title = None
    results.row_id = None
    if request.provider == PROVIDER_OPENAI and not request.api_key:
        results.notices = [("error", NO_KEY_NOTICE)]
        return
    provider, model, span, visual = (
        request.provider,
        request.model,
        request.span,
        request.visual,
    )
    transcript_path = request.transcript_path
    vision_spent: list[dict[str, Any]] = []
    used_frames: list[str] = []
    # Timed from here so the reported figure matches the wait the user actually
    # sits through, on-screen context included.
    started = time.monotonic()

    try:
        # Taken before the run: a recording from a folder that is still being
        # written would hash differently at the end, and what is discarded then
        # must be what this run saved.
        audio_digest = video_digest = None
        if provider == PROVIDER_OPENAI:
            audio_digest = _digest_if_any(hooks.digest, request.audio_path)
        if visual:
            video_digest = _digest_if_any(hooks.digest, request.video_path)
        visual_notes = None
        visual_complete = False
        if visual:
            # An account error stops an OpenAI run (its transcription would be
            # refused the same way) and only warns a Local one.
            visual_notes, visual_complete = collect_visual_notes(
                request.video_path,
                request.api_key,
                visual,
                notices,
                stop_on_account_error=provider == PROVIDER_OPENAI,
                hooks=hooks,
                spent=vision_spent,
                used=used_frames,
                span=span,
            )

        transcription_spent = _run_engine(request, hooks, visual_notes)
        results.transcript_path = transcript_path
        srt_path = (
            request.srt_path
            if request.with_timestamps and request.srt_path.exists()
            else None
        )
        results.srt_path = srt_path
        notices += run_warnings(transcription_spent)
        title, title_spent = _make_title(request, hooks, notices)
        if title is not None:
            results.title = title
        elapsed = time.monotonic() - started
        results.elapsed_seconds = elapsed
        # Cleared only now: a retry that fails before writing anything still
        # shows the earlier partial transcript, and must still say so.
        results.partial = None
        cache = None
        if visual and video_digest:
            cache = vision.cache_dir(video_digest, visual["model"], visual["detail"])
            if visual_complete:
                # The cache is discarded below. Descriptions an earlier,
                # unsaved attempt paid for outside this selection would then
                # count in no row, so this one takes them.
                vision_spent += vision.unbilled_usage(
                    cache, visual["model"], used_frames
                )
        cost = with_span(
            run_cost(
                provider, transcription_spent, vision_spent, title_records=title_spent
            ),
            span,
        )
        if request.batched:
            # Recorded with the row, to tell batched runs apart later.
            cost = {**cost, "local_batched": True}
        results.cost = cost
        row_id = _save_row(request, srt_path, elapsed, cost, title)
        # Only now: a run stopped after its last chunk has already written
        # the transcript but not saved it, and must be able to resume free.
        # And straight after the row, with nothing set on ``results`` in
        # between: for the page each one is a Streamlit call, where a Stop
        # takes effect, and a Stop between the row and the discard would let
        # Start save the same transcript as a second row. The parts are first
        # copied beside the history: they are the only record of what each
        # chunk's answer held.
        if audio_digest:
            parts = transcribe.checkpoint_dir(audio_digest, model, span)
            if row_id is not None:
                checkpoints.keep_run(parts, row_id)
            checkpoints.discard(parts)
        if cache is not None:
            if visual_complete:
                checkpoints.discard(cache)
            else:
                # Kept so a rerun pays only for the missing screenshots;
                # this row has counted the ones it used.
                vision.mark_billed(cache, used_frames)
        results.row_id = row_id
    except IncompleteTranscriptionError as exc:
        results.transcript_path = transcript_path
        results.srt_path = None
        results.partial = f"{exc.completed} of {exc.total}"
        results.cost = {
            **with_span(run_cost(provider, exc.spent, vision_spent), span),
            "partial": True,
        }
        hint = (
            _LOCAL_HINT
            if isinstance(exc.__cause__, OpenAIAccountError) and request.local_available
            else ""
        )
        notices.append(
            (
                "error",
                f"Stopped after part {exc.completed} of {exc.total}: {exc}{hint}",
            )
        )
        notices.append(("info", hooks.parts_hint(request, exc.total, hooks.digest)))
    except OpenAIAccountError as exc:
        hint = _LOCAL_HINT if request.local_available else ""
        notices.append(("error", f"{exc}{hint}"))
        _note_spent_before_failure(provider, vision_spent, notices, exc.spent)
    except AppError as exc:
        notices.append(("error", f"Transcription error: {exc}"))
        _note_spent_before_failure(provider, vision_spent, notices, exc.spent)
    results.notices = notices


def sibling_transcript(recording: Path, out_dir: Path | None = None) -> Path:
    """Return where a recording's transcript goes when it is put beside it.

    Args:
        recording: The recording.
        out_dir: Another folder for the transcript, if not the recording's own.

    Returns:
        ``transcript_<the recording's name>.txt`` in that folder.
    """
    folder = out_dir or recording.parent
    return folder / f"{TRANSCRIPT_PREFIX}{recording.stem}.txt"


def source_type(recording: Path) -> str | None:
    """Return whether a file is a supported ``"audio"`` or ``"video"`` recording.

    Args:
        recording: Any path.

    Returns:
        The source type, or ``None`` for a file type the app does not take.
    """
    extension = recording.suffix.lower().lstrip(".")
    if extension in AUDIO_FORMATS:
        return "audio"
    if extension in VIDEO_FORMATS:
        return "video"
    return None


def prepare_source(recording: Path, wav_path: Path) -> tuple[Path, Path | None, str]:
    """Make a recording on disk ready for a run.

    An audio file is used where it lies; a video has its audio track extracted
    to ``wav_path`` (mono 16 kHz).

    Args:
        recording: The recording.
        wav_path: Where a video's audio is written.

    Returns:
        The audio to transcribe, the video (``None`` for an audio file) and
        the source type.

    Raises:
        AudioProcessingError: If the file type is not supported, or ffmpeg
            cannot read the video.
    """
    kind = source_type(recording)
    if kind is None:
        raise AudioProcessingError(
            f"{recording.name} is not a supported video or audio file."
        )
    if kind == "audio":
        return recording, None, kind
    return audio.to_wav(recording, wav_path), recording, kind


def estimate_cost(
    seconds: float | None,
    provider: str,
    model: str,
    visual: dict[str, Any] | None = None,
) -> float | None:
    """Estimate what a run will cost before anything is sent.

    Transcription is priced per minute at OpenAI's own figure for the model —
    on the high side for ``gpt-4o-transcribe``, which is billed per token and
    came out cheaper in every recorded run. On-screen context is priced as a
    ceiling: one Full HD screenshot per look, up to the cap; fast screen
    changes can add some, a static screen describes far fewer.

    Args:
        seconds: The length to transcribe, or ``None`` when the file does not
            say.
        provider: The engine.
        model: The API model name, or the local model size.
        visual: On-screen context settings, or ``None`` without that step.

    Returns:
        The estimate in USD; ``None`` when it cannot be made (no length, or a
        screenshot model without a known price).
    """
    if seconds is None:
        return None
    cost = 0.0
    if provider == PROVIDER_OPENAI:
        per_minute = TRANSCRIPTION_PRICE_PER_MINUTE.get(
            model, TRANSCRIPTION_FALLBACK_PRICE_PER_MINUTE
        )
        cost += seconds / 60 * per_minute
    if visual:
        looks = int(seconds // visual["interval"]) + 1
        count = min(looks, get_settings().frame_max_count)
        described = vision.estimate_frame_cost(count, visual["model"], visual["detail"])
        if described is None:
            return None
        cost += described
    return cost
