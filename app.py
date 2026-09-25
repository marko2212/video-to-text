"""Streamlit UI for the video & audio transcription app.

This module is presentation-only: audio preparation lives in :mod:`audio`, the
transcription pipeline in :mod:`transcribe`, configuration in :mod:`config`, and
history persistence in :mod:`db`.

Clicking Start only records a job (in the button's callback); the run that
follows draws every control disabled and then does the work at the end of the
script. Streamlit stops a running script whenever a widget changes, so while a
job runs nothing that could change is left clickable — History lives in a
fragment, whose reruns wait for the job instead of cancelling it. A job records
the thread doing it, because a stopped script keeps running until its next
Streamlit call — possibly a minute later, after a paid request returns.
"""

import importlib.util
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

import audio
import checkpoints
import db
import frames
import transcribe
import vision
from config import (
    AUDIO_FORMATS,
    BROWSER_UNPLAYABLE_AUDIO,
    CHECKPOINT_MAX_AGE_DAYS,
    DEFAULT_FRAME_DETAIL,
    DEFAULT_LOCAL_MODEL,
    FRAME_DETAIL_LEVELS,
    FRAME_INTERVAL_MAX_SECONDS,
    FRAME_INTERVAL_MIN_SECONDS,
    FRAME_INTERVAL_STEP_SECONDS,
    FRAME_MAX_INTERVAL_SECONDS,
    JOB_WATCHDOG_SECONDS,
    LOCAL_MODEL_SIZES_MB,
    LOCAL_MODELS,
    PREVIEW_ABOVE_MB,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDERS,
    SCRATCH_MAX_AGE_HOURS,
    TIMESTAMP_MODELS,
    TRANSCRIPTION_MODELS,
    VIDEO_FORMATS,
    VISION_MODELS,
    get_settings,
)
from exceptions import AppError, IncompleteTranscriptionError, OpenAIAccountError
from logger import get_logger

logger = get_logger(__name__)

# faster-whisper is an optional dependency (install via `uv sync --extra local`).
LOCAL_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None

# Session-state keys describing one upload and its result. Listed once because
# they are initialised, reset on a new upload, and cleared on cleanup — three
# hand-kept copies drifted apart and left new keys uninitialised.
_RUN_STATE_KEYS = (
    "audio_path",
    "video_path",
    "preview_path",
    "transcript_path",
    "srt_path",
    "elapsed_seconds",
    "partial",
    "run_notices",
)
# Keys that live alongside the run state but are not reset by a new upload:
# ``upload_id`` is what detects the new upload, ``job`` is the run request, and
# ``uploader_generation`` is bumped to empty the uploader after a cleanup.
_SESSION_KEYS = (
    *_RUN_STATE_KEYS,
    "original_filename",
    "upload_id",
    "job",
    "uploader_generation",
)

ProgressCallback = Callable[[dict[str, Any]], None]


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load and cache a local faster-whisper model (downloads on first use).

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


def _model_is_cached(model_name: str) -> bool:
    """Return True if the local model is already downloaded on disk."""
    from faster_whisper import download_model

    try:
        download_model(
            model_name,
            cache_dir=str(get_settings().whisper_model_dir),
            local_files_only=True,
        )
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _cached_digest(path: str, size: int, mtime_ns: int) -> str:
    """Hash a file once; size and mtime are part of the key so edits re-hash.

    Args:
        path: File path.
        size: File size in bytes (cache key only).
        mtime_ns: Modification time (cache key only).

    Returns:
        The file's :func:`checkpoints.file_digest`.
    """
    del size, mtime_ns  # cache key only
    return checkpoints.file_digest(Path(path))


def _digest(path: Path) -> str:
    """Return a file's content digest, hashing it only the first time.

    Args:
        path: File to identify.

    Returns:
        The digest used to name checkpoint directories.
    """
    stat = path.stat()
    return _cached_digest(str(path), stat.st_size, stat.st_mtime_ns)


def resolve_openai_key() -> str | None:
    """Return the OpenAI key from settings (.env) or the sidebar field, if any.

    Returns:
        The API key string, or ``None`` if none has been provided.
    """
    return get_settings().openai_api_key or st.session_state.get("openai_api_key_ui")


def render_sidebar(disabled: bool) -> None:
    """Render the sidebar; offer an API key field when none is set via the env.

    Args:
        disabled: True while a job runs, so editing the key cannot stop it.
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        if get_settings().openai_api_key:
            st.success("OpenAI API key loaded from environment.")
        else:
            st.text_input(
                "OpenAI API key",
                type="password",
                key="openai_api_key_ui",
                placeholder="sk-...",
                disabled=disabled,
                help=(
                    "Needed only for the OpenAI API provider. Stored only for "
                    "this session — never written to disk."
                ),
            )
            st.caption(
                "Tip: set `OPENAI_API_KEY` in a `.env` file to load it "
                "automatically every run."
            )


def make_progress_callback(container: Any) -> ProgressCallback:
    """Return a callback that draws pipeline progress into one placeholder.

    The placeholder must be created by the run that uses it. One cached in
    ``session_state`` pointed, from the second run on, at whatever element now
    sat at its old position: the completion message vanished, and after a
    layout change the progress box replaced a widget.

    Args:
        container: An ``st.empty()`` created for this run.

    Returns:
        A callback accepting ``status`` / ``message`` (/ ``progress``) payloads.
    """

    def update(progress_info: dict[str, Any]) -> None:
        status = progress_info["status"]
        with container.container():
            if status in ("info", "start"):
                st.info(progress_info["message"])
            elif status == "progress":
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.progress(progress_info["progress"])
                with col2:
                    st.info(progress_info["message"])
            elif status == "complete":
                st.success(progress_info["message"])
            elif status == "error":
                st.error(progress_info["message"])

    return update


def save_to_history(
    source_type: str, provider: str, model: str, with_timestamps: bool
) -> None:
    """Persist the just-finished transcription into the SQLite history.

    Args:
        source_type: Either ``"audio"`` or ``"video"``.
        provider: Engine used (OpenAI API or local).
        model: Transcription model used.
        with_timestamps: Whether subtitles were generated.
    """
    transcript_path: Path | None = st.session_state.transcript_path
    if not transcript_path or not transcript_path.exists():
        return

    srt_path: Path | None = st.session_state.srt_path
    srt_text = srt_path.read_text(encoding="utf-8") if srt_path else None

    audio_path: Path | None = st.session_state.audio_path
    file_size_mb = None
    if audio_path and audio_path.exists():
        file_size_mb = round(audio_path.stat().st_size / (1024 * 1024), 2)

    db.add_transcription(
        filename=st.session_state.original_filename,
        source_type=source_type,
        model=model,
        provider=provider,
        with_timestamps=with_timestamps,
        transcript=transcript_path.read_text(encoding="utf-8"),
        srt=srt_text,
        audio_path=str(audio_path) if audio_path else None,
        file_size_mb=file_size_mb,
        elapsed_seconds=st.session_state.elapsed_seconds,
    )


def _is_protected(path: Path) -> bool:
    """Return True for a path that holds, or lies inside, data that must survive.

    Args:
        path: An entry of the temp or upload folder.

    Returns:
        True when it is, contains or sits inside the history or model folder
        (possible when DATA_DIR or WHISPER_MODEL_DIR point inside TEMP_DIR).
    """
    settings = get_settings()
    resolved = path.resolve()
    for keep in (settings.data_dir.resolve(), settings.whisper_model_dir.resolve()):
        if resolved.is_relative_to(keep) or keep.is_relative_to(resolved):
            return True
    return False


def clean_temp_files() -> None:
    """Remove working files from temp/ and uploads/ (history DB is untouched).

    A button callback, so it runs before the page is drawn: nothing on the page
    can point at a deleted file, and no mid-script rerun resets widgets drawn
    later (open History entries). Saved parts of unfinished runs live in temp/
    too, so they go as well — which is why nothing is deleted while any
    transcription is running, in this tab or another one.
    """
    if checkpoints.any_active_run():
        st.session_state.clean_message = (
            "warning",
            "A transcription is running (possibly in another tab) — clean up "
            "when it has finished.",
        )
        return
    settings = get_settings()
    try:
        for directory in (settings.temp_dir, settings.upload_dir):
            for path in directory.iterdir():
                if _is_protected(path):
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    except OSError as exc:
        logger.warning("Cleanup failed: %s", exc)
        st.session_state.clean_message = ("error", f"Error during cleanup: {exc}")
        return
    for key in (*_RUN_STATE_KEYS, "original_filename", "upload_id"):
        st.session_state[key] = None
    # Empty the uploader too (a new key is a new, empty widget), or the next
    # click would save and extract the same file again.
    st.session_state.uploader_generation = (
        st.session_state.uploader_generation or 0
    ) + 1
    st.session_state.clean_message = ("success", "Temporary files cleaned.")


def _track_upload(uploaded_file: Any) -> None:
    """Reset the previous file's state when a different upload arrives.

    Keyed on the upload's ``file_id``, not its name: phones and screen
    recorders reuse names, and a new ``call.wav`` used to be ignored in favour
    of the previous one — transcribed, billed and saved under the new upload.

    Args:
        uploaded_file: The Streamlit uploaded file.
    """
    if st.session_state.upload_id != uploaded_file.file_id:
        for key in _RUN_STATE_KEYS:
            st.session_state[key] = None
        st.session_state.upload_id = uploaded_file.file_id
    st.session_state.original_filename = uploaded_file.name


def _safe_stem(filename: str) -> str:
    """Return an upload's name without extension, safe to build paths from.

    Args:
        filename: The uploaded file's name as the browser sent it.

    Returns:
        The stem of :func:`audio.safe_name`.
    """
    return Path(audio.safe_name(filename)).stem


def _make_preview(source: Path, stem: str) -> Path | None:
    """Make a small MP3 for the player; a failure only loses the player.

    Args:
        source: Audio or WAV file to encode.
        stem: File stem for the preview.

    Returns:
        The preview path, or ``None`` if it could not be made.
    """
    try:
        return audio.to_preview(source, get_settings().temp_dir / f"{stem}_preview.mp3")
    except AppError as exc:
        logger.warning("No preview for %s: %s", source, exc)
        return None


def prepare_audio(uploaded_file: Any, is_audio: bool) -> None:
    """Make the uploaded media ready for transcription (run once per file).

    Audio uploads are stored as-is; videos have their audio track extracted.
    Videos, formats browsers cannot play (AMR, WMA, AIFF) and large files also
    get a small MP3 preview for the player. Paths are stored in session state.

    Args:
        uploaded_file: The Streamlit uploaded file.
        is_audio: True if the upload is an audio file.
    """
    if st.session_state.audio_path is not None:
        return
    stem = _safe_stem(uploaded_file.name)
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    try:
        if is_audio:
            source = audio.save_uploaded_file(uploaded_file)
            size_mb = source.stat().st_size / (1024 * 1024)
            if extension in BROWSER_UNPLAYABLE_AUDIO or size_mb > PREVIEW_ABOVE_MB:
                with st.spinner("Preparing a playable preview…", show_time=True):
                    st.session_state.preview_path = _make_preview(source, stem)
            else:
                st.session_state.preview_path = source
            st.session_state.audio_path = source
        else:
            with st.spinner("Extracting audio from video...", show_time=True):
                source = audio.save_uploaded_file(uploaded_file)
                # Kept so on-screen context can go back to the video for frames.
                st.session_state.video_path = source
                wav = audio.to_wav(source, get_settings().temp_dir / f"{stem}.wav")
                st.session_state.preview_path = _make_preview(wav, stem)
                st.session_state.audio_path = wav
    except AppError as exc:
        st.error(f"Error preparing audio: {exc}")


@st.cache_data(show_spinner=False)
def _video_length(video_path: str, size_bytes: int) -> float:
    """Return a video's duration in seconds, cached across reruns.

    Args:
        video_path: Path to the video file.
        size_bytes: File size, part of the cache key so a replaced file is
            probed again rather than reusing a stale duration.

    Returns:
        Duration in seconds, or 0.0 when it cannot be determined.
    """
    del size_bytes  # cache key only
    return frames.video_duration(Path(video_path))


def _format_cost(cost: float | None) -> str:
    """Phrase an estimated cost for a caption.

    Args:
        cost: Estimated cost in USD, or None when the model has no known price.

    Returns:
        A short phrase that reads naturally in parentheses.
    """
    if cost is None:
        return "cost unknown"
    if cost < 0.01:
        return "well under a cent"
    return f"about ${cost:.2f}"


def _format_length(seconds: float) -> str:
    """Render a video duration as ``M:SS`` or ``H:MM:SS``.

    Args:
        seconds: Duration in seconds.

    Returns:
        The duration as a clock string.
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _screenshot_estimate(interval: float, model: str, detail: str) -> str:
    """Describe how many screenshots the current settings are likely to take.

    Args:
        interval: Chosen maximum seconds between screenshots.
        model: Selected vision model.
        detail: Selected image fidelity.

    Returns:
        A caption stating the expected count and cost, and saying so plainly
        when the frame cap overrides the chosen interval.
    """
    video_path: Path | None = st.session_state.get("video_path")
    cap = frames.max_frames_setting()

    duration = 0.0
    if video_path and video_path.exists():
        duration = _video_length(str(video_path), video_path.stat().st_size)

    expected = frames.estimate_frame_count(duration, interval)
    if not expected:
        ceiling = _format_cost(vision.estimate_frame_cost(cap, model, detail))
        return (
            f"At most {cap} screenshots per video ({ceiling}). "
            "The estimate for your video appears once it has been prepared."
        )

    cost = _format_cost(vision.estimate_frame_cost(expected, model, detail))
    length = _format_length(duration)
    actual = frames.effective_interval(duration, interval)
    if actual > interval + 1:
        # The cap is binding, so the chosen interval is not what will happen.
        # Spell out the substitution rather than quietly applying it.
        return (
            f"One every {interval:.0f} s would exceed the {cap}-screenshot limit "
            f"for this {length} video, so they are spread across the whole video "
            f"instead: **{expected}** screenshots, one about every "
            f"{actual:.0f} s ({cost})."
        )
    return (
        f"About **{expected}** screenshots for this {length} video ({cost}), "
        "plus any scene changes. Near-identical frames are discarded before "
        "anything is sent."
    )


def render_visual_options(
    source_type: str, disabled: bool = False
) -> dict[str, Any] | None:
    """Offer on-screen context for video uploads.

    Args:
        source_type: Either ``"audio"`` or ``"video"``.
        disabled: True while a job runs.

    Returns:
        The chosen settings (``model``, ``detail``, ``interval``), or ``None``
        when the feature is switched off or unavailable.
    """
    if source_type != "video":
        return None

    enabled = st.checkbox(
        "🖥️ Describe what's on screen (slides, diagrams, code)",
        value=False,
        disabled=disabled,
        help=(
            "Extracts the frames where the picture changed and has a vision "
            "model describe them. Each note is placed in the transcript at its "
            "time — exactly with whisper-1 or the Local engine, approximately "
            "(~M:SS) with gpt-4o-transcribe. Needs an OpenAI API key even when "
            "transcribing locally."
        ),
    )
    if not enabled:
        return None
    if not resolve_openai_key():
        st.warning(
            "On-screen context needs an OpenAI API key — add one in the sidebar."
        )
        return None

    vision_model = st.selectbox(
        "Vision model",
        options=VISION_MODELS,
        index=0,
        disabled=disabled,
        help="The cheaper model is usually enough to read slide headings.",
    )
    detail = st.radio(
        "Image detail",
        options=FRAME_DETAIL_LEVELS,
        index=FRAME_DETAIL_LEVELS.index(DEFAULT_FRAME_DETAIL),
        horizontal=True,
        disabled=disabled,
        help="Use **high** only when you need to read small text off a slide.",
    )
    interval = st.slider(
        "Screenshot at least every (seconds)",
        min_value=FRAME_INTERVAL_MIN_SECONDS,
        max_value=FRAME_INTERVAL_MAX_SECONDS,
        value=int(FRAME_MAX_INTERVAL_SECONDS),
        step=FRAME_INTERVAL_STEP_SECONDS,
        disabled=disabled,
        help=(
            "How often to grab a frame even when the picture has not changed. "
            "Scene changes are always captured on top of this, and short videos "
            "get extra samples so they are not covered by a single frame. "
            "Near-identical frames are discarded before anything is sent."
        ),
    )

    st.caption(_screenshot_estimate(float(interval), vision_model, detail))
    return {"model": vision_model, "detail": detail, "interval": float(interval)}


def collect_visual_notes(
    visual: dict[str, Any],
    progress_callback: ProgressCallback,
    notices: list[tuple[str, str]],
    stop_on_account_error: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Extract key frames from the uploaded video and describe what they show.

    Failures are reported, and whatever was described before them is kept — it
    was paid for — because a transcript with some on-screen notes is still worth
    having. An account failure (bad key, no credit) is raised instead when the
    transcription itself would use OpenAI and be refused the same way.

    Args:
        visual: Settings from :func:`render_visual_options` (``model``,
            ``detail``, ``interval``).
        progress_callback: Where to report per-frame progress.
        notices: Messages for the user are appended here.
        stop_on_account_error: Raise account errors instead of warning.

    Returns:
        Notes with ``time`` and ``description`` (possibly empty), and whether
        every screenshot was described — only then may the cache be discarded.

    Raises:
        OpenAIAccountError: If OpenAI refuses the key or the account and
            ``stop_on_account_error`` is set.
    """
    video_path: Path | None = st.session_state.video_path
    api_key = resolve_openai_key()
    if not video_path or not api_key:
        return [], True

    # A folder of its own per run: two tabs used to share temp/frames, so one
    # run's extraction replaced the other's screenshots mid-run.
    temp_dir = get_settings().temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix="frames-", dir=temp_dir))
    failed: list[float] = []
    collected: list[dict[str, Any]] = []
    total = 0
    try:
        with st.spinner("Looking for scene changes in the video…", show_time=True):
            keyframes = frames.extract_keyframes(
                video_path, frame_dir, max_interval=visual["interval"]
            )
            cache = vision.cache_dir(
                _digest(video_path), visual["model"], visual["detail"]
            )
        if not keyframes:
            notices.append(
                ("info", "No on-screen changes were detected — nothing to describe.")
            )
            return [], True
        total = len(keyframes)
        with st.spinner(f"Describing {total} screenshots…", show_time=True):
            notes = vision.describe_keyframes(
                keyframes,
                api_key,
                model=visual["model"],
                detail=visual["detail"],
                progress_callback=progress_callback,
                cache=cache,
                failed=failed,
                collected=collected,
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
        notices.append(("warning", _partial_context_notice(exc, collected, total)))
    except (AppError, OSError) as exc:
        notices.append(("warning", _partial_context_notice(exc, collected, total)))
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return sorted(collected, key=lambda note: note["time"]), False


def _partial_context_notice(
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


def _run_pipeline(
    provider: str,
    model: str,
    with_timestamps: bool,
    paths: tuple[Path, Path | None],
    progress: ProgressCallback,
    visual_notes: list[dict[str, Any]] | None,
) -> bool:
    """Run the chosen engine; return False if it could not start.

    Args:
        provider: OpenAI API or local provider.
        model: Selected model (API model name, or local model size).
        with_timestamps: Whether to generate subtitles.
        paths: Transcript path, and the SRT path or ``None``.
        progress: Progress callback for the pipeline.
        visual_notes: On-screen notes to place into the transcript.

    Returns:
        True when the pipeline ran; False when no API key was available.
    """
    transcript_path, srt_path = paths
    if provider == PROVIDER_LOCAL:
        settings = get_settings()
        if _model_is_cached(model):
            spinner_msg = f"Loading model '{model}'…"
        else:
            size = LOCAL_MODEL_SIZES_MB.get(model)
            hint = f" (~{size:.0f} MB)" if size else ""
            spinner_msg = f"Downloading '{model}'{hint} — first run only, please wait…"
        with st.spinner(spinner_msg, show_time=True):
            whisper_model = load_whisper_model(
                model, settings.local_device, settings.local_compute_type
            )
        with st.spinner(
            "Transcribing locally… This may take a while on CPU.", show_time=True
        ):
            transcribe.transcribe_local(
                st.session_state.audio_path,
                transcript_path,
                whisper_model,
                with_timestamps=with_timestamps,
                srt_output_file=srt_path,
                progress_callback=progress,
                visual_notes=visual_notes,
            )
        return True

    api_key = resolve_openai_key()
    if not api_key:
        return False
    with st.spinner("Transcribing with the OpenAI API…", show_time=True):
        transcribe.transcribe_openai(
            st.session_state.audio_path,
            transcript_path,
            api_key,
            model=model,
            with_timestamps=with_timestamps,
            srt_output_file=srt_path,
            progress_callback=progress,
            visual_notes=visual_notes,
        )
    return True


def run_transcription(
    provider: str,
    model: str,
    with_timestamps: bool,
    source_type: str,
    visual: dict[str, Any] | None = None,
) -> None:
    """Run the selected provider's pipeline and store result paths.

    Messages for the user are stored in ``run_notices`` rather than drawn
    directly, because the page reruns as soon as the job ends.

    Args:
        provider: OpenAI API or local provider.
        model: Selected model (API model name, or local model size).
        with_timestamps: Whether to generate timestamps/subtitles.
        source_type: Either ``"audio"`` or ``"video"``.
        visual: On-screen context settings, or None to skip that step.
    """
    settings = get_settings()
    base_name = _safe_stem(st.session_state.original_filename)
    transcript_path = settings.temp_dir / f"transcript_{base_name}.txt"
    srt_path = settings.temp_dir / f"transcript_{base_name}.srt"
    notices: list[tuple[str, str]] = []
    # Drop the previous run's result state before starting. Without this, a run
    # that fails or returns early leaves the old figure on screen, so the result
    # panel claims "Finished in …" for a run that never finished.
    st.session_state.elapsed_seconds = None
    progress = make_progress_callback(st.empty())
    # Timed from here so the reported figure matches the wait the user actually
    # sits through, on-screen context included.
    started = time.monotonic()

    try:
        visual_notes = None
        visual_complete = False
        if visual:
            # An account error stops an OpenAI run (its transcription would be
            # refused the same way) and only warns a Local one.
            visual_notes, visual_complete = collect_visual_notes(
                visual,
                progress,
                notices,
                stop_on_account_error=provider == PROVIDER_OPENAI,
            )

        ran = _run_pipeline(
            provider,
            model,
            with_timestamps,
            (transcript_path, srt_path if with_timestamps else None),
            progress,
            visual_notes,
        )
        if not ran:
            notices.append(
                (
                    "error",
                    "Add your OpenAI API key in the sidebar, or set `OPENAI_API_KEY` "
                    "in a `.env` file. (The Local engine needs no key.)",
                )
            )
        else:
            st.session_state.transcript_path = transcript_path
            st.session_state.srt_path = (
                srt_path if with_timestamps and srt_path.exists() else None
            )
            st.session_state.elapsed_seconds = time.monotonic() - started
            # Cleared only now: a retry that fails before writing anything still
            # shows the earlier partial transcript, and must still say so.
            st.session_state.partial = None
            save_to_history(source_type, provider, model, with_timestamps)
            # Only now: a run stopped after its last chunk has already written
            # the transcript but not saved it, and must be able to resume free.
            if provider == PROVIDER_OPENAI:
                checkpoints.discard(
                    transcribe.checkpoint_dir(
                        _digest(st.session_state.audio_path), model
                    )
                )
            if visual and visual_complete and st.session_state.video_path:
                # Kept only so a failed run need not pay for them again.
                checkpoints.discard(
                    vision.cache_dir(
                        _digest(st.session_state.video_path),
                        visual["model"],
                        visual["detail"],
                    )
                )
    except IncompleteTranscriptionError as exc:
        st.session_state.transcript_path = transcript_path
        st.session_state.srt_path = None
        st.session_state.partial = f"{exc.completed} of {exc.total}"
        hint = (
            " The Local engine needs no key or credit."
            if isinstance(exc.__cause__, OpenAIAccountError) and LOCAL_AVAILABLE
            else ""
        )
        notices.append(
            (
                "error",
                f"Stopped after part {exc.completed} of {exc.total}: {exc}{hint}",
            )
        )
        notices.append(("info", _saved_parts_hint(model, exc.total)))
    except OpenAIAccountError as exc:
        hint = " The Local engine needs no key or credit." if LOCAL_AVAILABLE else ""
        notices.append(("error", f"{exc}{hint}"))
    except AppError as exc:
        notices.append(("error", f"Transcription error: {exc}"))
    st.session_state.run_notices = notices


def render_run_notices() -> None:
    """Show the messages the last run left (errors, warnings, hints)."""
    for level, message in st.session_state.run_notices or []:
        getattr(st, level)(message)


def _request_job(params: dict[str, Any]) -> None:
    """Record a job; the run this click triggers draws the page locked and runs it.

    A callback rather than ``if st.button(...): st.rerun()``: a rerun from the
    middle of the script dropped the state of every widget drawn after Start,
    closing any open History entry.

    Args:
        params: Keyword arguments for :func:`run_transcription`.
    """
    st.session_state.job = {"params": params, "worker": None}
    st.session_state.run_notices = None


def _saved_parts_hint(model: str, total: int | None = None) -> str:
    """Say how much of an unfinished run is really saved.

    Counted on disk rather than promised: the saved parts can be gone (another
    tab cleaned up, a disk error), and then Start would pay for them again.

    Args:
        model: The transcription model of the run.
        total: Parts in the recording, when known.

    Returns:
        One sentence for the notice.
    """
    audio_path: Path | None = st.session_state.audio_path
    saved = 0
    if audio_path and audio_path.exists():
        saved = transcribe.saved_chunk_count(_digest(audio_path), model)
    if not saved:
        return "No finished parts are saved — Start transcribes it from the beginning."
    of_total = f" of {total}" if total else ""
    return (
        f"{saved}{of_total} part(s) are saved — press Start to transcribe only "
        "the rest; saved parts are not sent or paid for again."
    )


def _stopped_notice(job: dict[str, Any]) -> list[tuple[str, str]]:
    """Explain a job whose run was stopped before it finished.

    Args:
        job: The abandoned job.

    Returns:
        The notices to show under Start.
    """
    params = job["params"]
    if params["provider"] == PROVIDER_OPENAI:
        hint = _saved_parts_hint(params["model"])
    else:
        hint = "Press Start to run it again."
    return [("warning", f"The last run was stopped before it finished. {hint}")]


def _settle_job() -> None:
    """At the start of a full run, retire a job whose run is gone.

    A job's worker is the thread that runs it. If that thread has ended without
    clearing the job, its run was stopped (the toolbar's Stop, the rerun
    shortcut, a reconnect). If it is still alive, it is finishing a request it
    was in when stopped — it dies at its next Streamlit call — so the job is
    kept and the controls stay locked until it does; otherwise Start could send
    the same, already paid, chunk a second time.
    """
    job = st.session_state.job
    if job is None or job["worker"] is None or job["worker"].is_alive():
        return
    st.session_state.job = None
    st.session_state.run_notices = _stopped_notice(job)


@st.fragment(run_every=JOB_WATCHDOG_SECONDS)
def _job_watchdog() -> None:
    """Unlock the page once a stopped job's thread has really ended.

    Drawn only once a job has a worker thread. Its body runs inline when drawn
    (the worker is alive then, so it does nothing) and then every couple of
    seconds as a fragment rerun — but a fragment rerun waits while the full
    script is running, so those only happen once the job's run was stopped.
    When the worker thread has ended, a full rerun reports the stop (see
    :func:`_settle_job`) and unlocks the controls.
    """
    job = st.session_state.job
    if job is not None and job["worker"] is not None and not job["worker"].is_alive():
        st.rerun()


def _execute_job(area: Any | None) -> None:
    """Run the requested job, then rerun so the controls come back.

    Called at the very end of the script, after the whole page — History
    included — has been drawn, so nothing is left stale while the job runs.

    Args:
        area: The container under the Start button, or ``None`` when the
            upload the job was for is gone.
    """
    job = st.session_state.job
    if job is None:
        return
    if job["worker"] is not None:
        # A stopped run's thread is still finishing its request; never start
        # the job a second time — wait for it instead.
        if area is not None:
            area.info("Stopping — waiting for the request in progress to finish…")
        _job_watchdog()
        return
    if area is None:
        # Nothing to run it on; do not leave every control disabled.
        st.session_state.job = None
        st.rerun()
    job["worker"] = threading.current_thread()
    _job_watchdog()
    with area, checkpoints.active_run():
        try:
            run_transcription(**job["params"])
        except Exception as exc:
            # Anything unexpected still has to release the disabled controls.
            logger.exception("Run failed")
            st.session_state.run_notices = [("error", f"Unexpected error: {exc}")]
    st.session_state.job = None
    st.rerun()


def render_results(disabled: bool = False) -> None:
    """Render the transcript preview and download buttons from session state.

    Args:
        disabled: True while a job runs — editing the preview is a widget
            change, which would stop the job.
    """
    st.subheader("📄 Result")
    transcript_path: Path | None = st.session_state.transcript_path
    if not transcript_path or not transcript_path.exists():
        st.info("The transcript will appear here after you run a transcription.")
        return

    partial: str | None = st.session_state.partial
    if partial:
        st.warning(
            f"Partial transcript — {partial} parts done; the rest is marked "
            "⚠️ at the end. It is not saved to history until the run finishes."
        )
    elapsed: float | None = st.session_state.elapsed_seconds
    if elapsed is not None:
        st.caption(f"⏱️ Finished in {_format_length(elapsed)}")

    st.text_area(
        "Transcript preview:",
        transcript_path.read_text(encoding="utf-8"),
        height=420,
        disabled=disabled,
    )
    st.download_button(
        label="📥 Download Transcript",
        data=transcript_path.read_bytes,
        file_name=transcript_path.name,
        mime="text/plain",
        on_click="ignore",
    )

    srt_path: Path | None = st.session_state.srt_path
    if srt_path and srt_path.exists():
        st.download_button(
            label="📥 Download Subtitles (.srt)",
            data=srt_path.read_bytes,
            file_name=srt_path.name,
            mime="text/plain",
            on_click="ignore",
        )


def _uploader_label_css() -> None:
    """Replace the uploader's long auto-generated format list with a short label."""
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span::after {
            visibility: visible;
            content: "Video or audio • Limit 2GB";
            position: absolute;
            left: 0;
            top: 0;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_engine_options(disabled: bool) -> tuple[str, str, bool]:
    """Offer the engine, model and timestamp choices.

    Args:
        disabled: True while a job runs.

    Returns:
        The provider, the model and whether subtitles were requested.
    """
    providers = PROVIDERS if LOCAL_AVAILABLE else [PROVIDER_OPENAI]
    provider = st.radio("Engine", providers, horizontal=True, disabled=disabled)

    if provider == PROVIDER_LOCAL:
        model = st.selectbox(
            "Local model",
            options=list(LOCAL_MODELS),
            index=list(LOCAL_MODELS).index(DEFAULT_LOCAL_MODEL),
            format_func=lambda name: f"{name} · {LOCAL_MODELS[name]}",
            disabled=disabled,
            help="Downloaded on first use; runs fully offline, no key.",
        )
        # Local models all return native timestamps.
        with_timestamps = st.checkbox(
            "Include timestamps & generate subtitles (.srt)",
            value=False,
            disabled=disabled,
        )
        return provider, model, with_timestamps

    model = st.selectbox(
        "Transcription model",
        options=TRANSCRIPTION_MODELS,
        index=0,
        disabled=disabled,
        help=(
            "**gpt-4o-transcribe** — newer, more accurate; paragraphs get "
            "approximate (~M:SS) times.\n\n"
            "**whisper-1** — exact timestamps & subtitles (.srt)."
        ),
    )
    if model in TIMESTAMP_MODELS:
        with_timestamps = st.checkbox(
            "Include timestamps & generate subtitles (.srt)",
            value=False,
            disabled=disabled,
        )
    else:
        with_timestamps = False
        st.caption(
            "ℹ️ Exact timestamps & subtitles are available only with the "
            "whisper-1 model."
        )
    return provider, model, with_timestamps


def render_resume_hint(provider: str, model: str) -> None:
    """Say so when an unfinished run of this recording can be continued.

    Args:
        provider: Selected engine.
        model: Selected model.
    """
    audio_path: Path | None = st.session_state.audio_path
    if provider != PROVIDER_OPENAI or not audio_path or not audio_path.exists():
        return
    saved = transcribe.saved_chunk_count(_digest(audio_path), model)
    if saved:
        st.caption(
            f"↻ {saved} part(s) of an unfinished run of this recording are saved — "
            "Start sends only the rest."
        )


def render_transcribe_tab() -> Any | None:
    """Render the main transcription workflow: upload, prepare, transcribe.

    Returns:
        The container under the Start button, where a requested job draws its
        progress, or ``None`` when no file is ready.
    """
    _uploader_label_css()
    running = st.session_state.job is not None
    job_area = None
    left, right = st.columns([2, 3])

    with left:
        uploaded_file = st.file_uploader(
            "Choose video or audio file",
            type=VIDEO_FORMATS + AUDIO_FORMATS,
            key=f"uploader_{st.session_state.uploader_generation or 0}",
            disabled=running,
            help=(
                "**Video**\n\n"
                "- Common: MKV, MP4, MOV, AVI, WebM, M4V\n"
                "- Legacy: WMV, FLV, MPEG, MPG\n"
                "- Mobile: 3GP\n"
                "- TV/streaming: TS, MTS, M2TS\n"
                "- Other: OGV, VOB\n\n"
                "**Audio**\n\n"
                "- MP3, WAV, M4A, AAC, FLAC, OGG, Opus, WMA, AIFF, AMR"
            ),
        )

        if uploaded_file:
            _track_upload(uploaded_file)

            extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
            is_audio = extension in AUDIO_FORMATS
            source_type = "audio" if is_audio else "video"

            st.subheader("1️⃣ Audio")
            prepare_audio(uploaded_file, is_audio)

            preview: Path | None = st.session_state.preview_path
            if preview and preview.exists():
                # st.audio labels everything audio/wav unless told otherwise;
                # browsers sniff raw uploads fine, so only the MP3 previews this
                # app makes get their real type (the OS registry is unreliable).
                mime = "audio/mpeg" if preview.suffix == ".mp3" else "audio/wav"
                st.audio(str(preview), format=mime)
            audio_path: Path | None = st.session_state.audio_path
            if not is_audio and audio_path and audio_path.exists():
                st.download_button(
                    label="📥 Download extracted audio (WAV)",
                    # A callable is read only on click, not on every rerun.
                    data=audio_path.read_bytes,
                    file_name=audio_path.name,
                    mime="audio/wav",
                    on_click="ignore",
                )

            if st.session_state.audio_path:
                st.subheader("2️⃣ Transcription")
                provider, model, with_timestamps = render_engine_options(running)
                visual = render_visual_options(source_type, disabled=running)
                render_resume_hint(provider, model)

                st.button(
                    "Start Transcription",
                    type="primary",
                    disabled=running,
                    on_click=_request_job,
                    args=(
                        {
                            "provider": provider,
                            "model": model,
                            "with_timestamps": with_timestamps,
                            "source_type": source_type,
                            "visual": visual,
                        },
                    ),
                )
                job_area = st.container()
                render_run_notices()

        st.divider()
        st.button(
            "🧹 Clean temporary files",
            disabled=running or checkpoints.any_active_run(),
            on_click=clean_temp_files,
            help=(
                "Deletes uploads, extracted audio and saved parts of unfinished "
                "runs; history is kept. Unavailable while a transcription runs."
            ),
        )
        message = st.session_state.get("clean_message")
        if message:
            getattr(st, message[0])(message[1])
            st.session_state.clean_message = None

    with right:
        render_results(disabled=running)
    return job_area


@st.fragment
def render_history_tab() -> None:
    """List past transcriptions stored in SQLite, with download/delete.

    A fragment, so opening an entry or deleting one reruns only this tab and
    waits for a running job instead of cancelling it. Entries are lazy: a
    transcript is read from the database only while its entry is open, instead
    of all of them on every click anywhere in the app.
    """
    st.subheader("📚 Transcription history")
    records = db.list_transcriptions()
    if not records:
        st.info("No transcriptions yet. Run one in the Transcribe tab.")
        return

    for record in records:
        flag = "⏱️ " if record["with_timestamps"] else ""
        label = (
            f"{flag}{record['filename']} · {record['model']} · {record['created_at']}"
        )
        entry = st.expander(label, key=f"hist_{record['id']}", on_change="rerun")
        if not entry.open:
            continue
        with entry:
            meta_parts = [record["source_type"]]
            if record["provider"]:
                meta_parts.append(record["provider"])
            if record["file_size_mb"]:
                meta_parts.append(f"{record['file_size_mb']} MB")
            if record["elapsed_seconds"]:
                meta_parts.append(f"took {_format_length(record['elapsed_seconds'])}")
            st.caption(" · ".join(meta_parts))

            full = db.get_transcription(record["id"])
            if full is None:
                # Another session deleted it between the list query and this
                # fetch; skip the row rather than blanking the whole tab.
                st.info("This entry was deleted in another session.")
                continue
            base_name = Path(full["filename"]).stem

            st.text_area(
                "Transcript",
                full["transcript"],
                height=200,
                key=f"hist_txt_{record['id']}",
            )
            st.download_button(
                label="📥 Download Transcript",
                data=full["transcript"],
                file_name=f"transcript_{base_name}.txt",
                mime="text/plain",
                key=f"hist_dl_txt_{record['id']}",
                on_click="ignore",
            )
            if full["srt"]:
                st.download_button(
                    label="📥 Download Subtitles (.srt)",
                    data=full["srt"],
                    file_name=f"transcript_{base_name}.srt",
                    mime="text/plain",
                    key=f"hist_dl_srt_{record['id']}",
                    on_click="ignore",
                )
            # A callback, not `if st.button(...)` + a fragment-scoped rerun: the
            # row is gone before the fragment redraws, in any kind of run.
            st.button(
                "🗑️ Delete",
                key=f"hist_del_{record['id']}",
                on_click=db.delete_transcription,
                args=(record["id"],),
            )


def init_session_state() -> None:
    """Initialise the session-state keys used across reruns."""
    for key in _SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Video & Audio Transcription",
        page_icon="📝",
        layout="wide",
    )

    get_settings()  # load settings and create working dirs (API key is optional)
    db.init_db()
    init_session_state()
    if "checkpoints_pruned" not in st.session_state:
        # Saved parts of abandoned runs hold transcript text; do not keep them
        # forever just because nobody pressed "Clean temporary files".
        checkpoints.prune(CHECKPOINT_MAX_AGE_DAYS)
        checkpoints.prune_scratch(SCRATCH_MAX_AGE_HOURS)
        st.session_state.checkpoints_pruned = True
    _settle_job()
    render_sidebar(disabled=st.session_state.job is not None)

    st.title("📝 Video & Audio Transcription")
    st.write("Transcribe video or audio — OpenAI API or a local offline Whisper model")

    tab_transcribe, tab_history = st.tabs(["🎙️ Transcribe", "📚 History"])
    with tab_transcribe:
        job_area = render_transcribe_tab()
    with tab_history:
        render_history_tab()

    st.markdown("---")
    st.markdown("Made with ❤️ by Marko A")
    _execute_job(job_area)


if __name__ == "__main__":
    main()
