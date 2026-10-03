"""Streamlit UI for the video & audio transcription app.

This module is presentation-only: audio preparation lives in :mod:`audio`, the
transcription pipeline in :mod:`transcribe`, configuration in :mod:`config`, and
history persistence in :mod:`db`.

Clicking Start only records a job (in the button's callback); the run that
follows draws every control disabled and then does the work at the end of the
script. Streamlit stops a running script whenever a widget changes, so while a
job runs nothing that could change is left clickable — except Stop, which uses
exactly that — and History lives in a fragment, whose reruns wait for the job
instead of cancelling it. A job records the thread doing it, because a stopped
script keeps running until its next Streamlit call — possibly a minute later,
after a paid request returns.
"""

import importlib.util
import json
import shutil
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
import recordings
import titles
import transcribe
import usage
import vision
from config import (
    AUDIO_FORMATS,
    BROWSER_UNPLAYABLE_AUDIO,
    CHECKPOINT_MAX_AGE_DAYS,
    DEFAULT_FRAME_DETAIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_TITLE_MODEL,
    FRAME_DETAIL_LEVELS,
    FRAME_INTERVAL_MAX_SECONDS,
    FRAME_INTERVAL_MIN_SECONDS,
    FRAME_INTERVAL_STEP_SECONDS,
    FRAME_MAX_INTERVAL_SECONDS,
    FRAME_MIN_INTERVAL_SECONDS,
    JOB_WATCHDOG_SECONDS,
    LOCAL_MODEL_SIZES_MB,
    LOCAL_MODELS,
    PREVIEW_ABOVE_MB,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDERS,
    RECORDING_PROBE_LIMIT,
    SCRATCH_MAX_AGE_HOURS,
    SOURCE_FOLDER,
    SOURCE_UPLOAD,
    SOURCES,
    TIMESTAMP_MODELS,
    TITLE_MODE_OFF,
    TITLE_MODELS,
    TITLE_MODES,
    TITLE_REPEAT_GUARD_SECONDS,
    TRANSCRIPTION_MODEL_NOTES,
    TRANSCRIPTION_MODELS,
    VIDEO_FORMATS,
    VISION_MODELS,
    WORKING_COPY_MAX_AGE_HOURS,
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
    "run_cost",
    "partial",
    "run_notices",
    "run_title",
    "run_row_id",
)
# Keys that live alongside the run state but are not reset by a new upload:
# ``upload_id`` is what detects a new file (an upload's id, or a folder file's
# path) and ``source_version`` the folder file's size and time when it was
# chosen; ``job`` is the run request, and ``uploader_generation`` is bumped to
# empty the uploader after a cleanup.
_SESSION_KEYS = (
    *_RUN_STATE_KEYS,
    "original_filename",
    "upload_id",
    "source_version",
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


def _digest_if_any(path: Path | None) -> str | None:
    """Return a file's digest, or ``None`` when there is no readable file.

    Args:
        path: File to identify, if any.

    Returns:
        :func:`_digest` of the file, or ``None``.
    """
    try:
        return _digest(path) if path else None
    except OSError:
        return None


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
        render_title_settings(disabled)


def _format_size(size: int) -> str:
    """Render a number of bytes as ``1.2 GB`` or ``340.5 MB``.

    Args:
        size: Bytes.

    Returns:
        The size, in GB from 1 GB up.
    """
    megabytes = size / (1024 * 1024)
    if megabytes >= 1024:
        return f"{megabytes / 1024:.1f} GB"
    if megabytes >= 0.1:
        return f"{megabytes:.1f} MB"
    return "under 0.1 MB"


def render_working_files(disabled: bool) -> None:
    """Say how much space the working files take, and offer to delete them now.

    In the sidebar, apart from the steps of a transcription: copies a day old
    go by themselves when the page is opened, so the button is rarely needed,
    and it also deletes saved parts of unfinished runs — nothing to have next
    to Start. Drawn after the page, so the size includes what this run
    prepared (an extracted WAV).

    Args:
        disabled: True while a job runs.
    """
    st.subheader("🧹 Working files")
    size = checkpoints.working_files_size()
    st.caption(f"**{_format_size(size)}** in use." if size else "Nothing to clean.")
    st.button(
        "Clean temporary files",
        disabled=disabled or checkpoints.any_active_run(),
        on_click=clean_temp_files,
        help=(
            "Deletes everything in `temp/` and `uploads/` now: uploads, extracted "
            "audio, transcripts and saved parts of unfinished runs. History is "
            "kept. Uploads and extracted audio older than a day are deleted anyway "
            "when the page is opened. Unavailable while a transcription runs."
        ),
    )
    message = st.session_state.get("clean_message")
    if message:
        getattr(st, message[0])(message[1])
        st.session_state.clean_message = None


# Choices kept across sessions: session-state key (also the widget's key and
# the stored name) → (offered values, or None for free text; default).
_PREFERENCES: dict[str, tuple[list[str] | None, str]] = {
    "title_mode": (list(TITLE_MODES), TITLE_MODE_OFF),
    "title_model": (TITLE_MODELS, DEFAULT_TITLE_MODEL),
    "source": (list(SOURCES), SOURCE_UPLOAD),
    "recordings_folder": (None, ""),
}


def _preference(key: str) -> str:
    """Return a kept choice, loading the stored one when the session has none.

    Args:
        key: A key of ``_PREFERENCES``.

    Returns:
        The value in session state; a stored value no longer offered (a model
        since removed) falls back to the default.
    """
    options, default = _PREFERENCES[key]
    value = st.session_state.get(key)
    if not isinstance(value, str) or (options is not None and value not in options):
        value = db.get_preferences().get(key)
        if value is None or (options is not None and value not in options):
            value = default
        st.session_state[key] = value
    return value


def _title_preferences() -> tuple[str, str]:
    """Return the AI title mode and model, loading the stored ones once a session.

    Returns:
        The mode (a ``TITLE_MODES`` key) and the chat model.
    """
    return _preference("title_mode"), _preference("title_model")


def _save_preference(key: str) -> None:
    """Store a sidebar preference as soon as it changes (a widget callback).

    Args:
        key: The widget's session-state key, also the preference name.
    """
    db.set_preference(key, st.session_state[key])


def render_title_settings(disabled: bool) -> None:
    """Offer the AI title mode and model; both are kept across sessions.

    Args:
        disabled: True while a job runs, so a change cannot stop it.
    """
    _title_preferences()
    st.subheader("🏷️ AI title")
    mode = st.radio(
        "Name each transcript from its content",
        options=list(TITLE_MODES),
        format_func=TITLE_MODES.get,
        key="title_mode",
        on_change=_save_preference,
        args=("title_mode",),
        disabled=disabled,
        help=(
            "After each run a chat model reads the transcript and writes a short "
            "title. **Replace** shows the title instead of the file name; **Add** "
            "shows `file name - title`. Used in History and for downloaded files; "
            "the original file name is kept, so changing this applies to earlier "
            "transcripts too."
        ),
    )
    model = st.selectbox(
        "Title model",
        options=TITLE_MODELS,
        key="title_model",
        on_change=_save_preference,
        args=("title_model",),
        disabled=disabled or mode == TITLE_MODE_OFF,
    )
    if mode == TITLE_MODE_OFF:
        return
    per_hour = titles.estimate_cost_per_hour(model)
    cost = ""
    if per_hour:
        cost = f" About {usage.format_usd(per_hour)} per hour of recording."
    st.caption(
        f"Uses your OpenAI key, after OpenAI and Local runs alike.{cost} Counted "
        "in the run's cost."
    )
    if not resolve_openai_key():
        st.warning("An AI title needs an OpenAI API key.")


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
) -> int | None:
    """Persist the just-finished transcription into the SQLite history.

    Args:
        source_type: Either ``"audio"`` or ``"video"``.
        provider: Engine used (OpenAI API or local).
        model: Transcription model used.
        with_timestamps: Whether subtitles were generated.

    Returns:
        The new row's id, or ``None`` when there was no transcript to save.
    """
    transcript_path: Path | None = st.session_state.transcript_path
    if not transcript_path or not transcript_path.exists():
        return None
    cost: dict[str, Any] | None = st.session_state.get("run_cost")

    srt_path: Path | None = st.session_state.srt_path
    srt_text = srt_path.read_text(encoding="utf-8") if srt_path else None

    audio_path: Path | None = st.session_state.audio_path
    file_size_mb = None
    if audio_path and audio_path.exists():
        file_size_mb = round(audio_path.stat().st_size / (1024 * 1024), 2)

    return db.add_transcription(
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
        cost_usd=cost["cost_usd"] if cost else None,
        usage_json=json.dumps(cost) if cost else None,
        title=st.session_state.get("run_title"),
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
    for key in (*_RUN_STATE_KEYS, "original_filename", "upload_id", "source_version"):
        st.session_state[key] = None
    # Empty the uploader and the folder pick too (a new uploader key is a new,
    # empty widget), or the next click would extract the same file again.
    st.session_state.uploader_generation = (
        st.session_state.uploader_generation or 0
    ) + 1
    st.session_state.folder_file = None
    st.session_state.clean_message = ("success", "Temporary files cleaned.")


def _track_source(
    source_id: str, name: str, version: tuple[int, int] | None = None
) -> None:
    """Reset the previous file's state when a different file is chosen.

    Keyed on the upload's ``file_id`` (or a folder file's path), not its name:
    phones and screen recorders reuse names, and a new ``call.wav`` used to be
    ignored in favour of the previous one — transcribed, billed and saved under
    the new upload.

    Args:
        source_id: What identifies the chosen file.
        name: Its file name.
        version: A folder file's size and time when chosen, to notice a change.
    """
    if st.session_state.upload_id != source_id:
        for key in _RUN_STATE_KEYS:
            st.session_state[key] = None
        st.session_state.upload_id = source_id
        st.session_state.source_version = version
    st.session_state.original_filename = name


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


def prepare_audio(name: str, is_audio: bool, obtain: Callable[[], Path]) -> None:
    """Make the chosen media ready for transcription (run once per file).

    Audio files are used as they are; videos have their audio track extracted.
    Videos, formats browsers cannot play (AMR, WMA, AIFF) and large files also
    get a small MP3 preview for the player. Paths are stored in session state.

    Args:
        name: The file's name.
        is_audio: True if the file is an audio file.
        obtain: Returns the file on disk: an upload is saved to ``uploads/``
            first, a file from a folder is used where it lies.
    """
    prepared: Path | None = st.session_state.audio_path
    video: Path | None = st.session_state.video_path
    if (
        prepared is not None
        and prepared.exists()
        and (is_audio or (video is not None and video.exists()))
    ):
        return
    # Not prepared yet, or a working copy was deleted meanwhile (they are
    # cleared after a day — an upload's copy can go a little before its WAV):
    # prepare it again from the source.
    stem = _safe_stem(name)
    extension = Path(name).suffix.lower().lstrip(".")
    try:
        if is_audio:
            source = obtain()
            size_mb = source.stat().st_size / (1024 * 1024)
            if extension in BROWSER_UNPLAYABLE_AUDIO or size_mb > PREVIEW_ABOVE_MB:
                with st.spinner("Preparing a playable preview…", show_time=True):
                    st.session_state.preview_path = _make_preview(source, stem)
            else:
                st.session_state.preview_path = source
            st.session_state.audio_path = source
        else:
            with st.spinner("Extracting audio from video...", show_time=True):
                source = obtain()
                # Kept so on-screen context can go back to the video for frames.
                st.session_state.video_path = source
                wav = audio.to_wav(source, get_settings().temp_dir / f"{stem}.wav")
                st.session_state.preview_path = _make_preview(wav, stem)
                st.session_state.audio_path = wav
    except (AppError, OSError) as exc:
        # OSError: the source itself is gone (moved or deleted meanwhile).
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


@st.cache_data(show_spinner=False)
def _video_frame_size(video_path: str, size_bytes: int) -> tuple[int, int]:
    """Return a video's width and height, cached across reruns.

    Args:
        video_path: Path to the video file.
        size_bytes: File size, part of the cache key (see :func:`_video_length`).

    Returns:
        ``(width, height)``, or ``(0, 0)`` when it cannot be determined.
    """
    del size_bytes  # cache key only
    return frames.video_frame_size(Path(video_path))


def _format_ceiling(cost: float | None) -> str:
    """Phrase the most a number of screenshots can cost, for a caption.

    Args:
        cost: Estimated cost in USD, or None when the model has no known price.

    Returns:
        E.g. ``up to $0.07``, its dollar sign escaped for Markdown: a caption
        with two of them showed the text between as a formula, without the
        signs. ``under a cent`` for less.
    """
    if cost is None:
        return "cost unknown"
    if cost < 0.01:
        return "under a cent"
    return f"up to \\${cost:.2f}"


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


def _scan_dir(video_path: Path) -> Path:
    """Return where the scan of a video is kept (named after its content).

    Args:
        video_path: The uploaded video.

    Returns:
        A folder under ``temp_dir``, removed by clean-up or after a day.
    """
    return get_settings().temp_dir / f"scan-{_digest(video_path)}"


def _scanned_estimate(
    scan: dict[str, Any], interval: float, model: str, detail: str
) -> str:
    """Say exactly how many screenshots a run will describe, and the cost.

    Args:
        scan: The video's scan.
        interval: Chosen maximum seconds between screenshots.
        model: Selected vision model.
        detail: Selected image fidelity.

    Returns:
        A caption with the count the run will use — the same selection.
    """
    cap = frames.max_frames_setting()
    # select_from_scan is exactly these candidates capped; the run calls it.
    candidates = frames.candidates_from_scan(scan, interval)
    chosen = frames.cap_frame_count(candidates, cap)
    tokens = (
        vision.frame_tokens(scan["width"], scan["height"]) if scan["width"] else None
    )
    cost = _format_cost(vision.estimate_frame_cost(len(chosen), model, detail, tokens))
    length = _format_length(scan["duration"])
    if len(candidates) > len(chosen):
        return (
            f"One every {interval:.0f} s would give {len(candidates)} screenshots, "
            f"over the {cap}-screenshot limit, so **{len(chosen)}** will be "
            f"described, spread evenly over the video ({cost}) — counted from a "
            f"scan of this {length} video."
        )
    used = frames.effective_interval(scan["duration"], interval, max_frames=0)
    return (
        f"**{len(chosen)}** screenshots will be described ({cost}) — counted from "
        f"a scan of this {length} video: every screen change, plus a look every "
        f"{used:.0f} s when nothing changed, with repeats of the same picture "
        "left out."
    )


def _screenshot_estimate(
    interval: float, model: str, detail: str, scan: dict[str, Any] | None = None
) -> str:
    """Describe how many screenshots the current settings will take.

    Args:
        interval: Chosen maximum seconds between screenshots.
        model: Selected vision model.
        detail: Selected image fidelity.
        scan: The video's scan; exact when given, a rough guess otherwise.

    Returns:
        A caption stating the count and cost, and saying so plainly when the
        frame cap overrides the chosen interval.
    """
    if scan is not None:
        return _scanned_estimate(scan, interval, model, detail)
    video_path: Path | None = st.session_state.get("video_path")
    cap = frames.max_frames_setting()

    duration = 0.0
    # Frames are described at the video's own size, so a 1440p or 4K screen
    # costs more per screenshot than the Full HD assumed without it.
    tokens: int | None = None
    ready = bool(video_path and video_path.exists())
    if ready:
        size_bytes = video_path.stat().st_size
        duration = _video_length(str(video_path), size_bytes)
        width, height = _video_frame_size(str(video_path), size_bytes)
        if width and height:
            tokens = vision.frame_tokens(width, height)

    def most(count: int) -> str:
        return _format_ceiling(vision.estimate_frame_cost(count, model, detail, tokens))

    if duration <= 0:
        if ready:
            return (
                f"At most **{cap}** screenshots per video ({most(cap)}) — this file "
                "does not say how long it is."
            )
        return (
            f"At most **{cap}** screenshots per video ({most(cap)}). The count for "
            "your video appears once it has been prepared."
        )

    length = _format_length(duration)
    fewer = (
        "Usually far fewer: a picture that repeats the one before is not "
        "described again. The exact number shows during the run."
    )
    # One per look at the screen, if every look shows a new picture.
    used = frames.effective_interval(duration, interval, max_frames=0)
    looks = int(duration // used) + 1
    if looks > cap:
        # The cap is binding, so the chosen interval is not what will happen.
        # Spell out the substitution rather than quietly applying it.
        return (
            f"At most **{cap}** screenshots ({most(cap)}), the limit per video: one "
            f"every {interval:.0f} s would give {looks} for this {length} video, "
            f"so they are spread evenly over it. {fewer}"
        )
    text = (
        f"At most **{looks}** screenshots ({most(looks)}) — one every {used:.0f} s of "
        f"this {length} video, if each look shows a new picture."
    )
    # A screen that changes faster is caught too, one picture per burst window.
    busiest = min(cap, int(duration // FRAME_MIN_INTERVAL_SECONDS) + 1)
    if busiest > looks:
        text += (
            " A screen that changes faster (scrolling, a video) can add more, up "
            f"to {busiest} ({most(busiest)})."
        )
    return f"{text} {fewer}"


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
    video_path: Path | None = st.session_state.get("video_path")
    ready = bool(video_path and video_path.exists())
    if not enabled:
        return None
    if not resolve_openai_key():
        st.warning(
            "On-screen context needs an OpenAI API key — add one in the sidebar."
        )
        return None
    # The video is scanned by the run, not here: ticking the box used to wait
    # for the scan (about a minute and a half per hour of video) before the
    # slider appeared. The caption gives a ceiling instead. While a job runs,
    # a scan stored by an earlier run of this video gives the exact count.
    scan: dict[str, Any] | None = None
    if ready and disabled:
        scan = frames.load_scan(_scan_dir(video_path))

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
        help=(
            "Measured with the gpt-5.4 models: **low** and **high** use the same "
            "number of tokens, so they cost the same."
        ),
    )
    interval = st.slider(
        "Screenshot at least every (seconds)",
        min_value=FRAME_INTERVAL_MIN_SECONDS,
        max_value=FRAME_INTERVAL_MAX_SECONDS,
        value=int(FRAME_MAX_INTERVAL_SECONDS),
        step=FRAME_INTERVAL_STEP_SECONDS,
        disabled=disabled,
        help=(
            "How often the screen is looked at, even when nothing changed. A "
            "change of screen noticed in between is taken too (at most one per "
            "2 s), and a look that shows the same picture as the one before is "
            "not sent. A smaller number catches short slides and puts their notes "
            "closer to when they appeared; it costs more only when the screen "
            "really changes more often."
        ),
    )

    st.caption(_screenshot_estimate(float(interval), vision_model, detail, scan))
    return {"model": vision_model, "detail": detail, "interval": float(interval)}


def collect_visual_notes(
    visual: dict[str, Any],
    progress_callback: ProgressCallback,
    notices: list[tuple[str, str]],
    stop_on_account_error: bool,
    spent: list[dict[str, Any]] | None = None,
    used: list[str] | None = None,
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
        spent: Usage records of the descriptions are appended here.
        used: The cache names of the selected frames are appended here, so
            the caller can mark exactly those as counted.

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

    failed: list[float] = []
    collected: list[dict[str, Any]] = []
    total = 0
    try:
        with st.spinner(
            "Looking through the video for screen changes — about a minute and a "
            "half per hour of video…",
            show_time=True,
        ):
            # The same scan and selection the caption counted, so the number
            # described is the number that was shown.
            scan = frames.scan_video(video_path, _scan_dir(video_path))
            keyframes = frames.select_from_scan(scan, visual["interval"])
            if used is not None:
                used.extend(vision.frame_name(frame["time"]) for frame in keyframes)
            cache = vision.cache_dir(
                _digest(video_path), visual["model"], visual["detail"]
            )
        if not keyframes:
            notices.append(
                ("info", "No on-screen changes were detected — nothing to describe.")
            )
            return [], True
        total = len(keyframes)
        with (
            st.spinner(f"Describing {total} screenshots…", show_time=True),
            frames.pictures(video_path) as picture,
        ):
            notes = vision.describe_keyframes(
                keyframes,
                api_key,
                model=visual["model"],
                detail=visual["detail"],
                progress_callback=progress_callback,
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
        notices.append(("warning", _partial_context_notice(exc, collected, total)))
    except (AppError, OSError) as exc:
        notices.append(("warning", _partial_context_notice(exc, collected, total)))
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
) -> tuple[bool, dict[str, Any] | None]:
    """Run the chosen engine.

    Args:
        provider: OpenAI API or local provider.
        model: Selected model (API model name, or local model size).
        with_timestamps: Whether to generate subtitles.
        paths: Transcript path, and the SRT path or ``None``.
        progress: Progress callback for the pipeline.
        visual_notes: On-screen notes to place into the transcript.

    Returns:
        Whether the pipeline ran (False when no API key was available), and
        what its OpenAI requests used (``None`` for the local engine).
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
        return True, None

    api_key = resolve_openai_key()
    if not api_key:
        return False, None
    with st.spinner("Transcribing with the OpenAI API…", show_time=True):
        spent = transcribe.transcribe_openai(
            st.session_state.audio_path,
            transcript_path,
            api_key,
            model=model,
            with_timestamps=with_timestamps,
            srt_output_file=srt_path,
            progress_callback=progress,
            visual_notes=visual_notes,
        )
    return True, spent


def _run_cost(
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
    vision = usage.summarize(vision_records) if vision_records else None
    title = usage.summarize(title_records) if title_records else None
    unknown = (
        provider == PROVIDER_OPENAI and transcription_ran and transcription is None
    )
    parts = [part for part in (transcription, vision, title) if part]
    return {
        "provider": provider,
        "transcription": transcription,
        "vision": vision,
        "title": title,
        "cost_usd": sum(part["cost_usd"] for part in parts),
        "estimated": unknown or any(part["estimated"] for part in parts),
    }


def _note_spent_before_failure(
    provider: str,
    vision_spent: list[dict[str, Any]],
    notices: list[tuple[str, str]],
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
        cost = _run_cost(
            provider,
            transcription_spent,
            vision_spent,
            transcription_ran=transcription_spent is not None,
        )
        notices.append(("info", _describe_cost({**cost, "partial": True})))


def _make_title(
    transcript_path: Path,
    progress: ProgressCallback,
    notices: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Name a finished transcript when the AI title setting asks for it.

    Runs after either engine: a Local run sends its transcript too. A title
    that cannot be made costs the run only a warning — the transcript is
    already paid for and is saved without one.

    Args:
        transcript_path: The finished transcript.
        progress: Progress callback of the run.
        notices: Messages for the user are appended here.

    Returns:
        The usage records of the title request (empty when none was sent).
    """
    mode, model = _title_preferences()
    if mode == TITLE_MODE_OFF:
        return []
    api_key = resolve_openai_key()
    if not api_key:
        notices.append(
            ("info", "No AI title: it needs an OpenAI API key (sidebar or `.env`).")
        )
        return []
    progress({"status": "info", "message": "Writing a title…"})
    try:
        title, records = titles.make_title(
            transcript_path.read_text(encoding="utf-8"), api_key, model
        )
    except AppError as exc:
        notices.append(("warning", f"No AI title: {exc}"))
        return list(getattr(exc, "usage_records", None) or [])
    st.session_state.run_title = title
    return records


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
    st.session_state.run_cost = None
    st.session_state.run_title = None
    st.session_state.run_row_id = None
    vision_spent: list[dict[str, Any]] = []
    used_frames: list[str] = []
    progress = make_progress_callback(st.empty())
    # Timed from here so the reported figure matches the wait the user actually
    # sits through, on-screen context included.
    started = time.monotonic()

    try:
        # Taken before the run: a recording from a folder that is still being
        # written would hash differently at the end, and what is discarded then
        # must be what this run saved.
        audio_digest = video_digest = None
        if provider == PROVIDER_OPENAI:
            audio_digest = _digest_if_any(st.session_state.audio_path)
        if visual:
            video_digest = _digest_if_any(st.session_state.video_path)
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
                spent=vision_spent,
                used=used_frames,
            )

        ran, transcription_spent = _run_pipeline(
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
            title_spent = _make_title(transcript_path, progress, notices)
            st.session_state.elapsed_seconds = time.monotonic() - started
            # Cleared only now: a retry that fails before writing anything still
            # shows the earlier partial transcript, and must still say so.
            st.session_state.partial = None
            cache = None
            if visual and video_digest:
                cache = vision.cache_dir(
                    video_digest, visual["model"], visual["detail"]
                )
                if visual_complete:
                    # The cache is discarded below. Descriptions an earlier,
                    # unsaved attempt paid for outside this selection would then
                    # count in no row, so this one takes them.
                    vision_spent += vision.unbilled_usage(
                        cache, visual["model"], used_frames
                    )
            st.session_state.run_cost = _run_cost(
                provider, transcription_spent, vision_spent, title_records=title_spent
            )
            row_id = save_to_history(source_type, provider, model, with_timestamps)
            # Only now: a run stopped after its last chunk has already written
            # the transcript but not saved it, and must be able to resume free.
            # And straight after the row, with no Streamlit call in between: each
            # one (session state included) is where a Stop takes effect, and a
            # Stop between the row and the discard would let Start save the
            # same transcript as a second row.
            if audio_digest:
                checkpoints.discard(transcribe.checkpoint_dir(audio_digest, model))
            if cache is not None:
                if visual_complete:
                    checkpoints.discard(cache)
                else:
                    # Kept so a rerun pays only for the missing screenshots;
                    # this row has counted the ones it used.
                    vision.mark_billed(cache, used_frames)
            st.session_state.run_row_id = row_id
    except IncompleteTranscriptionError as exc:
        st.session_state.transcript_path = transcript_path
        st.session_state.srt_path = None
        st.session_state.partial = f"{exc.completed} of {exc.total}"
        st.session_state.run_cost = {
            **_run_cost(provider, exc.spent, vision_spent),
            "partial": True,
        }
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
        _note_spent_before_failure(provider, vision_spent, notices, exc.spent)
    except AppError as exc:
        notices.append(("error", f"Transcription error: {exc}"))
        _note_spent_before_failure(provider, vision_spent, notices, exc.spent)
    st.session_state.run_notices = notices


def render_run_notices() -> None:
    """Show the messages the last run left (errors, warnings, hints)."""
    for level, message in st.session_state.run_notices or []:
        getattr(st, level)(message)


def _request_job() -> None:
    """Record a job; the run this click triggers draws the page locked and runs it.

    A callback rather than ``if st.button(...): st.rerun()``: a rerun from the
    middle of the script dropped the state of every widget drawn after Start,
    closing any open History entry. The settings are not captured here: a
    Start clicked while the page was still busy (scanning the video) carried
    the settings of the run before, without on-screen context. The job run
    reads them from the widgets it draws itself.
    """
    st.session_state.job = {"params": None, "worker": None}
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
    if params and params["provider"] == PROVIDER_OPENAI:
        hint = _saved_parts_hint(params["model"])
    else:
        hint = "Press Start to run it again."
    return [("warning", f"The last run was stopped before it finished. {hint}")]


def _request_stop() -> None:
    """Ask the running job to stop (the Stop button's callback).

    The click is a rerun request, which stops the job's script at its next
    Streamlit call — after the request in flight has returned and been saved.
    With ``runner.fastReruns`` (Streamlit's default) the rerun starts at once
    in a new thread while the old one finishes: the page stays locked, says
    "Stopping — waiting…", and the watchdog retires the job once that thread
    has ended — the toolbar Stop's path. Without it, the rerun runs in the
    job's own thread after the stop, and :func:`_settle_job` retires the job
    there instead of letting the rerun start it again. A click that arrives
    after the job has finished is ignored.
    """
    if st.session_state.job is not None:
        st.session_state.stop_requested = True


def _owns_job() -> bool:
    """Return True when this run is the one doing the requested job."""
    job = st.session_state.job
    return job is not None and job["worker"] is threading.current_thread()


def _settle_job() -> None:
    """At the start of a full run, retire a job whose run is gone.

    A job's worker is the thread that runs it. If the Stop button's rerun runs
    in that same thread (``runner.fastReruns`` off), the job is retired here.
    If the thread has ended without clearing the job, its run was stopped (the
    Stop button or the toolbar's Stop with fast reruns, a reconnect). If it is
    still alive in another run, it is finishing a request it was in when
    stopped — it dies at its next Streamlit call — so the job is kept and the
    controls stay locked until it does; otherwise Start could send the same,
    already paid, chunk a second time.
    """
    job = st.session_state.job
    if st.session_state.pop("stop_requested", False) and job is not None:
        worker = job["worker"]
        if worker is None or worker is threading.current_thread():
            # The Stop button: its click reran the script in the job's own
            # thread, which is here now, so the job is no longer running.
            st.session_state.job = None
            st.session_state.run_notices = _stopped_notice(job)
            return
    if job is None or job["worker"] is None or job["worker"].is_alive():
        return
    st.session_state.job = None
    st.session_state.run_notices = _stopped_notice(job)


@st.fragment(run_every=JOB_WATCHDOG_SECONDS)
def _job_watchdog() -> None:
    """Unlock the page once a stopped job's thread has really ended.

    Drawn only once a job has a worker thread (see :func:`_claim_job`). Its
    body runs inline when drawn
    (the worker is alive then, so it does nothing) and then every couple of
    seconds as a fragment rerun — but a fragment rerun waits while the full
    script is running, so those only happen once the job's run was stopped.
    When the worker thread has ended, a full rerun reports the stop (see
    :func:`_settle_job`) and unlocks the controls.
    """
    job = st.session_state.job
    if job is not None and job["worker"] is not None and not job["worker"].is_alive():
        st.rerun()


def _claim_job() -> None:
    """Make this run the owner of a newly requested job, from its first line.

    The owner used to be recorded only when the job started, at the end of the
    page. A Stop pressed while the page above it was still being drawn left a
    job with no owner: nothing ever unlocked the page, and the next rerun ran
    the stopped job. Owned from the start, a stopped run leaves a dead worker,
    which the watchdog and :func:`_settle_job` clean up.
    """
    job = st.session_state.job
    if job is None:
        return
    if job["worker"] is None:
        job["worker"] = threading.current_thread()
    _job_watchdog()


def _execute_job(area: Any | None, params: dict[str, Any] | None) -> None:
    """Run the requested job, then rerun so the controls come back.

    Called at the very end of the script, after the whole page — History
    included — has been drawn, so nothing is left stale while the job runs.

    Args:
        area: The container under the Start button, or ``None`` when the
            upload the job was for is gone.
        params: Keyword arguments for :func:`run_transcription`, read from the
            widgets this run drew, or ``None`` with ``area``.
    """
    job = st.session_state.job
    if job is None:
        return
    if job["worker"] is not threading.current_thread():
        # A stopped run's thread is still finishing its request; never start
        # the job a second time — wait for it instead.
        if area is not None:
            area.info("Stopping — waiting for the request in progress to finish…")
        return
    if area is None or params is None:
        # Nothing to run it on; do not leave every control disabled.
        st.session_state.job = None
        st.rerun()
    job["params"] = params
    with area, checkpoints.active_run():
        try:
            run_transcription(**params)
        except Exception as exc:
            # Anything unexpected still has to release the disabled controls.
            logger.exception("Run failed")
            st.session_state.run_notices = [("error", f"Unexpected error: {exc}")]
    st.session_state.job = None
    st.rerun()


def _describe_cost(cost: dict[str, Any]) -> str:
    """Phrase what a run cost and what it used, for a caption.

    Args:
        cost: The run's cost from :func:`_run_cost`.

    Returns:
        E.g. ``💵 $0.0071 — transcription 1 request, 0:56 of audio, 1,210 → 240
        tokens; screenshots 4 described, 2,580 → 190 tokens``.
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
            f"{_format_length(audio_seconds)} of audio{tokens}"
        )
    elif cost.get("provider") == PROVIDER_LOCAL:
        parts.append("transcription free (local)")
    vision = cost.get("vision")
    if vision:
        parts.append(
            f"screenshots {vision['requests']} described, "
            f"{vision['input_tokens']:,} → {vision['output_tokens']:,} tokens"
        )
    title = cost.get("title")
    if title:
        parts.append(
            f"title {title['input_tokens']:,} → {title['output_tokens']:,} tokens"
        )
    total = usage.format_usd(cost["cost_usd"], cost.get("estimated", False))
    text = f"💵 {total} — " + "; ".join(parts) if parts else f"💵 {total}"
    if cost.get("partial"):
        text += " — spent so far, not saved to history."
        if transcription:
            text += (
                " Start again with the same model to transcribe only the rest; "
                "that run counts this in its total."
            )
        if vision:
            text += (
                " The screenshot descriptions are kept: the next run with the same "
                "screenshot model and detail reuses them and counts them in its total."
            )
    return text


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
    cost: dict[str, Any] | None = st.session_state.get("run_cost")
    if cost is not None:
        st.caption(_describe_cost(cost))
    title: str | None = st.session_state.get("run_title")
    if title:
        st.caption(f"🏷️ AI title: {title}")

    st.text_area(
        "Transcript preview:",
        transcript_path.read_text(encoding="utf-8"),
        height=420,
        disabled=disabled,
    )
    filename: str | None = st.session_state.original_filename
    stem = _download_stem(filename, title) if filename else transcript_path.stem
    st.download_button(
        label="📥 Download Transcript",
        data=transcript_path.read_bytes,
        file_name=f"{stem}.txt",
        mime="text/plain",
        on_click="ignore",
    )

    srt_path: Path | None = st.session_state.srt_path
    if srt_path and srt_path.exists():
        st.download_button(
            label="📥 Download Subtitles (.srt)",
            data=srt_path.read_bytes,
            file_name=f"{stem}.srt",
            mime="text/plain",
            on_click="ignore",
        )


def _download_stem(filename: str, title: str | None) -> str:
    """Return the name, without extension, a transcript is downloaded under.

    Args:
        filename: The uploaded file's original name.
        title: The transcript's AI title, if it has one.

    Returns:
        Per the title mode, the title or ``file name - title``; otherwise
        ``transcript_<file name>``, as before titles existed.
    """
    stem = _safe_stem(filename)
    mode, _ = _title_preferences()
    return titles.display_name(stem, title, mode) or f"transcript_{stem}"


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
            "Local Whisper model",
            options=list(LOCAL_MODELS),
            index=list(LOCAL_MODELS).index(DEFAULT_LOCAL_MODEL),
            format_func=lambda name: f"{name} · {LOCAL_MODELS[name]}",
            disabled=disabled,
            help=(
                "Every option is OpenAI's open-source **Whisper**, run on this "
                "computer by faster-whisper; they differ in size. Bigger is more "
                "accurate but slower. Downloaded on first use; runs fully offline, "
                "no key, no cost."
            ),
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
        format_func=lambda name: f"{name} · {TRANSCRIPTION_MODEL_NOTES[name]}",
        disabled=disabled,
        help=(
            "**gpt-4o-transcribe** — newer (2025), more accurate and usually "
            "cheaper: billed per token, about $0.003–0.005 per minute of speech. "
            "Paragraphs get approximate (~M:SS) times.\n\n"
            "**whisper-1** — older (2023), a fixed $0.006 per minute; exact "
            "timestamps & subtitles (.srt)."
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


# What the uploader accepts, for its help.
_FORMATS_HELP = (
    "**Video**\n\n"
    "- Common: MKV, MP4, MOV, AVI, WebM, M4V\n"
    "- Legacy: WMV, FLV, MPEG, MPG\n"
    "- Mobile: 3GP\n"
    "- TV/streaming: TS, MTS, M2TS\n"
    "- Other: OGV, VOB\n\n"
    "**Audio**\n\n"
    "- MP3, WAV, M4A, AAC, FLAC, OGG, Opus, WMA, AIFF, AMR"
)

# A chosen file: what identifies it, its name, a function that returns it on
# disk (saving an upload first), and its version on disk (a folder file only).
ChosenFile = tuple[str, str, Callable[[], Path], tuple[int, int] | None]
# Server addresses that only this computer can reach.
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _local_files_allowed() -> bool:
    """Return True when the page may offer recordings from this computer's folders.

    Only while the app listens on this machine alone: started with
    ``--server.address=0.0.0.0`` (as in Docker), anyone who can open the page
    could browse the disk and transcribe what is on it with the owner's key.

    Returns:
        Whether ``ALLOW_LOCAL_FILES`` is on and the server is loopback-only.
    """
    address = st.get_option("server.address")
    return get_settings().allow_local_files and address in _LOOPBACK


def _choose_source(disabled: bool) -> ChosenFile | None:
    """Offer an upload, or a recording from a folder on this computer.

    Args:
        disabled: True while a job runs.

    Returns:
        The chosen file, or ``None`` while there is none.
    """
    local_files = _local_files_allowed()
    source = SOURCE_UPLOAD
    if local_files:
        _preference("source")
        source = st.radio(
            "Source",
            options=list(SOURCES),
            format_func=SOURCES.get,
            key="source",
            on_change=_save_preference,
            args=("source",),
            horizontal=True,
            disabled=disabled,
            label_visibility="collapsed",
        )
    if source == SOURCE_FOLDER:
        return _choose_from_folder(disabled)
    # A Browse click that the switch to Upload overtook must not open the
    # window later, when the folder source is shown again.
    st.session_state.pop("pick_pending", None)

    help_text = _FORMATS_HELP
    if local_files:
        help_text += (
            "\n\n**Large recordings:** choose *From a folder on this computer* "
            "instead. An upload is held in the app's memory while it arrives, and "
            "a big one fails when memory runs short."
        )
    uploaded_file = st.file_uploader(
        "Choose video or audio file",
        type=VIDEO_FORMATS + AUDIO_FORMATS,
        key=f"uploader_{st.session_state.uploader_generation or 0}",
        disabled=disabled,
        help=help_text,
    )
    if not uploaded_file:
        return None
    return (
        uploaded_file.file_id,
        uploaded_file.name,
        lambda: audio.save_uploaded_file(uploaded_file),
        None,
    )


def _save_folder() -> None:
    """Keep the recordings folder for next time (the folder field's callback).

    A pasted path of one recording keeps its folder and chooses that file.
    """
    text = st.session_state.recordings_folder
    folder, recording = recordings.parse_location(text)
    if folder is not None:
        st.session_state.recordings_folder = text = str(folder)
    if folder is not None and recording is not None:
        st.session_state.folder_file = str(folder / recording.name)
    db.set_preference("recordings_folder", text)


def _request_pick() -> None:
    """Open the computer's file window on this click's run (the button callback).

    The window itself is opened while the page is drawn, under a spinner that
    says where to look: a callback has no place to show that.
    """
    st.session_state.pick_pending = True


def _pick_recording() -> None:
    """Let the owner choose a recording in the computer's own file window.

    The chosen file's folder becomes the recordings folder (kept for next
    time), and the file is chosen in the list. Called before the folder field
    is drawn, so both widgets can still be set.
    """
    start, _ = recordings.parse_location(st.session_state.recordings_folder)
    with st.spinner(
        "Choose the recording in the window that opened — it may be behind this "
        "browser window…"
    ):
        try:
            chosen = recordings.ask_for_recording(start)
        except AppError as exc:
            st.warning(str(exc))
            return
    if chosen is None:
        return  # cancelled
    if not recordings.is_media(chosen):
        st.warning(f"{chosen.name} is not a video or audio file the app can read.")
        return
    folder = str(chosen.parent)
    st.session_state.recordings_folder = folder
    st.session_state.folder_file = str(chosen.parent / chosen.name)
    db.set_preference("recordings_folder", folder)


def _forget_source() -> None:
    """Drop what was made from a recording that changed, so it is prepared again."""
    for key in _RUN_STATE_KEYS:
        st.session_state[key] = None
    st.session_state.upload_id = None


def _is_app_folder(folder: Path) -> bool:
    """Return True for the app's own working folders, or a folder inside them.

    Args:
        folder: A folder given as the recordings folder.

    Returns:
        Whether it is (in) ``temp/``, ``uploads/``, the history or the model
        folder — "Clean temporary files" would delete a recording picked there.
    """
    settings = get_settings()
    work = (
        settings.temp_dir,
        settings.upload_dir,
        settings.data_dir,
        settings.whisper_model_dir,
    )
    try:
        resolved = folder.resolve()
        return any(resolved.is_relative_to(path.resolve()) for path in work)
    except OSError:
        return False


def _list_folder(folder: Path) -> list[str] | None:
    """List a folder's recordings for the pick list, reporting what goes wrong.

    Args:
        folder: The recordings folder.

    Returns:
        The recordings' paths, newest first; ``None`` (with a warning drawn)
        when the folder cannot be used.
    """
    if _is_app_folder(folder):
        st.warning(
            "This is one of the app's own working folders — choose the folder your "
            "recordings are saved in."
        )
        return None
    try:
        return [str(path) for path in recordings.list_recordings(folder)]
    except OSError as exc:
        st.warning(f"This folder cannot be read: {exc.strerror or exc}")
        return None


def _recording_labels(files: list[str]) -> dict[str, str]:
    """Return each listed recording's entry: length (when known), name, date.

    The length leads because long recorder names (``2026-09-30 15-02-17 -
    daily ….mkv``) are cut off at the end of the list's width. Only the newest
    ``RECORDING_PROBE_LIMIT`` get one.

    Args:
        files: The recordings' paths, as listed.

    Returns:
        Each path's entry, e.g. ``47:12 · Call.mkv · 2026-09-29 16:05``.
    """
    paths = [Path(file) for file in files]
    with st.spinner("Reading how long each recording is…"):
        lengths = recordings.durations(paths[:RECORDING_PROBE_LIMIT])
    labels = {}
    for file, path in zip(files, paths, strict=True):
        seconds = lengths.get(path)
        entry = recordings.label(path)
        labels[file] = f"{_format_length(seconds)} · {entry}" if seconds else entry
    return labels


def _choose_from_folder(disabled: bool) -> ChosenFile | None:
    """Offer the recordings in a folder on this computer, newest first.

    The file is read where it lies: nothing passes through the browser or the
    app's memory, and nothing is copied into ``uploads/``. It is identified by
    its path; when it changes on disk afterwards (still being recorded), the
    page says so instead of silently dropping what was made from it.

    Args:
        disabled: True while a job runs.

    Returns:
        The chosen recording, or ``None`` while there is none.
    """
    _preference("recordings_folder")
    # Popped on every run, and the window is opened only while nothing runs: a
    # click that lands together with Start must not open it in the job's run.
    if st.session_state.pop("pick_pending", False) and not disabled:
        _pick_recording()
    field, browse = st.columns([2, 1], vertical_alignment="bottom")
    with field:
        st.text_input(
            "Recordings folder",
            key="recordings_folder",
            on_change=_save_folder,
            disabled=disabled,
            placeholder="e.g. C:\\Users\\you\\Videos",
            help=(
                "The folder your recordings are saved in (OBS, Teams, a phone's "
                "folder), as a full path — or use **📂 Browse…** to find one "
                "recording in this computer's own file window. Pasting the path of "
                "one recording works too (Explorer: Shift + right-click it → "
                "*Copy as path*). Remembered for next time."
            ),
        )
    with browse:
        st.button(
            "📂 Browse…",
            key="pick_file",
            on_click=_request_pick,
            disabled=disabled,
            width="stretch",
            help=(
                "Opens this computer's own window for choosing a file, in the last "
                "folder used. The recording is read from there, not uploaded."
            ),
        )
    text = st.session_state.recordings_folder
    folder, _ = recordings.parse_location(text)
    if folder is None:
        if text.strip():
            st.warning("There is no such folder on this computer (give its full path).")
        return None
    files = _list_folder(folder)
    if files is None:
        return None
    picked = st.session_state.get("folder_file")
    if picked not in files:
        # Nothing chosen yet (start empty rather than on the newest file, so
        # opening the page does not start an extraction), a path typed in
        # another letter case, or a file that is gone.
        found = recordings.find(files, picked) if picked else None
        if picked and found is None:
            st.session_state.folder_gone = Path(picked).name
        st.session_state.folder_file = found
    if not files:
        st.info("No video or audio files in this folder (subfolders are not listed).")
        return None
    labels = _recording_labels(files)
    picked = st.session_state.folder_file
    shown = st.session_state.get("folder_shown") or {}
    if picked is not None and shown.get(picked, labels[picked]) != labels[picked]:
        # The chosen recording's entry is not the text the browser was given:
        # its length appeared (or grew) since the list was drawn — also when
        # it was picked from that list. The browser holds the old text and
        # sends it back on the next click, where it matches no entry and the
        # choice is lost; setting the value again sends the new text.
        st.session_state.folder_file = picked
    chosen = st.selectbox(
        "Recording",
        options=files,
        key="folder_file",
        format_func=lambda path: labels.get(path, Path(path).name),
        placeholder="Choose a recording (newest first)",
        disabled=disabled,
    )
    st.session_state.folder_shown = labels
    if not chosen:
        gone = st.session_state.get("folder_gone")
        if gone:
            st.warning(f"{gone} is no longer in this folder — choose it again.")
        return None
    st.session_state.folder_gone = None
    path = Path(chosen)
    try:
        version = recordings.version(path)
    except OSError:
        st.warning("This recording can no longer be read.")
        return None
    source_id = f"folder:{chosen}"
    prepared = st.session_state.source_version
    if st.session_state.upload_id == source_id and prepared not in (None, version):
        st.warning(
            "This recording has changed on disk since it was prepared — it may "
            "still be being recorded. The audio and result below are from before."
        )
        st.button("↻ Load it again", on_click=_forget_source, disabled=disabled)
    st.caption(f"{recordings.summary(path)} · read from disk, not uploaded")
    return source_id, path.name, lambda: path, version


def render_transcribe_tab() -> tuple[Any | None, dict[str, Any] | None]:
    """Render the main transcription workflow: upload, prepare, transcribe.

    Returns:
        The container under the Start button, where a requested job draws its
        progress, and the settings shown on the page (the job's parameters);
        both ``None`` when no file is ready.
    """
    _uploader_label_css()
    running = st.session_state.job is not None
    job_area = None
    params = None
    left, right = st.columns([2, 3])

    with left:
        chosen = _choose_source(running)

        if chosen:
            source_id, name, obtain, version = chosen
            _track_source(source_id, name, version)

            extension = Path(name).suffix.lower().lstrip(".")
            is_audio = extension in AUDIO_FORMATS
            source_type = "audio" if is_audio else "video"

            st.subheader("1️⃣ Audio")
            prepare_audio(name, is_audio, obtain)

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
                params = {
                    "provider": provider,
                    "model": model,
                    "with_timestamps": with_timestamps,
                    "source_type": source_type,
                    "visual": visual,
                }

                st.button(
                    "Start Transcription",
                    type="primary",
                    disabled=running,
                    on_click=_request_job,
                )
                if _owns_job():
                    # The one control left enabled during a run. Not while an
                    # earlier run's thread is still finishing (toolbar Stop):
                    # that one is already stopping.
                    st.button(
                        "⏹ Stop",
                        key="stop_job",
                        on_click=_request_stop,
                        help=(
                            "Stops after the part in progress: a part already sent "
                            "to OpenAI is paid for either way, so it is finished "
                            "and kept, and Start later sends only the rest."
                        ),
                    )
                    st.caption(
                        "⏹ stops after the part in progress (it is kept) — "
                        "usually within a minute."
                    )
                job_area = st.container()
                render_run_notices()

    with right:
        render_results(disabled=running)
    return job_area, params


def _history_cost(record: Any) -> str | None:
    """Phrase a history row's cost, keeping the estimate mark.

    Args:
        record: A row from :func:`db.list_transcriptions`.

    Returns:
        E.g. ``$0.07``, ``≈ $0.03`` or ``free (local)``; ``None`` for rows
        saved before costs were recorded.
    """
    if record["cost_usd"] is None:
        return None
    try:
        details = json.loads(record["usage_json"] or "{}")
    except ValueError:
        details = {}
    if (
        not record["cost_usd"]
        and details.get("provider") == PROVIDER_LOCAL
        and not details.get("estimated")
    ):
        return "free (local)"
    return usage.format_usd(record["cost_usd"], bool(details.get("estimated")))


def _request_retitle(record_id: int) -> None:
    """Ask for a History entry's title on the fragment's next run (a callback).

    The request itself is made while the list is drawn, under a spinner where
    the entry is: a callback has no place to show progress, and blocks the page
    without a sign. A click that arrives right after this entry's last title
    is taken as the second half of a double click and ignored — it would
    otherwise pay for a second title.

    Args:
        record_id: The transcription id.
    """
    done = st.session_state.get(f"hist_title_done_{record_id}")
    if done is not None and time.monotonic() - done < TITLE_REPEAT_GUARD_SECONDS:
        return
    st.session_state.hist_title_pending = record_id


def _retitle(record_id: int) -> None:
    """Ask for a new AI title for a saved transcript and store it with its cost.

    What the request cost is added to the row's cost whether or not a title
    came back — it was paid either way. Rows saved before costs were recorded
    keep no cost: the title's alone would read as the price of the whole
    transcript.

    Args:
        record_id: The transcription id.
    """
    full = db.get_transcription(record_id)
    api_key = resolve_openai_key()
    if full is None or not api_key:
        return
    _, model = _title_preferences()
    title = full["title"]
    try:
        title, records = titles.make_title(full["transcript"], api_key, model)
        message = ("success", f"🏷️ New title: {title}")
    except AppError as exc:
        records = list(getattr(exc, "usage_records", None) or [])
        message = ("warning", f"No new title: {exc}")
    cost_usd, usage_json = full["cost_usd"], full["usage_json"]
    if records and usage_json:
        try:
            cost = usage.add_to_cost(json.loads(usage_json), "title", records)
        except (ValueError, AttributeError):
            cost = None
        if cost is not None:
            cost_usd, usage_json = cost["cost_usd"], json.dumps(cost)
    db.set_title(record_id, title, cost_usd, usage_json)
    st.session_state[f"hist_title_msg_{record_id}"] = message
    st.session_state[f"hist_title_done_{record_id}"] = time.monotonic()
    if st.session_state.get("run_row_id") == record_id:
        # The result on the Transcribe tab names its downloads after it too.
        st.session_state.run_title = title


def _render_retitle(record_id: int, has_title: bool) -> None:
    """Offer a (new) AI title for a History entry, and say how the last try went.

    Args:
        record_id: The transcription id.
        has_title: Whether the row already has a title.
    """
    _, model = _title_preferences()
    has_key = bool(resolve_openai_key())
    per_hour = titles.estimate_cost_per_hour(model)
    cost = ""
    if per_hour:
        cost = f" — about {usage.format_usd(per_hour)} per hour of recording"
    st.button(
        "🏷️ New title" if has_title else "🏷️ Make a title",
        key=f"hist_title_{record_id}",
        on_click=_request_retitle,
        args=(record_id,),
        disabled=not has_key,
        help=(
            f"Sends this transcript to {model} for a title{cost}, added to this "
            "entry's cost. The model is set in the sidebar."
            if has_key
            else "Needs an OpenAI API key (sidebar or `.env`)."
        ),
    )
    # Kept while the entry stays open (a second click of a double click reruns
    # the list and would otherwise wipe it); closing the entry clears it.
    message = st.session_state.get(f"hist_title_msg_{record_id}")
    if message:
        getattr(st, message[0])(message[1])


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

    mode, _ = _title_preferences()
    pending = st.session_state.pop("hist_title_pending", None)
    # Entries to draw open whatever their state: Streamlit identifies an
    # expander by its label too, so a new title makes a new, closed widget.
    reopen: set[tuple[int, str]] = st.session_state.setdefault("hist_reopen", set())
    for record in records:
        if record["id"] == pending:
            with st.spinner("Writing a title…", show_time=True):
                _retitle(pending)
            record = db.get_transcription(pending) or record
        flag = "⏱️ " if record["with_timestamps"] else ""
        name = (
            titles.display_name(_safe_stem(record["filename"]), record["title"], mode)
            or record["filename"]
        )
        label = f"{flag}{name} · {record['model']} · {record['created_at']}"
        if record["id"] == pending:
            reopen.add((pending, label))
        entry = st.expander(
            label,
            key=f"hist_{record['id']}",
            on_change="rerun",
            expanded=(record["id"], label) in reopen,
        )
        if not entry.open:
            st.session_state.pop(f"hist_title_msg_{record['id']}", None)
            continue
        with entry:
            meta_parts = [record["source_type"]]
            if name != record["filename"]:
                # The label shows the title; the file it came from stays findable.
                meta_parts.insert(0, record["filename"])
            if record["provider"]:
                meta_parts.append(record["provider"])
            if record["file_size_mb"]:
                meta_parts.append(f"{record['file_size_mb']} MB")
            if record["elapsed_seconds"]:
                meta_parts.append(f"took {_format_length(record['elapsed_seconds'])}")
            cost_text = _history_cost(record)
            if cost_text:
                meta_parts.append(cost_text)
            st.caption(" · ".join(meta_parts))

            full = db.get_transcription(record["id"])
            if full is None:
                # Another session deleted it between the list query and this
                # fetch; skip the row rather than blanking the whole tab.
                st.info("This entry was deleted in another session.")
                continue
            base_name = _download_stem(full["filename"], full["title"])

            st.text_area(
                "Transcript",
                full["transcript"],
                height=200,
                key=f"hist_txt_{record['id']}",
            )
            st.download_button(
                label="📥 Download Transcript",
                data=full["transcript"],
                file_name=f"{base_name}.txt",
                mime="text/plain",
                key=f"hist_dl_txt_{record['id']}",
                on_click="ignore",
            )
            if full["srt"]:
                st.download_button(
                    label="📥 Download Subtitles (.srt)",
                    data=full["srt"],
                    file_name=f"{base_name}.srt",
                    mime="text/plain",
                    key=f"hist_dl_srt_{record['id']}",
                    on_click="ignore",
                )
            if mode != TITLE_MODE_OFF:
                _render_retitle(record["id"], has_title=bool(full["title"]))
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
        # Uploads, extracted WAVs and player MP3s (not transcripts): each page
        # opened clears those a day old, so they no longer pile up for weeks.
        checkpoints.prune_working_copies(WORKING_COPY_MAX_AGE_HOURS)
        st.session_state.checkpoints_pruned = True
    _settle_job()
    _claim_job()
    running = st.session_state.job is not None
    render_sidebar(disabled=running)

    st.title("📝 Video & Audio Transcription")
    st.write("Transcribe video or audio — OpenAI API or a local offline Whisper model")

    tab_transcribe, tab_history = st.tabs(["🎙️ Transcribe", "📚 History"])
    with tab_transcribe:
        job_area, params = render_transcribe_tab()
    with tab_history:
        render_history_tab()
    with st.sidebar:
        render_working_files(disabled=running)

    st.markdown("---")
    st.markdown("Made with ❤️ by Marko A")
    _execute_job(job_area, params)


if __name__ == "__main__":
    main()
