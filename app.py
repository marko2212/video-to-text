"""Streamlit UI for the video & audio transcription app.

This module is presentation-only: audio preparation lives in :mod:`audio`, the
transcription pipeline in :mod:`transcribe`, configuration in :mod:`config`, and
history persistence in :mod:`db`.
"""

import importlib.util
import shutil
import time
from pathlib import Path
from typing import Any

import streamlit as st

import audio
import db
import frames
import transcribe
import vision
from config import (
    AUDIO_FORMATS,
    DEFAULT_FRAME_DETAIL,
    DEFAULT_LOCAL_MODEL,
    FRAME_DETAIL_LEVELS,
    FRAME_INTERVAL_MAX_SECONDS,
    FRAME_INTERVAL_MIN_SECONDS,
    FRAME_INTERVAL_STEP_SECONDS,
    FRAME_MAX_INTERVAL_SECONDS,
    LOCAL_MODEL_SIZES_MB,
    LOCAL_MODELS,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDERS,
    TIMESTAMP_MODELS,
    TRANSCRIPTION_MODELS,
    VIDEO_FORMATS,
    VISION_MODELS,
    get_settings,
)
from exceptions import AppError
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
    "transcript_path",
    "srt_path",
    "elapsed_seconds",
)


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


def resolve_openai_key() -> str | None:
    """Return the OpenAI key from settings (.env) or the sidebar field, if any.

    Returns:
        The API key string, or ``None`` if none has been provided.
    """
    return get_settings().openai_api_key or st.session_state.get("openai_api_key_ui")


def render_sidebar() -> None:
    """Render the sidebar; offer an API key field when none is set via the env."""
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
                help=(
                    "Needed only for the OpenAI API provider. Stored only for "
                    "this session — never written to disk."
                ),
            )
            st.caption(
                "Tip: set `OPENAI_API_KEY` in a `.env` file to load it "
                "automatically every run."
            )


def update_progress(progress_info: dict[str, Any]) -> None:
    """Render transcription progress updates in the UI.

    Args:
        progress_info: Status payload with ``status`` and ``message`` keys
            (and ``progress`` for the ``"progress"`` status).
    """
    if "progress_container" not in st.session_state:
        st.session_state.progress_container = st.empty()

    with st.session_state.progress_container:
        st.empty()  # Clear previous content
        status = progress_info["status"]
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


def clean_temp_files() -> None:
    """Remove working files from temp/ and uploads/ (history DB is untouched)."""
    settings = get_settings()
    try:
        for directory in (settings.temp_dir, settings.upload_dir):
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
        for key in (*_RUN_STATE_KEYS, "original_filename"):
            st.session_state[key] = None
        if "progress_container" in st.session_state:
            st.session_state.progress_container.empty()
            del st.session_state.progress_container
        st.success("Temporary files cleaned!")
    except OSError as exc:
        logger.warning("Cleanup failed: %s", exc)
        st.error(f"Error during cleanup: {exc}")


def prepare_audio(uploaded_file: Any, is_audio: bool) -> None:
    """Make the uploaded media ready for transcription (run once per file).

    Audio uploads are stored as-is; videos have their audio track extracted.
    The resulting path is stored in ``st.session_state.audio_path``.

    Args:
        uploaded_file: The Streamlit uploaded file.
        is_audio: True if the upload is an audio file.
    """
    if st.session_state.audio_path is not None:
        return
    try:
        if is_audio:
            st.session_state.audio_path = audio.save_uploaded_file(uploaded_file)
        else:
            with st.spinner("Extracting audio from video..."):
                source = audio.save_uploaded_file(uploaded_file)
                # Kept so on-screen context can go back to the video for frames.
                st.session_state.video_path = source
                wav_name = f"{Path(uploaded_file.name).stem}.wav"
                st.session_state.audio_path = audio.to_wav(
                    source, get_settings().temp_dir / wav_name
                )
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


def render_visual_options(source_type: str) -> dict[str, Any] | None:
    """Offer on-screen context for video uploads.

    Args:
        source_type: Either ``"audio"`` or ``"video"``.

    Returns:
        The chosen settings (``model``, ``detail``, ``interval``), or ``None``
        when the feature is switched off or unavailable.
    """
    if source_type != "video":
        return None

    enabled = st.checkbox(
        "🖥️ Describe what's on screen (slides, diagrams, code)",
        value=False,
        help=(
            "Extracts the frames where the picture changed and has a vision "
            "model describe them, merged into the transcript by timestamp. "
            "Needs an OpenAI API key even when transcribing locally."
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
        help="The cheaper model is usually enough to read slide headings.",
    )
    detail = st.radio(
        "Image detail",
        options=FRAME_DETAIL_LEVELS,
        index=FRAME_DETAIL_LEVELS.index(DEFAULT_FRAME_DETAIL),
        horizontal=True,
        help="Use **high** only when you need to read small text off a slide.",
    )
    interval = st.slider(
        "Screenshot at least every (seconds)",
        min_value=FRAME_INTERVAL_MIN_SECONDS,
        max_value=FRAME_INTERVAL_MAX_SECONDS,
        value=int(FRAME_MAX_INTERVAL_SECONDS),
        step=FRAME_INTERVAL_STEP_SECONDS,
        help=(
            "How often to grab a frame even when the picture has not changed. "
            "Scene changes are always captured on top of this, and short videos "
            "get extra samples so they are not covered by a single frame. "
            "Near-identical frames are discarded before anything is sent."
        ),
    )

    st.caption(_screenshot_estimate(float(interval), vision_model, detail))
    return {"model": vision_model, "detail": detail, "interval": float(interval)}


def collect_visual_notes(visual: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract key frames from the uploaded video and describe what they show.

    Failures here are reported but never fatal: a transcript without on-screen
    context is still worth having, so the run continues without notes.

    Args:
        visual: Settings from :func:`render_visual_options` (``model``,
            ``detail``, ``interval``).

    Returns:
        Notes with ``time`` and ``description``, or an empty list.
    """
    video_path: Path | None = st.session_state.video_path
    api_key = resolve_openai_key()
    if not video_path or not api_key:
        return []

    frame_dir = get_settings().temp_dir / "frames"
    try:
        with st.spinner("Looking for scene changes in the video…"):
            keyframes = frames.extract_keyframes(
                video_path, frame_dir, max_interval=visual["interval"]
            )
        if not keyframes:
            st.info("No on-screen changes were detected — nothing to describe.")
            return []
        with st.spinner(f"Describing {len(keyframes)} key frames…"):
            return vision.describe_keyframes(
                keyframes,
                api_key,
                model=visual["model"],
                detail=visual["detail"],
                progress_callback=update_progress,
            )
    except AppError as exc:
        st.warning(f"On-screen context skipped: {exc}")
        return []
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def run_transcription(
    provider: str,
    model: str,
    with_timestamps: bool,
    source_type: str,
    visual: dict[str, Any] | None = None,
) -> None:
    """Run the selected provider's pipeline and store result paths.

    Args:
        provider: OpenAI API or local provider.
        model: Selected model (API model name, or local model size).
        with_timestamps: Whether to generate timestamps/subtitles.
        source_type: Either ``"audio"`` or ``"video"``.
        visual: On-screen context settings, or None to skip that step.
    """
    settings = get_settings()
    base_name = Path(st.session_state.original_filename).stem
    transcript_path = settings.temp_dir / f"transcript_{base_name}.txt"
    srt_path = settings.temp_dir / f"transcript_{base_name}.srt"
    srt_arg = srt_path if with_timestamps else None
    # Drop the previous run's timing before starting. Without this, a run that
    # fails or returns early leaves the old figure on screen, so the result panel
    # claims "Finished in …" for a run that never finished.
    st.session_state.elapsed_seconds = None
    # Timed from here so the reported figure matches the wait the user actually
    # sits through, on-screen context included.
    started = time.monotonic()
    # Gathered before transcription so the notes are ready for the single writer
    # that renders the transcript at the end of either pipeline.
    visual_notes = collect_visual_notes(visual) if visual else None

    try:
        if provider == PROVIDER_LOCAL:
            if _model_is_cached(model):
                spinner_msg = f"Loading model '{model}'…"
            else:
                size = LOCAL_MODEL_SIZES_MB.get(model)
                hint = f" (~{size:.0f} MB)" if size else ""
                spinner_msg = (
                    f"Downloading '{model}'{hint} — first run only, please wait…"
                )
            with st.spinner(spinner_msg):
                whisper_model = load_whisper_model(
                    model, settings.local_device, settings.local_compute_type
                )
            with st.spinner("Transcribing locally… This may take a while on CPU."):
                transcribe.transcribe_local(
                    st.session_state.audio_path,
                    transcript_path,
                    whisper_model,
                    with_timestamps=with_timestamps,
                    srt_output_file=srt_arg,
                    progress_callback=update_progress,
                    visual_notes=visual_notes,
                )
        else:
            api_key = resolve_openai_key()
            if not api_key:
                st.error(
                    "Add your OpenAI API key in the sidebar, or set "
                    "`OPENAI_API_KEY` in a `.env` file. (The Local engine "
                    "needs no key.)"
                )
                return
            with st.spinner("Transcribing with the OpenAI API…"):
                transcribe.transcribe_openai(
                    st.session_state.audio_path,
                    transcript_path,
                    api_key,
                    model=model,
                    with_timestamps=with_timestamps,
                    srt_output_file=srt_arg,
                    progress_callback=update_progress,
                    visual_notes=visual_notes,
                )

        st.session_state.transcript_path = transcript_path
        st.session_state.srt_path = (
            srt_path if with_timestamps and srt_path.exists() else None
        )
        st.session_state.elapsed_seconds = time.monotonic() - started
        save_to_history(source_type, provider, model, with_timestamps)
    except AppError as exc:
        st.error(f"Transcription error: {exc}")


def render_results() -> None:
    """Render the transcript preview and download buttons from session state."""
    st.subheader("📄 Result")
    transcript_path: Path | None = st.session_state.transcript_path
    if not transcript_path or not transcript_path.exists():
        st.info("The transcript will appear here after you run a transcription.")
        return

    elapsed: float | None = st.session_state.elapsed_seconds
    if elapsed is not None:
        st.caption(f"⏱️ Finished in {_format_length(elapsed)}")

    st.text_area(
        "Transcript preview:", transcript_path.read_text(encoding="utf-8"), height=420
    )
    st.download_button(
        label="📥 Download Transcript",
        data=transcript_path.read_bytes(),
        file_name=transcript_path.name,
        mime="text/plain",
    )

    srt_path: Path | None = st.session_state.srt_path
    if srt_path and srt_path.exists():
        st.download_button(
            label="📥 Download Subtitles (.srt)",
            data=srt_path.read_bytes(),
            file_name=srt_path.name,
            mime="text/plain",
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


def render_transcribe_tab() -> None:
    """Render the main transcription workflow: upload, prepare, transcribe."""
    _uploader_label_css()
    left, right = st.columns([2, 3])

    with left:
        uploaded_file = st.file_uploader(
            "Choose video or audio file",
            type=VIDEO_FORMATS + AUDIO_FORMATS,
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
            # Reset previous results when a different file is uploaded.
            if st.session_state.original_filename != uploaded_file.name:
                for key in _RUN_STATE_KEYS:
                    st.session_state[key] = None
            st.session_state.original_filename = uploaded_file.name

            extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
            is_audio = extension in AUDIO_FORMATS
            source_type = "audio" if is_audio else "video"

            st.subheader("1️⃣ Audio")
            prepare_audio(uploaded_file, is_audio)

            audio_path: Path | None = st.session_state.audio_path
            if audio_path and audio_path.exists():
                st.audio(str(audio_path))
                if not is_audio:
                    st.download_button(
                        label="📥 Download extracted audio (WAV)",
                        data=audio_path.read_bytes(),
                        file_name=audio_path.name,
                        mime="audio/wav",
                    )

            if st.session_state.audio_path:
                st.subheader("2️⃣ Transcription")

                providers = PROVIDERS if LOCAL_AVAILABLE else [PROVIDER_OPENAI]
                provider = st.radio("Engine", providers, horizontal=True)

                if provider == PROVIDER_LOCAL:
                    model = st.selectbox(
                        "Local model",
                        options=list(LOCAL_MODELS),
                        index=list(LOCAL_MODELS).index(DEFAULT_LOCAL_MODEL),
                        format_func=lambda name: f"{name} · {LOCAL_MODELS[name]}",
                        help="Downloaded on first use; runs fully offline, no key.",
                    )
                    # Local models all return native timestamps.
                    with_timestamps = st.checkbox(
                        "Include timestamps & generate subtitles (.srt)", value=False
                    )
                else:
                    model = st.selectbox(
                        "Transcription model",
                        options=TRANSCRIPTION_MODELS,
                        index=0,
                        help=(
                            "**gpt-4o-transcribe** — newer, more accurate.\n\n"
                            "**whisper-1** — supports timestamps & subtitles (.srt)."
                        ),
                    )
                    if model in TIMESTAMP_MODELS:
                        with_timestamps = st.checkbox(
                            "Include timestamps & generate subtitles (.srt)",
                            value=False,
                        )
                    else:
                        with_timestamps = False
                        st.caption(
                            "ℹ️ Timestamps & subtitles are available only with the "
                            "whisper-1 model."
                        )

                visual = render_visual_options(source_type)

                if st.button("Start Transcription"):
                    run_transcription(
                        provider, model, with_timestamps, source_type, visual=visual
                    )

        st.divider()
        if st.button("🧹 Clean temporary files"):
            clean_temp_files()

    with right:
        render_results()


def render_history_tab() -> None:
    """List past transcriptions stored in SQLite, with download/delete."""
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
        with st.expander(label):
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
            )
            if full["srt"]:
                st.download_button(
                    label="📥 Download Subtitles (.srt)",
                    data=full["srt"],
                    file_name=f"transcript_{base_name}.srt",
                    mime="text/plain",
                    key=f"hist_dl_srt_{record['id']}",
                )
            if st.button("🗑️ Delete", key=f"hist_del_{record['id']}"):
                db.delete_transcription(record["id"])
                st.rerun()


def init_session_state() -> None:
    """Initialise the session-state keys used across reruns."""
    for key in (*_RUN_STATE_KEYS, "original_filename"):
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
    render_sidebar()

    st.title("📝 Video & Audio Transcription")
    st.write("Transcribe video or audio — OpenAI API or a local offline Whisper model")

    tab_transcribe, tab_history = st.tabs(["🎙️ Transcribe", "📚 History"])
    with tab_transcribe:
        render_transcribe_tab()
    with tab_history:
        render_history_tab()

    st.markdown("---")
    st.markdown("Made with ❤️ by Marko A")


if __name__ == "__main__":
    main()
