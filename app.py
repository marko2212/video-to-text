"""Streamlit UI for the video & audio transcription app.

This module is presentation-only: audio preparation lives in :mod:`audio`, the
transcription pipeline in :mod:`transcribe`, configuration in :mod:`config`, and
history persistence in :mod:`db`.
"""

import importlib.util
from pathlib import Path
from typing import Any

import streamlit as st

import audio
import db
import transcribe
from config import (
    AUDIO_FORMATS,
    DEFAULT_LOCAL_MODEL,
    LOCAL_MODELS,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDERS,
    TIMESTAMP_MODELS,
    TRANSCRIPTION_MODELS,
    VIDEO_FORMATS,
    get_settings,
)
from exceptions import AppError
from logger import get_logger

logger = get_logger(__name__)

# faster-whisper is an optional dependency (install via `uv sync --extra local`).
LOCAL_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None


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
    )


def clean_temp_files() -> None:
    """Remove working files from temp/ and uploads/ (history DB is untouched)."""
    settings = get_settings()
    try:
        for directory in (settings.temp_dir, settings.upload_dir):
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
        for key in ("audio_path", "transcript_path", "srt_path", "original_filename"):
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
                wav_name = f"{Path(uploaded_file.name).stem}.wav"
                st.session_state.audio_path = audio.to_wav(
                    source, get_settings().temp_dir / wav_name
                )
    except AppError as exc:
        st.error(f"Error preparing audio: {exc}")


def run_transcription(
    provider: str, model: str, with_timestamps: bool, source_type: str
) -> None:
    """Run the selected provider's pipeline and store result paths.

    Args:
        provider: OpenAI API or local provider.
        model: Selected model (API model name, or local model size).
        with_timestamps: Whether to generate timestamps/subtitles.
        source_type: Either ``"audio"`` or ``"video"``.
    """
    settings = get_settings()
    base_name = Path(st.session_state.original_filename).stem
    transcript_path = settings.temp_dir / f"transcript_{base_name}.txt"
    srt_path = settings.temp_dir / f"transcript_{base_name}.srt"
    srt_arg = srt_path if with_timestamps else None

    try:
        if provider == PROVIDER_LOCAL:
            with st.spinner(f"Loading model '{model}' (first run downloads it)…"):
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
                )
        else:
            api_key = resolve_openai_key()
            if not api_key:
                st.error("Enter your OpenAI API key in the sidebar to use the API.")
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
                )

        st.session_state.transcript_path = transcript_path
        st.session_state.srt_path = (
            srt_path if with_timestamps and srt_path.exists() else None
        )
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
                for key in ("audio_path", "transcript_path", "srt_path"):
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
                if not LOCAL_AVAILABLE:
                    st.caption(
                        "ℹ️ Install offline Whisper with `uv sync --extra local` "
                        "to transcribe locally without an API key."
                    )

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

                if st.button("Start Transcription"):
                    run_transcription(provider, model, with_timestamps, source_type)

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
            st.caption(" · ".join(meta_parts))

            full = db.get_transcription(record["id"])
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
    for key in ("audio_path", "transcript_path", "srt_path", "original_filename"):
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
