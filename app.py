import os

import ffmpeg
import streamlit as st
from dotenv import load_dotenv

import db
from transcribe import process_audio  # importing function from previous code

# Load environment variables
load_dotenv()

# Configuration
TEMP_DIR = "temp"
UPLOAD_DIR = "uploads"
API_KEY = os.getenv("OPENAI_API_KEY")

VIDEO_FORMATS = [
    "mkv",
    "mp4",
    "mov",
    "avi",
    "webm",
    "m4v",
    "wmv",
    "flv",
    "mpeg",
    "mpg",
    "3gp",
    "ts",
    "mts",
    "m2ts",
    "ogv",
    "vob",
]

AUDIO_FORMATS = [
    "mp3",
    "wav",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "opus",
    "wma",
    "aiff",
    "aif",
    "amr",
]

# Transcription models offered in the UI (first is the default).
TRANSCRIPTION_MODELS = ["gpt-4o-transcribe", "whisper-1"]
# Models that can return per-segment timestamps for subtitle (SRT) export.
TIMESTAMP_MODELS = {"whisper-1"}

# Create necessary directories
for dir in [TEMP_DIR, UPLOAD_DIR]:
    os.makedirs(dir, exist_ok=True)


def extract_audio(input_path, output_path=None):
    """Extracts audio from a video file (MKV, MP4, etc.) to WAV."""
    try:
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".wav"

        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, acodec="pcm_s16le")
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)

        return output_path
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None


def save_uploaded_file(uploaded_file):
    """Saves uploaded file to uploads directory."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def update_progress(progress_info):
    """Updates progress information in Streamlit interface."""
    # Create a container for progress information if it doesn't exist
    if "progress_container" not in st.session_state:
        st.session_state.progress_container = st.empty()

    # Create a new container with the latest information
    with st.session_state.progress_container:
        st.empty()  # Clear previous content
        if progress_info["status"] == "info" or progress_info["status"] == "start":
            st.info(progress_info["message"])
        elif progress_info["status"] == "progress":
            col1, col2 = st.columns([1, 2])
            with col1:
                st.progress(progress_info["progress"])
            with col2:
                st.info(progress_info["message"])
        elif progress_info["status"] == "complete":
            st.success(progress_info["message"])
        elif progress_info["status"] == "error":
            st.error(progress_info["message"])


def save_to_history(source_type, model, with_timestamps):
    """Persists the just-finished transcription into the SQLite history."""
    transcript_path = st.session_state.transcript_path
    if not transcript_path or not os.path.exists(transcript_path):
        return

    with open(transcript_path, encoding="utf-8") as file:
        transcript_text = file.read()

    srt_text = None
    srt_path = st.session_state.srt_path
    if srt_path and os.path.exists(srt_path):
        with open(srt_path, encoding="utf-8") as file:
            srt_text = file.read()

    audio_path = st.session_state.audio_path
    file_size_mb = None
    if audio_path and os.path.exists(audio_path):
        file_size_mb = round(os.path.getsize(audio_path) / (1024 * 1024), 2)

    db.add_transcription(
        filename=st.session_state.original_filename,
        source_type=source_type,
        model=model,
        with_timestamps=with_timestamps,
        transcript=transcript_text,
        srt=srt_text,
        audio_path=audio_path,
        file_size_mb=file_size_mb,
    )


def clean_temp_files():
    """Removes working files from temp/ and uploads/ (history DB is untouched)."""
    try:
        for dir_path in [TEMP_DIR, UPLOAD_DIR]:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error deleting {file}: {str(e)}")
        st.session_state.audio_path = None
        st.session_state.transcript_path = None
        st.session_state.srt_path = None
        st.session_state.original_filename = None
        st.session_state.progress = 0
        if "progress_container" in st.session_state:
            st.session_state.progress_container.empty()
            del st.session_state.progress_container
        st.success("Temporary files cleaned!")
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")


def render_results():
    """Renders the transcript preview and download buttons from session state."""
    st.subheader("📄 Result")
    transcript_path = st.session_state.transcript_path
    if not transcript_path or not os.path.exists(transcript_path):
        st.info("The transcript will appear here after you run a transcription.")
        return

    with open(transcript_path, encoding="utf-8") as file:
        transcript_text = file.read()
    st.text_area("Transcript preview:", transcript_text, height=420)

    with open(transcript_path, "rb") as file:
        st.download_button(
            label="📥 Download Transcript",
            data=file,
            file_name=os.path.basename(transcript_path),
            mime="text/plain",
        )

    srt_path = st.session_state.srt_path
    if srt_path and os.path.exists(srt_path):
        with open(srt_path, "rb") as file:
            st.download_button(
                label="📥 Download Subtitles (.srt)",
                data=file,
                file_name=os.path.basename(srt_path),
                mime="text/plain",
            )


def render_transcribe_tab():
    """The main transcription workflow: upload, prepare, transcribe."""
    # Hide Streamlit's auto-generated long format list under the uploader
    # and replace it with a clean short label. Full list lives in `help` tooltip.
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
            # Reset previous results when a different file is uploaded
            if st.session_state.original_filename != uploaded_file.name:
                st.session_state.audio_path = None
                st.session_state.transcript_path = None
                st.session_state.srt_path = None
            st.session_state.original_filename = uploaded_file.name

            extension = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
            is_audio = extension in AUDIO_FORMATS
            source_type = "audio" if is_audio else "video"

            # 1) Prepare audio automatically (guarded so it runs once per file).
            #    Audio is used as-is; video has its audio track extracted.
            st.subheader("1️⃣ Audio")
            if st.session_state.audio_path is None:
                if is_audio:
                    st.session_state.audio_path = save_uploaded_file(uploaded_file)
                else:
                    with st.spinner("Extracting audio from video..."):
                        video_path = save_uploaded_file(uploaded_file)
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        wav_path = os.path.join(TEMP_DIR, f"{base_name}.wav")
                        if extract_audio(video_path, wav_path):
                            st.session_state.audio_path = wav_path
                        else:
                            st.error("Error extracting audio.")

            audio_path = st.session_state.audio_path
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
                if not is_audio:
                    with open(audio_path, "rb") as audio_file:
                        st.download_button(
                            label="📥 Download extracted audio (WAV)",
                            data=audio_file,
                            file_name=os.path.basename(audio_path),
                            mime="audio/wav",
                        )

            # 2) Transcription controls
            if st.session_state.audio_path:
                st.subheader("2️⃣ Transcription")

                model = st.selectbox(
                    "Transcription model",
                    options=TRANSCRIPTION_MODELS,
                    index=0,
                    help=(
                        "**gpt-4o-transcribe** — newer, more accurate.\n\n"
                        "**whisper-1** — supports timestamps & subtitle (.srt) export."
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
                    with st.spinner(
                        "Transcription in progress... This may take several minutes."
                    ):
                        try:
                            base_name = os.path.splitext(
                                st.session_state.original_filename
                            )[0]
                            transcript_path = os.path.join(
                                TEMP_DIR, f"transcript_{base_name}.txt"
                            )
                            srt_path = os.path.join(
                                TEMP_DIR, f"transcript_{base_name}.srt"
                            )

                            process_audio(
                                st.session_state.audio_path,
                                transcript_path,
                                API_KEY,
                                model=model,
                                with_timestamps=with_timestamps,
                                srt_output_file=srt_path if with_timestamps else None,
                                progress_callback=update_progress,
                            )

                            st.session_state.transcript_path = transcript_path
                            st.session_state.srt_path = (
                                srt_path
                                if with_timestamps and os.path.exists(srt_path)
                                else None
                            )
                            save_to_history(source_type, model, with_timestamps)
                        except Exception as e:
                            st.error(f"Transcription error: {str(e)}")

        st.divider()
        if st.button("🧹 Clean temporary files"):
            clean_temp_files()

    with right:
        render_results()


def render_history_tab():
    """Lists past transcriptions stored in SQLite, with download/delete."""
    st.subheader("📚 Transcription history")
    records = db.list_transcriptions()
    if not records:
        st.info("No transcriptions yet. Run one in the Transcribe tab.")
        return

    for record in records:
        flag = "⏱️ " if record["with_timestamps"] else ""
        label = (
            f"{flag}{record['filename']} · "
            f"{record['model']} · {record['created_at']}"
        )
        with st.expander(label):
            meta = record["source_type"]
            if record["file_size_mb"]:
                meta += f" · {record['file_size_mb']} MB"
            st.caption(meta)

            full = db.get_transcription(record["id"])
            base_name = os.path.splitext(full["filename"])[0]

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


def main():
    st.set_page_config(
        page_title="Video & Audio Transcription",
        page_icon="📝",
        layout="wide",
    )
    db.init_db()

    st.title("📝 Video & Audio Transcription")
    st.write("Transcribe audio from video or audio files using OpenAI Whisper")

    # Initialize session state variables
    for key in ("audio_path", "transcript_path", "srt_path", "original_filename"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "progress" not in st.session_state:
        st.session_state.progress = 0

    tab_transcribe, tab_history = st.tabs(["🎙️ Transcribe", "📚 History"])

    with tab_transcribe:
        render_transcribe_tab()

    with tab_history:
        render_history_tab()

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Marko A")


if __name__ == "__main__":
    main()
