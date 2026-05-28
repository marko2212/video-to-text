import os

import ffmpeg
import streamlit as st
from dotenv import load_dotenv

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


def main():
    st.title("📝 Video & Audio Transcription")
    st.write("Transcribe audio from video or audio files using OpenAI Whisper")

    # Initialize session state variables
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcript_path" not in st.session_state:
        st.session_state.transcript_path = None
    if "srt_path" not in st.session_state:
        st.session_state.srt_path = None
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None
    if "progress" not in st.session_state:
        st.session_state.progress = 0

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

        # Determine whether the upload is an audio or a video file
        extension = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
        is_audio = extension in AUDIO_FORMATS

        if is_audio:
            # Audio input: no extraction needed, the transcription pipeline
            # normalizes it internally — feed it straight to transcription.
            st.subheader("1️⃣ Audio File")
            if st.session_state.audio_path is None:
                st.session_state.audio_path = save_uploaded_file(uploaded_file)
            st.success("Audio ready — start the transcription below.")
        else:
            # Video input: extract the audio track first
            st.subheader("1️⃣ Audio Extraction")

            if st.button("Extract Audio"):
                with st.spinner("Extracting audio..."):
                    # Save uploaded file
                    video_path = save_uploaded_file(uploaded_file)

                    # Extract audio
                    audio_filename = f"{os.path.splitext(uploaded_file.name)[0]}.wav"
                    audio_path = os.path.join(TEMP_DIR, audio_filename)

                    if extract_audio(video_path, audio_path):
                        st.session_state.audio_path = audio_path
                        st.success("Audio successfully extracted!")

                        # Audio download option
                        with open(audio_path, "rb") as audio_file:
                            st.download_button(
                                label="📥 Download Audio",
                                data=audio_file,
                                file_name=audio_filename,
                                mime="audio/wav",
                            )
                    else:
                        st.error("Error extracting audio.")

        # Transcription section (shared by audio and video inputs)
        if st.session_state.audio_path:
            st.subheader("2️⃣ Transcription")

            # Model selection
            model = st.selectbox(
                "Transcription model",
                options=TRANSCRIPTION_MODELS,
                index=0,
                help=(
                    "**gpt-4o-transcribe** — newer, more accurate.\n\n"
                    "**whisper-1** — supports timestamps & subtitle (.srt) export."
                ),
            )

            # Timestamp/subtitle option (only for models that support it)
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
                        # Output file names based on the original filename
                        base_name = os.path.splitext(
                            st.session_state.original_filename
                        )[0]
                        transcript_path = os.path.join(
                            TEMP_DIR, f"transcript_{base_name}.txt"
                        )
                        srt_path = os.path.join(TEMP_DIR, f"transcript_{base_name}.srt")

                        # Start transcription with progress callback
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
                    except Exception as e:
                        st.error(f"Transcription error: {str(e)}")

            # Show results/downloads outside the button handler so they persist
            # across reruns (e.g. clicking one download won't hide the other).
            transcript_path = st.session_state.transcript_path
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, encoding="utf-8") as file:
                    transcript_text = file.read()
                st.text_area("Transcript preview:", transcript_text, height=200)

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

    # Clean old files
    if st.button("🧹 Clean temporary files"):
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

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Marko A")


if __name__ == "__main__":
    main()
