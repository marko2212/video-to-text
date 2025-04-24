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

# Create necessary directories
for dir in [TEMP_DIR, UPLOAD_DIR]:
    os.makedirs(dir, exist_ok=True)


def extract_audio(input_path, output_path=None):
    """Extracts audio from MKV file."""
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
    st.title("📝 Video Transcription")
    st.write("Application for extracting and transcribing audio from video files")

    # Initialize session state variables
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcript_path" not in st.session_state:
        st.session_state.transcript_path = None
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None
    if "progress" not in st.session_state:
        st.session_state.progress = 0

    # File uploader
    uploaded_file = st.file_uploader("Choose video file (MKV format)", type=["mkv"])

    if uploaded_file:
        st.session_state.original_filename = uploaded_file.name

        # Audio extraction section
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

        # Transcription section
        if st.session_state.audio_path:
            st.subheader("2️⃣ Transcription")

            if st.button("Start Transcription"):
                with st.spinner(
                    "Transcription in progress... This may take several minutes."
                ):
                    try:
                        # Original filename for transcript
                        transcript_filename = f"transcript_{os.path.splitext(st.session_state.original_filename)[0]}.txt"
                        transcript_path = os.path.join(TEMP_DIR, transcript_filename)

                        # Start transcription with progress callback
                        process_audio(
                            st.session_state.audio_path,
                            transcript_path,
                            API_KEY,
                            progress_callback=update_progress,
                        )

                        st.session_state.transcript_path = transcript_path

                        # Display and download transcript
                        with open(transcript_path, encoding="utf-8") as file:
                            transcript_text = file.read()
                            st.text_area(
                                "Transcript preview:", transcript_text, height=200
                            )

                        with open(transcript_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Transcript",
                                data=file,
                                file_name=transcript_filename,
                                mime="text/plain",
                            )
                    except Exception as e:
                        st.error(f"Transcription error: {str(e)}")

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
