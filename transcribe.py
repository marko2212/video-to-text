import os
import subprocess
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# Load environment variables
load_dotenv()


def ensure_temp_folder(temp_folder):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)


def convert_to_wav_ffmpeg(input_file, output_file):
    """Converts audio to WAV format using FFmpeg directly"""
    command = [
        "ffmpeg",
        "-i",
        input_file,
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16kHz sample rate
        output_file,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False


def get_audio_info(file_path, segment_duration):
    """Returns information about the audio file"""
    audio = AudioSegment.from_file(file_path)
    duration_minutes = len(audio) / 1000 / 60
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return {
        "duration_minutes": duration_minutes,
        "size_mb": size_mb,
        "total_segments": len(audio) // segment_duration + 1,
    }


def save_segment(segment, index, temp_folder):
    """Saves segment as WAV then converts to MP3"""
    temp_wav = os.path.join(temp_folder, f"segment_{index}_temp.wav")
    temp_mp3 = os.path.join(temp_folder, f"segment_{index}.mp3")

    try:
        print(f"\nSaving segment {index + 1}...")

        # Save as WAV
        segment.export(temp_wav, format="wav", parameters=["-ac", "1", "-ar", "16000"])

        # Convert to MP3 using FFmpeg
        command = [
            "ffmpeg",
            "-i",
            temp_wav,
            "-acodec",
            "libmp3lame",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "192k",
            temp_mp3,
        ]
        subprocess.run(command, check=True, capture_output=True)

        # Check file size
        size_mb = os.path.getsize(temp_mp3) / (1024 * 1024)
        print(f"Segment {index + 1} size: {size_mb:.2f} MB")

        # Only check minimum size if segment duration is close to full length
        if (
            len(segment) >= segment.frame_rate * 5 and size_mb < 0.1
        ):  # If segment is longer than 5 seconds
            raise Exception(f"Generated segment is too small: {size_mb:.2f} MB")

        return temp_mp3

    except Exception as e:
        print(f"Error saving segment {index}: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def split_audio(file_path, temp_folder, segment_duration):
    print("Loading audio file...")
    temp_wav = os.path.join(temp_folder, "temp_full.wav")
    try:
        if not convert_to_wav_ffmpeg(file_path, temp_wav):
            raise Exception("Error converting to WAV format")

        print("Loading converted WAV file...")
        audio = AudioSegment.from_wav(temp_wav)

        print("Checking audio parameters:")
        print(f"Channels: {audio.channels}")
        print(f"Sample width: {audio.sample_width}")
        print(f"Frame rate: {audio.frame_rate}")
        print(f"Duration: {len(audio) / 1000} seconds")

        segments = []
        for i in range(0, len(audio), segment_duration):
            segment = audio[i : i + segment_duration]
            segments.append(segment)

        return segments
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def transcribe_segment(file_path, client, retry_count=3, min_duration_seconds=5):
    """Transcribes one segment with retry mechanism and checks"""
    for attempt in range(retry_count):
        try:
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")

            # Get audio duration using pydub
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000

            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > 25:
                raise Exception(
                    f"File is too large ({size_mb:.2f} MB). Maximum is 25 MB"
                )
            elif duration_seconds >= min_duration_seconds and size_mb < 0.1:
                # Only apply minimum size check for segments longer than min_duration_seconds
                raise Exception(f"File is too small ({size_mb:.2f} MB)")
            elif duration_seconds < min_duration_seconds and size_mb < 0.01:
                # For very short segments, still have a minimum threshold
                raise Exception(f"File is too small ({size_mb:.2f} MB)")

            print(f"\nSending segment for transcription (attempt {attempt + 1})...")
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            return transcription.text
        except Exception as e:
            print(f"Transcription error (attempt {attempt + 1}): {str(e)}")
            if attempt == retry_count - 1:
                raise
            time.sleep(2**attempt)


def process_audio(
    input_file,
    output_file,
    api_key,
    segment_duration_minutes=10,
    progress_callback=None,
):
    """Main method for processing audio file"""
    temp_folder = "temp_audio_segments"
    segment_duration = segment_duration_minutes * 60 * 1000
    client = OpenAI(api_key=api_key)

    try:
        ensure_temp_folder(temp_folder)

        # Get file information
        info = get_audio_info(input_file, segment_duration)
        if progress_callback:
            progress_callback(
                {
                    "status": "info",
                    "message": f"Audio information:\nDuration: {info['duration_minutes']:.2f} minutes\nSize: {info['size_mb']:.2f} MB\nNumber of segments: {info['total_segments']}",
                }
            )

        # Split audio into segments
        segments = split_audio(input_file, temp_folder, segment_duration)

        # Prepare file for results
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription started: {datetime.now()}\n\n")

        if progress_callback:
            progress_callback(
                {
                    "status": "start",
                    "message": f"Transcription started at {datetime.now()}",
                }
            )

        # Process each segment
        all_transcriptions = []
        total_segments = len(segments)
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(
                    {
                        "status": "progress",
                        "message": f"Processing segment {i + 1}/{total_segments}",
                        "progress": (i + 1) / total_segments,
                    }
                )

            temp_path = save_segment(segment, i, temp_folder)
            try:
                transcription = transcribe_segment(temp_path, client)
                all_transcriptions.append(transcription)

                # Save progress
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n--- Segment {i + 1} ---\n")
                    f.write(transcription + "\n")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Save complete transcription
        completion_time = datetime.now()
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n\nTranscription completed: {completion_time}")

        if progress_callback:
            progress_callback(
                {
                    "status": "complete",
                    "message": f"Transcription completed at {completion_time}",
                }
            )

        print(
            f"\nTranscription has been successfully completed and saved to: {output_file}"
        )

    except Exception as e:
        if progress_callback:
            progress_callback(
                {"status": "error", "message": f"An error occurred: {str(e)}"}
            )
        print(f"\nAn error occurred: {str(e)}")
        raise
    finally:
        # Clean up temporary folder
        try:
            if os.path.exists(temp_folder):
                for file in os.listdir(temp_folder):
                    try:
                        os.remove(os.path.join(temp_folder, file))
                    except Exception as e:
                        print(f"Error deleting temp file: {str(e)}")
                os.rmdir(temp_folder)
        except Exception as e:
            print(f"Error cleaning up temp folder: {str(e)}")


if __name__ == "__main__":
    try:
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Configuration
        INPUT_FILE = "audio.wav"  # Change this to your input file path
        OUTPUT_FILE = "transcript.txt"  # Change this to your desired output file path
        SEGMENT_DURATION = 10  # minutes

        # Processing
        process_audio(INPUT_FILE, OUTPUT_FILE, api_key, SEGMENT_DURATION)

    except Exception as e:
        print(f"\nProgram terminated due to error: {str(e)}")
