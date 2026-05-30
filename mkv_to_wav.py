import os

import ffmpeg


def extract_audio(input_path, output_path=None):
    """
    Extract audio from MKV file to WAV using ffmpeg

    Args:
        input_path (str): Path to input MKV file
        output_path (str): Path for output WAV file (optional)
    """
    try:
        # If output path not specified, create one based on input filename
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".wav"

        # Run ffmpeg command - directly extract to WAV
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(
            stream, output_path, acodec="pcm_s16le"
        )  # Standard WAV codec
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)

        print(f"Successfully extracted audio to: {output_path}")
        return output_path

    except ffmpeg.Error as e:
        print("An error occurred: ", e.stderr.decode())
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_file = "test1.mkv"  # promeni ovu putanju
    output_file = "audio.wav"  # promeni ovu putanju

    extract_audio(input_file, output_file)
