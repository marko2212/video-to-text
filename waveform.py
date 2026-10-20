"""A picture of a recording's sound: how loud each stretch of it is.

Drawn under the trim slider, above the player, so that quiet stretches can be
seen — a recording left running for two hours after the meeting ended — and
left out of the run with that slider. This module is UI-agnostic: the app draws
what it returns.
"""

import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from audio import to_wav
from config import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    WAVEFORM_BUCKETS,
    WAVEFORM_HEIGHT,
    WAVEFORM_WIDTH,
    get_settings,
)
from exceptions import AudioProcessingError
from logger import get_logger

logger = get_logger(__name__)

# Colours of the bars: the stretch that will be transcribed (Streamlit's red)
# and the stretch left out, grey and faint, under a light shade. The background
# stays transparent, so the picture suits the light and the dark theme alike.
_KEPT = (255, 75, 75, 255)
_LEFT_OUT = (150, 150, 150, 110)
_SHADE = (128, 128, 128, 40)
# Bars grow with loudness to this power, so speech and pauses both stay visible.
_CURVE = 0.7
# Bars are scaled to the loudness this share of the stretches stay under, not to
# the single loudest one: one clap or a dropped microphone would flatten the rest.
_TOP_SHARE = 0.98


def _is_plain_wav(path: Path) -> bool:
    """Return True for a 16-bit PCM WAV that is already mono and at most 16 kHz.

    Args:
        path: The file to check.

    Returns:
        Whether the file can be read as it is, without ffmpeg.
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == TARGET_CHANNELS
                and handle.getsampwidth() == 2
                and handle.getframerate() <= TARGET_SAMPLE_RATE
            )
    except (wave.Error, EOFError, OSError):
        return False


def _load(path: Path) -> AudioSegment:
    """Load a recording as mono 16 kHz audio.

    The WAV the app extracts from a video is read as it is. Anything else goes
    through ffmpeg once, to a small mono WAV: decoding a two-hour stereo file at
    its own rate would take well over a gigabyte of memory.

    Args:
        path: Any audio file ffmpeg can read.

    Returns:
        The decoded recording.

    Raises:
        AudioProcessingError: If ffmpeg cannot decode the file.
    """
    if _is_plain_wav(path):
        return AudioSegment.from_file(str(path), format="wav")
    temp_dir = get_settings().temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="loudness-", suffix=".wav", dir=temp_dir)
    os.close(handle)
    temp = Path(name)
    try:
        to_wav(path, temp)
        return AudioSegment.from_file(str(temp), format="wav")
    finally:
        temp.unlink(missing_ok=True)


def loudness(path: Path, buckets: int = WAVEFORM_BUCKETS) -> tuple[list[float], float]:
    """Measure how loud each of ``buckets`` equal stretches of a recording is.

    Args:
        path: Any audio file ffmpeg can read; the app's own WAV is read directly.
        buckets: How many stretches to measure.

    Returns:
        One value per stretch, from 0 (silence) to 1 (as loud as the loudest
        stretches), and the recording's length in seconds. Empty, and 0, for a
        recording without any sound.

    Raises:
        AudioProcessingError: If the file cannot be decoded.
    """
    try:
        segment = _load(path)
    except (CouldntDecodeError, OSError, EOFError) as exc:
        raise AudioProcessingError(
            f"Could not read the sound of {path.name}: {exc}"
        ) from exc
    total = len(segment)
    if total <= 0 or buckets <= 0:
        return [], 0.0
    step = total / buckets
    values = [
        segment[int(index * step) : int((index + 1) * step)].rms
        for index in range(buckets)
    ]
    ranked = sorted(values)
    top = ranked[int(_TOP_SHARE * (len(ranked) - 1))] or ranked[-1] or 1
    return [min(1.0, value / top) for value in values], total / 1000


def picture(
    levels: list[float],
    kept: tuple[float, float] | None = None,
    width: int = WAVEFORM_WIDTH,
    height: int = WAVEFORM_HEIGHT,
) -> bytes:
    """Draw the loudness as bars, with the stretch to transcribe in colour.

    Args:
        levels: Values from :func:`loudness`.
        kept: Where the transcribed stretch starts and ends, as fractions of
            the recording (0–1); the whole recording when ``None``.
        width: Picture width in pixels.
        height: Picture height in pixels.

    Returns:
        A PNG with a transparent background.
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    low, high = kept or (0.0, 1.0)
    first, last = round(low * width), round(high * width)
    middle = height / 2
    reach = middle - 2
    for index, level in enumerate(levels):
        x = int(index * width / len(levels))
        half = max(1.0, max(0.0, level) ** _CURVE * reach)
        colour = _KEPT if first <= x < last else _LEFT_OUT
        draw.line([(x, middle - half), (x, middle + half)], fill=colour)
    shade = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shading = ImageDraw.Draw(shade)
    if first > 0:
        shading.rectangle([0, 0, first - 1, height - 1], fill=_SHADE)
    if last < width:
        shading.rectangle([last, 0, width - 1, height - 1], fill=_SHADE)
    image = Image.alpha_composite(image, shade)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
