"""Application configuration and shared constants.

Centralises environment-driven settings (via Pydantic Settings) and the static
constants — supported formats and model lists — used across the app. This is the
single source of truth so the UI and the transcription pipeline never drift.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent

# Supported upload formats (used by the file uploader and source-type detection).
VIDEO_FORMATS: list[str] = [
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
AUDIO_FORMATS: list[str] = [
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

# Transcription models. The first entry is the default shown in the UI.
DEFAULT_MODEL: str = "gpt-4o-transcribe"
TRANSCRIPTION_MODELS: list[str] = [DEFAULT_MODEL, "whisper-1"]
# Models that can return per-segment timestamps (verbose_json) for SRT export.
TIMESTAMP_MODELS: set[str] = {"whisper-1"}

# OpenAI rejects transcription files larger than 25 MB per request.
MAX_SEGMENT_SIZE_MB: float = 25.0
# Default length of each audio chunk sent to the API.
DEFAULT_SEGMENT_DURATION_MINUTES: int = 10
# Target audio parameters for transcription (mono, 16 kHz is plenty for speech).
TARGET_CHANNELS: int = 1
TARGET_SAMPLE_RATE: int = 16000

# On-screen context: key frames pulled from a video and described by a vision
# model, so slides and shared screens end up in the transcript alongside speech.
DEFAULT_VISION_MODEL: str = "gpt-5.4-nano"
VISION_MODELS: list[str] = [DEFAULT_VISION_MODEL, "gpt-5.4-mini"]
# How different a frame must look from the previous one to count as a new scene
# (0–1). Far below the 0.3 usually quoted for film, because ffmpeg's score is
# tuned for natural footage: a measured full-screen slide change from navy to
# dark red scored only 0.077. Scene detection is treated as a cheap candidate
# generator here, not as the thing that guarantees coverage.
SCENE_THRESHOLD: float = 0.1
# A frame is taken at least this often even when nothing trips scene detection,
# which is what actually guarantees coverage of slow fades and subtle changes.
# Adjustable in the UI within this range.
FRAME_MAX_INTERVAL_SECONDS: float = 30.0
FRAME_INTERVAL_MIN_SECONDS: int = 5
FRAME_INTERVAL_MAX_SECONDS: int = 300
FRAME_INTERVAL_STEP_SECONDS: int = 5
# A single hard cut can trip scene detection several times in a row; one of
# those frames is enough.
FRAME_MIN_INTERVAL_SECONDS: float = 2.0
# Hard cap on frames per video: a backstop, not the main control. It has to stay
# well clear of ordinary use or it silently overrides the interval the user
# chose — at 40 an 83-minute meeting was pinned to the cap at every slider
# position. The binding constraint is wall-clock, not money: frames are captioned
# one request at a time, so 200 is a few minutes of waiting and roughly 5 cents.
# Override per-machine with FRAME_MAX_COUNT in .env.
DEFAULT_FRAME_MAX_COUNT: int = 200
# Edge length of the difference hash; 8 yields the usual 64-bit hash. A bigger
# hash is tempting but measurably worse here: slides are mostly flat, and the
# extra bits sample flat area where adjacent pixels tie, so they are noise.
HASH_SIZE: int = 8
# Hamming distance below which two frames count as the same picture. Deliberately
# tight: re-encodes of one static slide land at 0–3 bits, while distinct slides
# from the same template sit around 7. The costs are asymmetric — a false merge
# silently loses a slide forever, a false keep only spends a fraction of a cent.
# The known limit is that a slide differing only in a word or a number is ~1 bit
# away, i.e. inside the noise floor, so it will be treated as a duplicate.
FRAME_DUPLICATE_DISTANCE: int = 2
# JPEG quality for extracted frames (ffmpeg -qscale:v, 2 = best, 31 = worst).
FRAME_QUALITY: int = 4
# Image fidelity sent to the vision model. "low" downsamples server-side to
# 512x512 for a flat, predictable token cost — enough to read slide headings.
FRAME_DETAIL_LEVELS: list[str] = ["low", "high"]
DEFAULT_FRAME_DETAIL: str = "low"
# Approximate image tokens charged per frame, for the pre-run cost estimate.
VISION_TOKENS_PER_FRAME: dict[str, int] = {"low": 630, "high": 2300}
# A caption is short, but output tokens cost several times more than input ones,
# so leaving them out understated the estimate by roughly half.
VISION_OUTPUT_TOKENS_PER_FRAME: int = 80
# Indicative (input, output) price in USD per million tokens, for that estimate
# only. Checked 2026-07-20 — re-check OpenAI's pricing page if it looks wrong.
VISION_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
}

# Transcription providers (engine choice shown in the UI).
PROVIDER_OPENAI: str = "OpenAI API"
PROVIDER_LOCAL: str = "Local (offline)"
PROVIDERS: list[str] = [PROVIDER_OPENAI, PROVIDER_LOCAL]

# Local faster-whisper models with a short size/speed hint for the UI dropdown.
LOCAL_MODELS: dict[str, str] = {
    "tiny": "~75 MB · fastest, lowest quality",
    "base": "~145 MB · fast, decent quality",
    "small": "~480 MB · balanced",
    "medium": "~1.5 GB · slower, more accurate",
    "large-v3": "~3 GB · slowest, best accuracy",
    "large-v3-turbo": "~1.5 GB · accurate, 2–5× faster than large-v3",
}
DEFAULT_LOCAL_MODEL: str = "base"
# Approximate download sizes (MB), used to drive the download progress bar.
LOCAL_MODEL_SIZES_MB: dict[str, float] = {
    "tiny": 75,
    "base": 145,
    "small": 480,
    "medium": 1530,
    "large-v3": 3090,
    "large-v3-turbo": 1620,
}


class Settings(BaseSettings):
    """Runtime settings loaded from the environment / ``.env`` file.

    Attributes:
        openai_api_key: API key used to authenticate with the OpenAI audio API.
        temp_dir: Directory for transient working files (segments, outputs).
        upload_dir: Directory where uploaded source files are stored.
        data_dir: Directory holding the SQLite history database.
        whisper_model_dir: Download cache for local faster-whisper models.
        default_model: Transcription model used unless overridden in the UI.
        segment_duration_minutes: Length of each audio chunk in minutes.
        local_device: Device for local Whisper ("auto", "cpu" or "cuda").
        local_compute_type: Quantization for local Whisper (e.g. "int8").
        frame_max_count: Hard cap on screenshots described per video.
    """

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional; can also be entered in the UI).",
    )
    temp_dir: Path = Field(default=BASE_DIR / "temp")
    upload_dir: Path = Field(default=BASE_DIR / "uploads")
    data_dir: Path = Field(default=BASE_DIR / "data")
    whisper_model_dir: Path = Field(default=BASE_DIR / "models")
    default_model: str = Field(default=DEFAULT_MODEL)
    segment_duration_minutes: int = Field(default=DEFAULT_SEGMENT_DURATION_MINUTES)
    # CPU works everywhere; set LOCAL_DEVICE=cuda only with a working CUDA setup.
    local_device: str = Field(default="cpu")
    local_compute_type: str = Field(default="int8")
    # Raising this costs mostly time: frames are described one request at a time.
    frame_max_count: int = Field(default=DEFAULT_FRAME_MAX_COUNT, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: Any) -> None:
        """Create the working directories as soon as settings are loaded."""
        for directory in (
            self.temp_dir,
            self.upload_dir,
            self.data_dir,
            self.whisper_model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (singleton).

    Returns:
        The validated settings. The OpenAI key is optional, so this does not
        raise when it is missing — the UI lets the user provide it at runtime.
    """
    return Settings()
