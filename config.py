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


class Settings(BaseSettings):
    """Runtime settings loaded from the environment / ``.env`` file.

    Attributes:
        openai_api_key: API key used to authenticate with the OpenAI audio API.
        temp_dir: Directory for transient working files (segments, outputs).
        upload_dir: Directory where uploaded source files are stored.
        data_dir: Directory holding the SQLite history database.
        default_model: Transcription model used unless overridden in the UI.
        segment_duration_minutes: Length of each audio chunk in minutes.
    """

    openai_api_key: str = Field(..., description="OpenAI API key for transcription.")
    temp_dir: Path = Field(default=BASE_DIR / "temp")
    upload_dir: Path = Field(default=BASE_DIR / "uploads")
    data_dir: Path = Field(default=BASE_DIR / "data")
    default_model: str = Field(default=DEFAULT_MODEL)
    segment_duration_minutes: int = Field(default=DEFAULT_SEGMENT_DURATION_MINUTES)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: Any) -> None:
        """Create the working directories as soon as settings are loaded."""
        for directory in (self.temp_dir, self.upload_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (singleton).

    Returns:
        The validated settings. Accessing this the first time will raise a
        ``pydantic.ValidationError`` if required variables (e.g.
        ``OPENAI_API_KEY``) are missing.
    """
    return Settings()
