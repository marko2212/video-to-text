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
# Length of each audio chunk sent to the API, per model. The gpt-4o family stops
# writing at about 2,000 output tokens per request, and dense Serbian speech
# reaches 2,450-2,920 tokens per 10 minutes, so 10-minute chunks silently lost
# their last minute or so (measured 2026-09-25). Five minutes stays well under
# the cap; whisper-1 has no such cap. The price follows the audio length either
# way (per minute, or audio tokens), so chunk length does not change it.
SEGMENT_MINUTES_BY_MODEL: dict[str, int] = {"gpt-4o-transcribe": 5, "whisper-1": 10}
# Used for any model not listed above, until its output cap is known.
DEFAULT_SEGMENT_MINUTES: int = 5
# A response this long is treated as cut off by the output cap: the chunk is
# split in two and both halves are transcribed again.
OUTPUT_TOKEN_CAP_GUARD: int = 1900
# Chunks are never split below this length, so a runaway split cannot loop.
MIN_SPLIT_SECONDS: float = 30.0
# Chunk boundaries move to the quietest moment within this distance of the
# nominal cut, so a boundary falls in a pause rather than mid-word.
CUT_SEARCH_SECONDS: float = 10.0
# A final chunk shorter than this is merged into the previous one. Tiny tails
# used to fail the whole run after the earlier chunks had been paid for.
MIN_TAIL_SECONDS: float = 10.0
# The SDK retries connection errors, timeouts, 408/409/429 and 5xx (honouring
# Retry-After). Errors that cannot succeed — bad key, no credit — are never
# retried; see openai_api.py.
OPENAI_MAX_RETRIES: int = 3
# Read timeout per request (a 5-minute chunk normally answers in seconds), and a
# short connect timeout so an offline machine fails fast.
OPENAI_TIMEOUT_SECONDS: float = 300.0
OPENAI_CONNECT_TIMEOUT_SECONDS: float = 10.0
# Target audio parameters for transcription (mono, 16 kHz is plenty for speech).
TARGET_CHANNELS: int = 1
TARGET_SAMPLE_RATE: int = 16000
# Audio formats Chromium-based browsers cannot play. They, and every video, get
# a small MP3 preview for the player instead of the raw file or the big WAV.
BROWSER_UNPLAYABLE_AUDIO: set[str] = {"amr", "wma", "aiff", "aif"}
# Bitrate of that preview: plenty for checking speech, ~22 MB for 90 minutes.
PREVIEW_BITRATE: str = "32k"
# Playable uploads bigger than this get the preview too: the player re-reads its
# file on every click, which cost ~0.5 s per click with a 150 MB WAV.
PREVIEW_ABOVE_MB: float = 25.0

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
# one request at a time, so 200 is a few minutes of waiting and about 12 cents
# on gpt-5.4-nano for Full HD frames (about 45 on gpt-5.4-mini).
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
# Containers without an index (ffprobe format names): seeking in them lands up
# to a keyframe interval late, so a frame cannot be re-extracted exactly and the
# scan keeps every frame's own JPEG (.ts/.mts/.m2ts, .mpg/.mpeg/.vob).
FRAME_UNINDEXED_FORMATS: tuple[str, ...] = ("mpegts", "mpeg", "vob")
# Image fidelity sent to the vision model as `detail`. On the gpt-5.4 models it
# does not change the token count (measured 2026-09-25, see
# VISION_TOKENS_PER_FRAME); the cost follows the frame's pixel count.
FRAME_DETAIL_LEVELS: list[str] = ["low", "high"]
DEFAULT_FRAME_DETAIL: str = "low"
# Input tokens charged per frame (image + prompt), for the pre-run estimate.
# Measured 2026-09-25 with gpt-5.4-nano and -mini: a 1920x1080 frame costs 2,519
# tokens and a 1280x720 one 1,175 — roughly proportional to the pixel count —
# and `detail` makes no difference. Screen recordings are usually Full HD, so
# that figure is used; the old 630 understated the cost by half or more.
VISION_TOKENS_PER_FRAME: dict[str, int] = {"low": 2519, "high": 2519}
# The same two measurements as a line, for frames whose size is known (from the
# video scan): tokens ≈ fixed part + per-megapixel part × megapixels.
VISION_TOKENS_FIXED: float = 100.0
VISION_TOKENS_PER_MEGAPIXEL: float = 1166.7
# A caption is short, but output tokens cost several times more than input ones,
# so leaving them out understated the estimate by roughly half.
VISION_OUTPUT_TOKENS_PER_FRAME: int = 80
# (input, output) price in USD per million tokens — used for the pre-run
# estimate and for the recorded cost of each run. Checked 2026-09-25 on
# developers.openai.com/api/docs/pricing; re-check if a figure looks wrong.
VISION_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
}
# Transcription prices, same source and date. Token-billed models: (input,
# output) USD per million tokens, the input price covering the audio tokens.
TRANSCRIPTION_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-transcribe": (2.50, 10.00),
    "gpt-4o-mini-transcribe": (1.25, 5.00),
}
# Per-minute prices: how whisper-1 and gpt-transcribe are billed, and OpenAI's
# own estimate for the token-billed models when an answer carries no usage.
TRANSCRIPTION_PRICE_PER_MINUTE: dict[str, float] = {
    "whisper-1": 0.006,
    "gpt-transcribe": 0.0045,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
}
# Used for a transcription model missing from both tables, so its cost is at
# least in the right range; the record is marked as an estimate.
TRANSCRIPTION_FALLBACK_PRICE_PER_MINUTE: float = 0.006

# Saved parts of unfinished runs hold transcript text; they are deleted after
# this many days without use, even if nobody cleans temporary files.
CHECKPOINT_MAX_AGE_DAYS: float = 14.0
# Scratch folders of a run killed mid-way (audio chunks, screenshots) are deleted
# after this long. Generous: a live screenshot folder can sit unchanged for over
# an hour while its frames are described.
SCRATCH_MAX_AGE_HOURS: float = 24.0

# While a job runs, a hidden fragment checks this often whether the run was
# stopped (toolbar Stop), so the disabled controls come back without a reload.
JOB_WATCHDOG_SECONDS: float = 2.0

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
        segment_duration_minutes: Chunk length for every model, overriding
            the per-model defaults; unset means per model.
        local_device: Device for local Whisper ("auto", "cpu" or "cuda").
        local_compute_type: Quantization for local Whisper (e.g. "int8").
        frame_max_count: Hard cap on screenshots described per video.
        serbian_latin: Rewrite Serbian Cyrillic in transcripts as Latin (off).
    """

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional; can also be entered in the UI).",
    )
    temp_dir: Path = Field(default=BASE_DIR / "temp")
    upload_dir: Path = Field(default=BASE_DIR / "uploads")
    data_dir: Path = Field(default=BASE_DIR / "data")
    whisper_model_dir: Path = Field(default=BASE_DIR / "models")
    # Capped at 15 so a chunk stays under the 25 MB request limit.
    segment_duration_minutes: int | None = Field(default=None, ge=1, le=15)
    # CPU works everywhere; set LOCAL_DEVICE=cuda only with a working CUDA setup.
    local_device: str = Field(default="cpu")
    local_compute_type: str = Field(default="int8")
    # Raising this costs mostly time: frames are described one request at a time.
    frame_max_count: int = Field(default=DEFAULT_FRAME_MAX_COUNT, ge=1)
    # Off by default: the app keeps the script the model returned. Macedonian
    # shares the letters that identify Serbian Cyrillic and would be rewritten in
    # Serbian Latin, and a transcript read by another AI needs no single script.
    # On, Serbian Cyrillic is rewritten in Latin (the API picks the script per
    # chunk, so a long Serbian meeting can alternate between the two).
    serbian_latin: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `SEGMENT_DURATION_MINUTES=` in .env means "not set", not a startup error.
        env_ignore_empty=True,
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
