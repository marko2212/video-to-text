"""Application configuration and shared constants.

Centralises environment-driven settings (via Pydantic Settings) and the static
constants — supported formats and model lists — used across the app. This is the
single source of truth so the UI and the transcription pipeline never drift.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
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

# Where the file to transcribe comes from. A browser upload passes through the
# app's memory several times over (see recordings.py); a file in a folder on
# this computer is read where it lies. Kept across sessions like the title mode.
SOURCE_UPLOAD: str = "upload"
SOURCE_FOLDER: str = "folder"
SOURCES: dict[str, str] = {
    SOURCE_UPLOAD: "Upload a file",
    SOURCE_FOLDER: "From a folder on this computer",
}
# How long the computer's own file window may stay open before the page stops
# waiting for it (the page is busy while it is open).
FILE_PICKER_TIMEOUT_SECONDS: float = 600.0
# The folder list shows each recording's length, read by ffprobe: about 0.13 s a
# file (mostly starting the program), so several are read at once, and each file
# only once until it changes.
RECORDING_PROBE_WORKERS: int = 8
# A file that takes longer (a network drive that hangs) is listed without one.
RECORDING_PROBE_TIMEOUT_SECONDS: float = 15.0
# Only the newest recordings of a folder get a length: a phone's call folder can
# hold thousands of files (1,500 would take about 25 s, eight at a time).
RECORDING_PROBE_LIMIT: int = 200

# Transcription models. The first entry is the default shown in the UI.
DEFAULT_MODEL: str = "gpt-4o-transcribe"
TRANSCRIPTION_MODELS: list[str] = [
    DEFAULT_MODEL,
    "gpt-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
]
# A short hint per model for the UI dropdown: release year, strength, price.
# At most ~360 px of text (about 55 characters) shows in the dropdown on a
# 1860 px wide window before it is cut off. gpt-4o-transcribe is billed per
# token: $0.0027-0.0033 per minute was recorded on the owner's first 4 runs
# (2026-09-29), and dense Serbian speech (more output tokens) works out near
# $0.0045 — hence ~$0.004. whisper-1's and gpt-transcribe's figures are their
# per-minute prices below; a test keeps them in step. gpt-4o-mini-transcribe
# costs half of gpt-4o-transcribe per token, hence ~$0.002. gpt-transcribe and
# the mini model were added on 2026-11-19 at the owner's request, after a loop
# in a gpt-4o-transcribe answer: gpt-transcribe replaces all the others when
# OpenAI retires them on 2027-02-26 (§3), and the mini model is the cheap one
# (OpenAI reported fewer invented words for its 2025-12-15 snapshot). Neither
# has been compared with the default on the owner's recordings yet.
TRANSCRIPTION_MODEL_NOTES: dict[str, str] = {
    "gpt-4o-transcribe": "2025, more accurate · ~$0.004/min",
    "gpt-transcribe": "2026, successor of all · $0.0045/min",
    "gpt-4o-mini-transcribe": "2025, cheaper, less accurate · ~$0.002/min",
    "whisper-1": "2023, exact timestamps & .srt · $0.006/min",
}
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
SEGMENT_MINUTES_BY_MODEL: dict[str, int] = {
    "gpt-4o-transcribe": 5,
    "gpt-4o-mini-transcribe": 5,
    "whisper-1": 10,
}
# Used for any model not listed above, until its output cap is known.
DEFAULT_SEGMENT_MINUTES: int = 5
# A response this long is treated as cut off by the output cap: the chunk is
# split in two and both halves are transcribed again.
OUTPUT_TOKEN_CAP_GUARD: int = 1900
# A chunk whose answer has far fewer words per minute than the run's other
# chunks is treated as incomplete too — the API can answer with part of a chunk
# and no error. Measured 2026-11-03 on a 52:10 company presentation sent as 11
# five-minute chunks to gpt-4o-transcribe, twice: every chunk came back at
# about 120 words per minute, except chunk 08 (39:38–44:33), which one run
# answered with almost nothing (1 word/min), and chunk 09 (44:33–49:43), which
# both runs answered at 68 words/min — 2:16 of speech in its middle was
# missing, as a local faster-whisper pass of the same audio showed (124
# words/min there). Neither reached the output cap. A chunk below this share of
# the running median of the chunks answered so far is split in two and both
# halves are sent again, once; a half that is still thin is kept and marked in
# the transcript.
THIN_CHUNK_SHARE: float = 0.5
# The guard stays silent while the median itself is under this: a recording
# that is mostly silence, music or a demo has no typical rate to compare with,
# and re-sending its quiet chunks would only cost money.
THIN_CHUNK_MIN_MEDIAN_WORDS_PER_MINUTE: float = 40.0
# A part answered before the run had a typical rate — the first one, or the
# first few of a recording that starts slowly — is judged once the other parts
# are back, and sent again in halves only if it also holds sound: at least this
# share of its quarter-second windows within THIN_RECHECK_LOUD_DB of the loud
# level of the parts that answered normally. A silent start (a recording left
# running before the meeting) has none, and re-sending it would only cost
# money. Measured 2026-11-15 on the four real recordings whose first part came
# back nearly empty (1–50 words/min against 88–141): 37–67 % of their windows
# were that loud, against 40–80 % in the parts that answered normally.
THIN_RECHECK_MIN_SOUND_SHARE: float = 0.2
THIN_RECHECK_LOUD_DB: float = 20.0
# Words per minute are counted over a part's speech, not its whole length, and a
# part with less speech than this is not judged at all: a rate over a few
# seconds says nothing. Measured 2026-11-16 on a 78-minute meeting: the first
# half of its first part (0:00–2:30) came back with 8 words and was marked
# "Speech may be missing", yet 143 of its 150 seconds were silence.
THIN_MIN_SPEECH_SECONDS: float = 30.0
# A looped answer: the model repeats a stretch of text instead of transcribing
# what follows, so the answer has as many words as usual (or more) and the
# thin-part guard cannot see it. Measured as the share of LOOP_NGRAM_WORDS-word
# windows that already appeared earlier in the same answer. On 2026-11-19, over
# 298 saved parts of 51 runs with 50 words or more: the median share was 0 and
# the 90th percentile 0; seven parts reached 0.15–0.36, and every one was a loop
# (one sentence ten times in a 19-minute call, "file" 103 times, a passage
# repeated four times), while the next highest real part was 0.09 (three
# genuine repeats of a request on a call). A looped answer is handled like a
# thin one: its halves are sent again, and a half still looped is kept and
# marked. Answers shorter than LOOP_MIN_WORDS are never judged.
LOOP_NGRAM_WORDS: int = 8
LOOP_REPEAT_SHARE: float = 0.12
LOOP_MIN_WORDS: int = 50
# Silence within a part is measured on quarter-second windows, like ffmpeg's
# silencedetect: a window is silent when it is SILENCE_BELOW_PEAK_DB quieter
# than the part's loudest window, with that line kept between
# SILENCE_DBFS_FLOOR and SILENCE_DBFS_CEILING (so a quiet recording's speech is
# not silence, and room noise under -60 dBFS always is), and only runs of
# SILENCE_MIN_SECONDS or more count — shorter gaps are pauses between words.
SILENCE_BELOW_PEAK_DB: float = 30.0
SILENCE_DBFS_FLOOR: float = -60.0
SILENCE_DBFS_CEILING: float = -40.0
SILENCE_MIN_SECONDS: float = 1.0
# Models that take OpenAI's ``chunking_strategy`` (server-side voice activity
# detection); THIN_RETRY_AUTO_CHUNKING uses it only with these.
CHUNKING_STRATEGY_MODELS: set[str] = {"gpt-4o-transcribe", "gpt-4o-mini-transcribe"}
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
# How long OpenAI takes to answer, in seconds per minute of audio sent: the
# measure behind the "about M:SS left" of a run until its own first part is
# back; after that the run's own parts are the measure. Read from the owner's
# history on 2026-11-15 (159 parts of 19 runs): 2.3 s per minute of audio at
# the median, 1.8 to 3.4 for four parts in five and 8.5 for the slowest — a
# five-minute part answers in about 11 s. Within one run the parts usually
# differ by a factor of 1.3 to 1.7, so the figure is an estimate, said as one.
OPENAI_SECONDS_PER_AUDIO_MINUTE: float = 2.3
# The time left is rounded to this many seconds, and never shown as less.
REMAINING_ROUND_SECONDS: int = 5
# How the page finds that figure in a progress message, to count it down in the
# browser until the next estimate arrives (a regular expression Python and
# JavaScript both read; it must match ``transcribe.format_remaining``).
REMAINING_PATTERN: str = r"about ((?:\d+:)?\d+:\d\d) left"
# Target audio parameters for transcription (mono, 16 kHz is plenty for speech).
TARGET_CHANNELS: int = 1
TARGET_SAMPLE_RATE: int = 16000
# The programs that read every recording; any ffmpeg build brings both. Without
# ffmpeg nothing can be prepared, without ffprobe the folder list shows no
# lengths and on-screen context cannot read the video.
FFMPEG_TOOLS: tuple[str, ...] = ("ffmpeg", "ffprobe")
# The one-line Windows install the page and the README suggest (no admin
# rights; winget puts both programs on PATH).
FFMPEG_WINDOWS_INSTALL: str = "winget install --id Gyan.FFmpeg.Essentials -e"
# Audio formats Chromium-based browsers cannot play. They, and every video, get
# a small MP3 preview for the player instead of the raw file or the big WAV.
BROWSER_UNPLAYABLE_AUDIO: set[str] = {"amr", "wma", "aiff", "aif"}
# Bitrate of that preview: plenty for checking speech, ~22 MB for 90 minutes.
PREVIEW_BITRATE: str = "32k"
# Playable uploads bigger than this get the preview too: the player re-reads its
# file on every click, which cost ~0.5 s per click with a 150 MB WAV.
PREVIEW_ABOVE_MB: float = 25.0
# A picture of the sound between the trim slider and the player: the loudness
# of this many equal stretches of the recording, one bar each, so an empty
# stretch — a recording left running for hours after the meeting ended — can be
# seen and left out of the run with the slider.
WAVEFORM_BUCKETS: int = 1200
# Size of that picture in pixels (drawn again at each move of the slider).
WAVEFORM_WIDTH: int = 1200
WAVEFORM_HEIGHT: int = 100
# The stretch to transcribe is chosen in whole seconds. The slider shows times
# on a clock, so a recording of a day or longer gets no slider.
TRIM_STEP_SECONDS: int = 1
TRIM_CLOCK_LIMIT_SECONDS: int = 24 * 3600

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
# (input, output) price in USD per million tokens of the chat models used for
# screenshots and titles — used for the pre-run estimate and for the recorded
# cost of each run. Checked 2026-09-25 on developers.openai.com/api/docs/pricing;
# re-check if a figure looks wrong.
CHAT_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
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

# AI title: after a run, a chat model names the transcript, and the name is used
# in History and for downloaded files. A preference kept in the history
# database, off by default. The title is stored beside the original file name,
# never over it, so the mode only decides how the two are shown.
TITLE_MODE_OFF: str = "off"
TITLE_MODE_REPLACE: str = "replace"
TITLE_MODE_APPEND: str = "append"
TITLE_MODES: dict[str, str] = {
    TITLE_MODE_OFF: "Off",
    TITLE_MODE_REPLACE: "Replace the file name",
    TITLE_MODE_APPEND: "Add to the end of the file name",
}
DEFAULT_TITLE_MODEL: str = "gpt-5.4-nano"
TITLE_MODELS: list[str] = [DEFAULT_TITLE_MODEL, "gpt-5.4-mini"]
# A title is a few words; anything longer is cut at a word boundary.
TITLE_MAX_CHARS: int = 80
# Transcript characters sent for a title: about four hours of speech, roughly
# 60k tokens — about a cent on gpt-5.4-nano. A longer transcript is sent as its
# beginning and its end.
TITLE_MAX_TRANSCRIPT_CHARS: int = 200_000
# Text tokens in an hour of transcript, for the cost hint next to the setting:
# dense Serbian speech measured 2,450-2,920 tokens per 10 minutes (2026-09-25).
TITLE_TOKENS_PER_HOUR: int = 17_000
# Read timeout of a title request. nano answers in seconds; the transcription
# timeout (5 minutes, per retry) would leave the page waiting far too long.
TITLE_TIMEOUT_SECONDS: float = 60.0
# A History title click this soon after the last title of the same entry is
# taken as the second half of a double click, not as a request to pay again.
TITLE_REPEAT_GUARD_SECONDS: float = 2.0

# Saved parts of unfinished runs hold transcript text; they are deleted after
# this many days without use, even if nobody cleans temporary files.
CHECKPOINT_MAX_AGE_DAYS: float = 14.0
# Scratch folders of a run killed mid-way (audio chunks, screenshots) are deleted
# after this long. Generous: a live screenshot folder can sit unchanged for over
# an hour while its frames are described.
SCRATCH_MAX_AGE_HOURS: float = 24.0
# Working copies of recordings — uploads stored in uploads/, and a video's
# extracted WAV and the player's MP3 in temp/ — are deleted after this long, each
# time the page is opened. Nothing else removed them but the "Clean temporary
# files" button: 3.4 GB piled up in two weeks (2026-09-30).
WORKING_COPY_MAX_AGE_HOURS: float = 24.0
# The working copies in temp/, by extension. Transcripts (.txt, .srt) are left
# alone for now — the owner's call (2026-09-30), to be decided later.
WORKING_COPY_TEMP_SUFFIXES: tuple[str, ...] = (".wav", ".mp3")

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
    "large-v3-turbo": "~1.5 GB · near large-v3, 2–5× faster",
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
# How much faster than real time a local model transcribes on the owner's CPU
# (int8): `small` did a 52-minute recording in about 13 minutes (2026-11-03).
# `base` and `tiny` are faster; the others were not measured.
LOCAL_SPEED_FACTORS: dict[str, float] = {"small": 3.9}
# Batched decoding, an opt-in box on the page: faster-whisper cuts the recording
# at pauses into pieces of up to 30 s and writes this many at once. Each piece is
# written without the text before it and without a second try.
LOCAL_BATCH_SIZE: int = 8

# Cross-check of an OpenAI transcript against a local Whisper pass (opt-in for
# gpt-4o-transcribe, whose paragraph times are estimated). The two texts are
# aligned word by word; an API paragraph takes the local pass's time of its
# words once a run of at least this many words in a row matches — shorter
# matches ("and then the") occur anywhere.
VERIFY_MIN_ANCHOR_WORDS: int = 4
# A stretch the local pass has and the API answer lacks is added to the
# transcript only from this many words: shorter differences are wording, not
# missing speech (the two engines disagree on a word here and there).
VERIFY_MIN_GAP_WORDS: int = 25
# An added stretch is broken into paragraphs about this long, at sentence starts.
VERIFY_GAP_PARAGRAPH_SECONDS: float = 40.0
# The local model offered first for the cross-check: the smallest that reads
# well enough to align on; `tiny` misses too many words.
DEFAULT_VERIFY_MODEL: str = DEFAULT_LOCAL_MODEL
# After an OpenAI run, the parts still marked incomplete (thin, looped or at the
# output cap once sent again in halves) are transcribed by local Whisper — those
# parts only — and what the answer lacked is added from it (`LOCAL_FILL`, on).
# `medium`: what the owner ran by hand for exactly this on real meetings (it
# filled 25:05–27:34 of a 48-minute meeting, 2026-11-20); smaller models read
# Serbian worse. It runs at about 0.7× the length on the owner's CPU, so a
# 2:30 half takes under two minutes. A first use downloads it (~1.5 GB).
DEFAULT_LOCAL_FILL_MODEL: str = "medium"

# The application log: every run leaves one line per chunk sent (bounds, size,
# request time and id, tokens, words per minute, guard flags), so an answer
# that came back incomplete can be found afterwards. The console shows the
# same lines but keeps nothing. Rotated so it cannot grow without bound.
LOG_DIRNAME: str = "logs"
LOG_FILENAME: str = "app.log"
LOG_FILE_MAX_BYTES: int = 5 * 1024 * 1024
LOG_FILE_BACKUPS: int = 5
# The parts of every finished run (one JSON per chunk, with its text and its
# record) are kept beside the history, in a folder named after the history
# row, until that row is deleted: history keeps only the rendered transcript
# and the run's totals, which is not enough to tell which chunk lost speech.
RUNS_DIRNAME: str = "runs"

# The command line (cli.py) — the same run as the page, started by a script or
# an AI agent. Its engine names, and how a transcript is named when it is put
# beside its recording: transcript_<the recording's name>.txt.
CLI_ENGINES: dict[str, str] = {"openai": PROVIDER_OPENAI, "local": PROVIDER_LOCAL}
TRANSCRIPT_PREFIX: str = "transcript_"
# On the page nothing is sent to OpenAI before the Start click. The command
# line has no click, so a run estimated above this is refused until it is
# repeated with --yes: about 2 h 45 min of gpt-4o-transcribe at OpenAI's own
# per-minute estimate. --max-cost changes it for one call.
CLI_MAX_COST_USD: float = 1.0
# How many recordings `latest` lists unless told otherwise.
CLI_LATEST_COUNT: int = 10


class Settings(BaseSettings):
    """Runtime settings loaded from the environment / ``.env`` file.

    Attributes:
        openai_api_key: API key used to authenticate with the OpenAI audio API.
        temp_dir: Directory for transient working files (segments, outputs).
        upload_dir: Directory where uploaded source files are stored.
        data_dir: Directory holding the SQLite history database.
        log_dir: Directory of the rotating application log; ``data_dir/logs``
            unless ``LOG_DIR`` says otherwise.
        whisper_model_dir: Download cache for local faster-whisper models.
        segment_duration_minutes: Chunk length for every model, overriding
            the per-model defaults; unset means per model.
        local_device: Device for local Whisper ("auto", "cpu" or "cuda").
        local_compute_type: Quantization for local Whisper (e.g. "int8").
        frame_max_count: Hard cap on screenshots described per video.
        serbian_latin: Rewrite Serbian Cyrillic in transcripts as Latin (off).
        allow_local_files: Offer recordings from a folder on this computer.
        thin_retry_auto_chunking: Before splitting a thin part, send it once
            more whole with ``chunking_strategy="auto"`` and log what changed.
        local_fill: After an OpenAI run, transcribe the parts still marked
            incomplete with local Whisper and add what the answer lacked
            (needs the offline backend; on).
        local_fill_model: The local model for that (``medium``).
    """

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional; can also be entered in the UI).",
    )
    temp_dir: Path = Field(default=BASE_DIR / "temp")
    upload_dir: Path = Field(default=BASE_DIR / "uploads")
    data_dir: Path = Field(default=BASE_DIR / "data")
    # Relative to the history folder by default, so it is backed up with it
    # and survives "Clean temporary files"; resolved in model_post_init.
    log_dir: Path | None = Field(default=None)
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
    # On for the usual local use: the page can list and read any folder of the
    # computer it runs on. Turn it off (ALLOW_LOCAL_FILES=false) wherever other
    # people can open the page — a hosted copy would show them its server's disk.
    allow_local_files: bool = Field(default=True)
    # Off by default: an experiment, one more paid request per thin part. On
    # (THIN_RETRY_AUTO_CHUNKING=true), a thin part of a model that takes it is
    # sent once more whole with the server's voice-activity chunking before it
    # is split; the log says whether that alone brought the speech back.
    thin_retry_auto_chunking: bool = Field(default=False)
    # On by default (the owner's choice, 2026-11-20): a part that still came
    # back incomplete is filled from local Whisper, that part only, instead of
    # a second pass over the whole recording started by hand. Free; it costs
    # the time of those parts. LOCAL_FILL=false leaves them marked ⚠️.
    local_fill: bool = Field(default=True)
    local_fill_model: str = Field(default=DEFAULT_LOCAL_FILL_MODEL)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `SEGMENT_DURATION_MINUTES=` in .env means "not set", not a startup error.
        env_ignore_empty=True,
    )

    @field_validator("local_fill_model")
    @classmethod
    def _a_local_model(cls, value: str) -> str:
        """Refuse a ``LOCAL_FILL_MODEL`` that is not a local model's name."""
        if value not in LOCAL_MODELS:
            raise ValueError(
                f"LOCAL_FILL_MODEL must be one of {', '.join(LOCAL_MODELS)}"
            )
        return value

    def model_post_init(self, __context: Any) -> None:
        """Create the working directories as soon as settings are loaded."""
        if self.log_dir is None:
            self.log_dir = self.data_dir / LOG_DIRNAME
        for directory in (
            self.temp_dir,
            self.upload_dir,
            self.data_dir,
            self.log_dir,
            self.whisper_model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Return the application log file (``app.log`` in ``log_dir``)."""
        return (self.log_dir or self.data_dir / LOG_DIRNAME) / LOG_FILENAME

    @property
    def runs_dir(self) -> Path:
        """Return the folder that keeps the parts of finished runs, by row id."""
        return self.data_dir / RUNS_DIRNAME


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (singleton).

    Returns:
        The validated settings. The OpenAI key is optional, so this does not
        raise when it is missing — the UI lets the user provide it at runtime.
    """
    return Settings()
