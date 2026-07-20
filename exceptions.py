"""Domain-specific exceptions for the transcription app.

Using specific exception types (instead of bare ``Exception``) lets callers
distinguish audio-preparation failures from transcription failures and keeps
error messages meaningful in the UI.
"""


class AppError(Exception):
    """Base class for all application-specific errors."""


class AudioProcessingError(AppError):
    """Raised when extracting or preparing audio fails (e.g. ffmpeg error)."""


class TranscriptionError(AppError):
    """Raised when the transcription pipeline or the OpenAI request fails."""


class VisualContextError(AppError):
    """Raised when extracting or describing video key frames fails."""
