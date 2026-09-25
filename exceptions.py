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


class IncompleteTranscriptionError(TranscriptionError):
    """Raised when a run stopped part-way, after some chunks were transcribed.

    The finished chunks are checkpointed and a partial transcript, ending with a
    visible marker, has already been written, so the caller can show it and a
    rerun of the same file resumes where this one stopped.

    Attributes:
        completed: Chunks transcribed (in this run or restored from checkpoints).
        total: Chunks in the recording.
    """

    def __init__(self, message: str, completed: int, total: int) -> None:
        """Store the progress reached alongside the message.

        Args:
            message: The cause, phrased for the user.
            completed: Chunks transcribed before the failure.
            total: Chunks in the recording.
        """
        super().__init__(message)
        self.completed = completed
        self.total = total


class VisualContextError(AppError):
    """Raised when extracting or describing video key frames fails."""


class OpenAIAccountError(AppError):
    """Raised when OpenAI refuses the account itself: bad key, no access, no credit.

    Deliberately not a subclass of :class:`TranscriptionError` or
    :class:`VisualContextError`: code that tolerates one failed chunk or frame
    must not swallow this, because every further request would fail the same way.
    """
