"""Domain-specific exceptions for the transcription app.

Using specific exception types (instead of bare ``Exception``) lets callers
distinguish audio-preparation failures from transcription failures and keeps
error messages meaningful in the UI.
"""


class AppError(Exception):
    """Base class for all application-specific errors.

    Attributes:
        spent: What the failed step had already paid for (a usage summary, see
            :mod:`usage`), when known; ``None`` otherwise.
    """

    spent: dict | None = None


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
        spent: What the saved parts cost (a usage summary), or ``None``.
    """

    def __init__(
        self,
        message: str,
        completed: int,
        total: int,
        spent: dict | None = None,
    ) -> None:
        """Store the progress reached alongside the message.

        Args:
            message: The cause, phrased for the user.
            completed: Chunks transcribed before the failure.
            total: Chunks in the recording.
            spent: Usage summary of the requests paid for so far.
        """
        super().__init__(message)
        self.completed = completed
        self.total = total
        self.spent = spent


class VisualContextError(AppError):
    """Raised when extracting or describing video key frames fails.

    Attributes:
        usage_record: The usage record of a request that was paid for although
            it produced no description (an answer without choices), else ``None``.
    """

    usage_record: dict | None = None


class TitleError(AppError):
    """Raised when the AI title of a finished transcript cannot be made.

    Attributes:
        usage_record: The usage record of a request that was paid for although
            it produced no title (an empty answer), else ``None``.
    """

    usage_record: dict | None = None


class OpenAIAccountError(AppError):
    """Raised when OpenAI refuses the account itself: bad key, no credit, no billing.

    Deliberately not a subclass of :class:`TranscriptionError` or
    :class:`VisualContextError`: code that tolerates one failed chunk or frame
    must not swallow this, because every further request would fail the same way.
    """
