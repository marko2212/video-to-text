"""Lightweight logging setup.

Provides :func:`get_logger`, returning a stdlib logger with a readable console
format, and :func:`enable_file_log`, which adds a rotating file beside the
history database. Plain text logging (not structured JSON) is the right fit for
a local Streamlit app — there is no log aggregator to consume JSON. The file
exists because the console keeps nothing: when a paid answer came back
incomplete, the per-chunk lines are the only trace of what was sent and what
came back (see ``transcribe``).

The API key never appears in a message on purpose; as a second line of
defence, anything shaped like an OpenAI key is blanked before it is written.
"""

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_FILE_BACKUPS, LOG_FILE_MAX_BYTES, get_settings

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# OpenAI keys start with ``sk-`` (project keys ``sk-proj-``); anything that
# long after the prefix is blanked, whatever it was.
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
_REDACTED = "sk-***"


class _RedactingFormatter(logging.Formatter):
    """A formatter that blanks anything shaped like an API key."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and blank key-shaped strings in the result.

        Args:
            record: The record to format.

        Returns:
            The formatted line, with any ``sk-…`` token replaced.
        """
        return _KEY_PATTERN.sub(_REDACTED, super().format(record))


def _configure_console() -> None:
    """Attach the console handler to the root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    root = logging.getLogger()
    # basicConfig leaves a root that already has handlers (a test runner's)
    # alone, level included; INFO is what the per-chunk lines are written at.
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    for handler in root.handlers:
        handler.setFormatter(_RedactingFormatter(_LOG_FORMAT))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` writing to the console at INFO level (and to the
        application log once :func:`enable_file_log` has been called).
    """
    _configure_console()
    return logging.getLogger(name)


def enable_file_log(path: Path | None = None) -> Path:
    """Write every log line to a rotating file as well as the console.

    Called once by the app at start-up; anything else that runs the pipeline
    (a script, a test) opts in explicitly, so importing a module never writes
    into the real ``data/``. A second call with the same path does nothing; a
    call with another path replaces the file (the tests move it per test).

    Args:
        path: The log file; ``Settings.log_path`` by default.

    Returns:
        The file being written.
    """
    _configure_console()
    target = Path(path) if path is not None else get_settings().log_path
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, RotatingFileHandler):
            if Path(handler.baseFilename) == target.resolve():
                return target
            root.removeHandler(handler)
            handler.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        target,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(_RedactingFormatter(_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    return target
