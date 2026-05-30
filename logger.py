"""Lightweight logging setup.

Provides a single helper, :func:`get_logger`, returning a stdlib logger with a
readable console format. Plain text logging (not structured JSON) is the right
fit for a local Streamlit app — there is no log aggregator to consume JSON.
"""

import logging

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` writing to the console at INFO level.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
        _CONFIGURED = True
    return logging.getLogger(name)
