"""An AI title for a finished transcript.

After a run, a chat model reads the transcript and names it in a few words, in
the language the speakers use. The title is stored beside the original file
name, never over it: the title mode (see :mod:`config`) only decides how the two
are shown in History and in the names of downloaded files, so changing the mode
applies to earlier rows as well and never pays for a title twice. Like the rest
of the pipeline this module is UI-agnostic.
"""

from typing import Any

from openai import OpenAIError

import openai_api
import usage
from config import (
    CHAT_PRICE_PER_MTOK,
    TITLE_MAX_CHARS,
    TITLE_MAX_TRANSCRIPT_CHARS,
    TITLE_MODE_APPEND,
    TITLE_MODE_REPLACE,
    TITLE_TOKENS_PER_HOUR,
)
from exceptions import TitleError

_PROMPT = (
    "You name recordings. The user message is the transcript of one. Reply with "
    "a short, specific title for it: at most 8 words, in the language and script "
    "the speakers use, naming what it is about rather than its format (not just "
    "'Meeting', 'Call' or 'Transcript'). Reply with the title only: no quotes, "
    "no trailing period."
)
# A title is a few words; the rest is headroom, since a reasoning model's
# output tokens include its reasoning.
_MAX_TITLE_TOKENS = 200
# Labels a model sometimes puts in front of the title despite the prompt.
_LABELS = {"title", "naslov", "наслов"}
_QUOTES = "\"'`“”„«»‘’‚"
# Characters Windows does not allow in a file name (the backslash is chr(92)),
# and control characters. The title becomes a file name on download.
_FORBIDDEN = set('<>"/|?*') | {chr(92)} | {chr(code) for code in range(32)}
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{n}" for n in range(1, 10)),
    *(f"LPT{n}" for n in range(1, 10)),
}


def clean_title(text: str) -> str | None:
    """Turn a model's answer into a title that is safe as a file name.

    Args:
        text: The answer.

    Returns:
        The first non-empty line without label, quotes, trailing period or
        characters Windows forbids in a file name, at most
        ``TITLE_MAX_CHARS`` long (cut at a word); ``None`` if nothing is left.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    title = lines[0]
    label, colon, rest = title.partition(":")
    if colon and label.strip().strip("*#").strip().lower() in _LABELS:
        title = rest
    # "Project X: kickoff" reads better as "Project X - kickoff" than without it.
    title = title.replace(":", " -")
    title = "".join(" " if char in _FORBIDDEN else char for char in title)
    title = " ".join(title.split()).strip(_QUOTES + "*# ").rstrip(". ")
    if len(title) > TITLE_MAX_CHARS:
        cut = title[:TITLE_MAX_CHARS]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        title = cut.rstrip(" -,;.")
    if title.upper() in _RESERVED:
        title = f"{title} recording"
    return title or None


def _excerpt(transcript: str) -> str:
    """Return the part of a transcript that is sent for its title.

    Args:
        transcript: The whole transcript.

    Returns:
        The transcript, or its beginning and end when it is longer than
        ``TITLE_MAX_TRANSCRIPT_CHARS``.
    """
    if len(transcript) <= TITLE_MAX_TRANSCRIPT_CHARS:
        return transcript
    half = TITLE_MAX_TRANSCRIPT_CHARS // 2
    return f"{transcript[:half]}\n\n[…]\n\n{transcript[-half:]}"


def make_title(transcript: str, api_key: str, model: str) -> tuple[str, dict[str, Any]]:
    """Ask a chat model for the title of a transcript.

    Args:
        transcript: The finished transcript.
        api_key: OpenAI API key.
        model: Chat model name (see ``TITLE_MODELS``).

    Returns:
        The cleaned title, and the usage record of the request.

    Raises:
        OpenAIAccountError: If the key or the account is refused.
        TitleError: If the request fails for any other reason, or the answer
            holds no usable title (its usage record is attached, since it was
            paid for all the same).
    """
    if not transcript.strip():
        raise TitleError("the transcript is empty")
    client = openai_api.make_client(api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=_MAX_TITLE_TOKENS,
            messages=[
                {"role": "developer", "content": _PROMPT},
                {"role": "user", "content": _excerpt(transcript)},
            ],
        )
    except OpenAIError as exc:
        raise openai_api.translate_error(exc, TitleError) from exc
    finally:
        client.close()

    spent = usage.title_record(response, model)
    answer = response.choices[0].message.content if response.choices else None
    title = clean_title(answer or "")
    if title is None:
        error = TitleError("OpenAI returned no title")
        error.usage_record = spent
        raise error
    return title, spent


def estimate_cost_per_hour(model: str) -> float | None:
    """Estimate what a title costs per hour of recording, for the setting's hint.

    Args:
        model: Chat model name.

    Returns:
        The approximate cost in USD, or ``None`` if the model has no price.
    """
    prices = CHAT_PRICE_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (
        TITLE_TOKENS_PER_HOUR * input_price + _MAX_TITLE_TOKENS * output_price
    ) / 1_000_000


def display_name(stem: str, title: str | None, mode: str) -> str | None:
    """Return the name a titled transcript is shown and downloaded under.

    Args:
        stem: The original file name without its extension.
        title: The stored AI title, if the row has one.
        mode: The title mode (``TITLE_MODES``).

    Returns:
        The title, or the stem followed by the title; ``None`` when the mode is
        off or there is no title, so the caller keeps its usual name.
    """
    if not title:
        return None
    if mode == TITLE_MODE_REPLACE:
        return title
    if mode == TITLE_MODE_APPEND:
        return f"{stem} - {title}"
    return None
