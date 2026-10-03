"""An AI title for a finished transcript.

After a run, a chat model reads the transcript and names it in a few words, in
the language the speakers use. The title is stored beside the original file
name, never over it: the title mode (see :mod:`config`) only decides how the two
are shown in History and in the names of downloaded files, so changing the mode
applies to earlier rows as well and never pays for a title twice. Like the rest
of the pipeline this module is UI-agnostic.

The smallest model does not always keep to the transcript's language: an English
meeting got a Chinese title. So the script the transcript is written in is
counted here, named in a reminder after the transcript, and a title in a script
the transcript never uses is sent back once to be written again.
"""

import unicodedata
from collections import Counter
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
    TITLE_TIMEOUT_SECONDS,
    TITLE_TOKENS_PER_HOUR,
)
from exceptions import AppError, TitleError

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
# The first answer, and one more after a title in a script the transcript
# does not use. A third try would mostly pay for the same mistake again.
_ATTEMPTS = 2
# Japanese writes kanji, hiragana and katakana together: one script here.
_SCRIPT_GROUPS = {"HIRAGANA": "CJK", "KATAKANA": "CJK"}
# The share of a transcript's letters a title's main script needs. A Serbian
# meeting can alternate scripts by chunk, and a title in either passes while it
# is a tenth of the text; a stray line (Whisper hallucinates Chinese subtitle
# credits in silence) does not.
_MAIN_SCRIPT_MIN_SHARE = 0.1
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


def letter_scripts(text: str) -> Counter[str]:
    """Count the letters of a text by the script they belong to.

    The text is NFKC-normalised first, so letter-like symbols count as the
    letters they stand for (``ª`` as a, ``ＡＩ`` as AI, ``µ`` as Greek mu), and
    modifier letters (``ʼ``, the Japanese ``ー``) are not counted at all.

    Args:
        text: Any text.

    Returns:
        Letters per script, named by the first word of their Unicode name:
        ``LATIN`` (with č, ć, š…), ``CYRILLIC``, ``CJK`` (with Japanese
        hiragana and katakana), ``HANGUL``…
    """
    scripts: Counter[str] = Counter()
    for char in unicodedata.normalize("NFKC", text):
        if not char.isalpha() or unicodedata.category(char) == "Lm":
            continue
        if name := unicodedata.name(char, ""):
            script = name.split(" ", 1)[0]
            scripts[_SCRIPT_GROUPS.get(script, script)] += 1
    return scripts


def _script_name(script: str) -> str:
    """Return a script's name as it reads in an instruction (``Latin``, ``CJK``)."""
    return script if script == "CJK" else script.title()


def foreign_scripts(title: str, transcript_scripts: Counter[str]) -> list[str]:
    """Return the scripts of a title that do not fit its transcript.

    The title's main script must be a main one of the transcript (at least
    ``_MAIN_SCRIPT_MIN_SHARE`` of its letters): a single hallucinated Chinese
    line in an English transcript must not let a Chinese title through. Its
    other letters only need to occur in the transcript at all, so a Cyrillic
    title may keep "Azure" when the transcript has it in Latin too.

    Args:
        title: A cleaned title.
        transcript_scripts: :func:`letter_scripts` of the transcript.

    Returns:
        The scripts that do not fit, the title's main one first (empty when
        the title is fine, or when either has no letters to compare).
    """
    used = [script for script, _ in letter_scripts(title).most_common()]
    if not transcript_scripts or not used:
        return []
    foreign = [script for script in used if script not in transcript_scripts]
    main = used[0]
    share = transcript_scripts[main] / sum(transcript_scripts.values())
    if main not in foreign and share < _MAIN_SCRIPT_MIN_SHARE:
        foreign.insert(0, main)
    return foreign


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


def _request(transcript: str, main_script: str | None) -> str:
    """Return the user message: the transcript, then a reminder of its script.

    The reminder comes after the transcript because a small model follows the
    last thing it read more reliably than the instructions before a long text.

    Args:
        transcript: The whole transcript.
        main_script: The script most of its letters are in, if it has letters.

    Returns:
        The message text.
    """
    reminder = "End of transcript. Reply with its title in the speakers' language"
    if main_script:
        script = _script_name(main_script)
        reminder += f", written in {script} script like the transcript"
    return f"{_excerpt(transcript)}\n\n---\n{reminder}."


def _ask(client: Any, model: str, messages: list[dict[str, str]]) -> Any:
    """Send one title request.

    Args:
        client: The OpenAI client.
        model: Chat model name.
        messages: The conversation so far.

    Returns:
        The chat completion.

    Raises:
        OpenAIAccountError: If the key or the account is refused.
        TitleError: If the request fails for any other reason.
    """
    try:
        return client.chat.completions.create(
            model=model,
            max_completion_tokens=_MAX_TITLE_TOKENS,
            messages=messages,
            timeout=TITLE_TIMEOUT_SECONDS,
        )
    except OpenAIError as exc:
        raise openai_api.translate_error(exc, TitleError) from exc


def make_title(
    transcript: str, api_key: str, model: str
) -> tuple[str, list[dict[str, Any]]]:
    """Ask a chat model for the title of a transcript.

    A title in a script the transcript never uses (a Chinese title for an
    English meeting) is sent back once with a correction; if the second answer
    is no better, there is no title.

    Args:
        transcript: The finished transcript.
        api_key: OpenAI API key.
        model: Chat model name (see ``TITLE_MODELS``).

    Returns:
        The cleaned title, and the usage records of the requests (two when
        the first answer was sent back).

    Raises:
        OpenAIAccountError: If the key or the account is refused on the first
            request (nothing was paid for yet).
        TitleError: If a request fails for any other reason, or no usable title
            came back. Its ``usage_records`` hold the requests that were paid
            for all the same.
    """
    if not transcript.strip():
        raise TitleError("the transcript is empty")
    scripts = letter_scripts(transcript)
    main_script = scripts.most_common(1)[0][0] if scripts else None
    # Used only after a foreign script was found, which needs letters to compare.
    right = _script_name(main_script) if main_script else ""
    messages = [
        {"role": "developer", "content": _PROMPT},
        {"role": "user", "content": _request(transcript, main_script)},
    ]
    spent: list[dict[str, Any]] = []
    wrong = ""
    client = openai_api.make_client(api_key)
    try:
        for _ in range(_ATTEMPTS):
            try:
                response = _ask(client, model, messages)
            except AppError as exc:
                if not spent:
                    raise
                # The first answer was paid for; it must not go uncounted.
                raise _failed(str(exc), spent) from exc
            spent.append(usage.title_record(response, model))
            answer = response.choices[0].message.content if response.choices else None
            title = clean_title(answer or "")
            if title is None:
                raise _failed("OpenAI returned no title", spent)
            foreign = foreign_scripts(title, scripts)
            if not foreign:
                return title, spent
            wrong = _script_name(foreign[0])
            messages += [
                {"role": "assistant", "content": answer or ""},
                {
                    "role": "user",
                    "content": (
                        f"That title is in {wrong} script, but the transcript is in "
                        f"{right}. Write it again in the speakers' language, in "
                        f"{right} script. Reply with the title only."
                    ),
                },
            ]
    finally:
        client.close()
    raise _failed(
        f"the model wrote it in {wrong} script twice, but the transcript is in {right}",
        spent,
    )


def _failed(reason: str, spent: list[dict[str, Any]]) -> TitleError:
    """Return a title error that carries what its requests cost.

    Args:
        reason: Why there is no title, phrased for the user.
        spent: Usage records of the requests paid for.

    Returns:
        The error, with ``usage_records`` set.
    """
    error = TitleError(reason)
    error.usage_records = list(spent)
    return error


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
