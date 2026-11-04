"""Cross-check of an API transcript against a local Whisper pass.

``gpt-4o-transcribe`` returns one text per chunk and no times, so the app stamps
its paragraphs by character share within the chunk — an estimate that assumes
an even speaking rate and bends around any speech the answer lacks. A local
faster-whisper pass of the same audio returns timed segments, and the two texts
can be aligned word by word (``difflib.SequenceMatcher`` over normalised words).
From that alignment:

- an API paragraph whose words the local pass also has — a matching run of at
  least ``VERIFY_MIN_ANCHOR_WORDS`` words — takes the local pass's time of its
  first matched words (the recording's own time);
- a paragraph without such a run keeps its estimate, shifted by the drift of
  its nearest re-timed neighbour;
- a stretch the local pass has and the API text lacks, of at least
  ``VERIFY_MIN_GAP_WORDS`` words, comes back as paragraphs of the local text
  with the local times, to be inserted where they belong.

Ported from a one-off script written after a 52-minute presentation lost part
of two chunks (2026-11-03). Pure functions, no I/O; the pipeline renders the
result (``transcribe.build_verified_transcript``).
"""

import re
from difflib import SequenceMatcher
from typing import Any

from config import (
    VERIFY_GAP_PARAGRAPH_SECONDS,
    VERIFY_MIN_ANCHOR_WORDS,
    VERIFY_MIN_GAP_WORDS,
)

# Everything that is not a letter, a digit or an apostrophe separates words.
_NON_WORD = re.compile(r"[^\w']+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SENTENCE_END = (".", "!", "?")
# A matched run earlier in a paragraph than its anchor is trusted only if it
# lies within this long per word before the anchor: nobody speaks slower than
# that for long, so an earlier time is a chance match of a common phrase.
_MAX_SECONDS_PER_WORD = 2.0
# Used for the words before a paragraph's first matched word when the local
# pass gives no rate of its own (no words at all).
_DEFAULT_SECONDS_PER_WORD = 0.4


def normalise(text: str) -> list[str]:
    """Return the words of a text as the alignment compares them.

    Case, punctuation and the kind of apostrophe are ignored; letters of any
    script, digits and apostrophes inside a word are kept.

    Args:
        text: Any text.

    Returns:
        The words, lower-cased, in order.
    """
    folded = text.translate(_APOSTROPHES).casefold().replace("_", " ")
    words = (word.strip("'") for word in _NON_WORD.sub(" ", folded).split())
    return [word for word in words if word]


def _flatten(texts: list[str]) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Turn a list of texts into one word list that remembers where each word came from.

    Args:
        texts: The paragraphs or segments.

    Returns:
        The normalised words; for each word the index of its text; and for each
        word its ``(text index, position)`` among that text's whitespace tokens,
        so the original wording can be put back.
    """
    words: list[str] = []
    owners: list[int] = []
    tokens: list[tuple[int, int]] = []
    for index, text in enumerate(texts):
        for position, token in enumerate(text.split()):
            for word in normalise(token):
                words.append(word)
                owners.append(index)
                tokens.append((index, position))
    return words, owners, tokens


def cross_check(
    paragraphs: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Re-time API paragraphs by a local pass and find what the API text lacks.

    Args:
        paragraphs: The API transcript's paragraphs with their estimated
            ``start`` (seconds) and ``text``; other keys are carried through.
        segments: The local pass's timed segments with ``start``, ``end``
            (seconds) and ``text``.

    Returns:
        ``paragraphs``: the input paragraphs with a new ``start`` and
        ``exact`` (True when the time is the local pass's own, False when it is
        the estimate shifted like a neighbour's); ``gaps``: paragraphs of local
        text the API lacks, each with ``start``, ``text`` and ``words``; and
        ``stats``: ``paragraphs``, ``anchored`` (re-timed), ``shifted``,
        ``clamped`` (moved to keep the order), ``gaps`` (stretches),
        ``added_paragraphs``, ``added_words``, ``api_words``, ``local_words``,
        ``matched_share`` (of the API words) and ``api_only_words``.
    """
    api_words, api_owner, _ = _flatten([str(p["text"]) for p in paragraphs])
    local_words, _, local_token = _flatten([str(s["text"]) for s in segments])
    token_counts = [len(str(s["text"]).split()) for s in segments]
    spoken = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    seconds_per_word = _DEFAULT_SECONDS_PER_WORD
    if local_words and spoken:
        seconds_per_word = spoken / len(local_words)

    def local_time(word_index: int) -> float:
        """Place a local word on the timeline, by its share of its segment."""
        segment_index, position = local_token[word_index]
        segment = segments[segment_index]
        length = max(0.0, segment["end"] - segment["start"])
        share = position / max(token_counts[segment_index], 1)
        return segment["start"] + length * share

    first_word: dict[int, int] = {}
    for index, owner in enumerate(api_owner):
        first_word.setdefault(owner, index)

    matcher = SequenceMatcher(None, api_words, local_words, autojunk=False)
    opcodes = matcher.get_opcodes()

    # 1. The matching runs of each paragraph: (first API word, length, local time).
    runs: dict[int, list[tuple[int, int, float]]] = {}
    matched = 0
    for tag, i1, i2, j1, _j2 in opcodes:
        if tag != "equal":
            continue
        matched += i2 - i1
        k = i1
        while k < i2:
            owner = api_owner[k]
            end = k
            while end < i2 and api_owner[end] == owner:
                end += 1
            runs.setdefault(owner, []).append((k, end - k, local_time(j1 + k - i1)))
            k = end

    # 2. A paragraph with a long enough run takes the local time of its first
    # trustworthy matched word, less the words before it at the local rate.
    count = len(paragraphs)
    times: list[float | None] = [None] * count
    for owner, owner_runs in runs.items():
        anchor_k, anchor_n, anchor_t = max(owner_runs, key=lambda run: run[1])
        if anchor_n < VERIFY_MIN_ANCHOR_WORDS:
            continue
        first_k, first_t = anchor_k, anchor_t
        for k, n, t in owner_runs:
            if k >= anchor_k:
                break
            if n >= 2 and t >= anchor_t - _MAX_SECONDS_PER_WORD * (anchor_k - k):
                first_k, first_t = k, t
                break
        before = first_k - first_word[owner]
        times[owner] = max(0.0, first_t - before * seconds_per_word)
    anchored = [index for index in range(count) if times[index] is not None]

    # 3. The others keep their estimate, moved like their nearest re-timed one.
    for index in range(count):
        if times[index] is not None:
            continue
        estimate = float(paragraphs[index]["start"])
        if not anchored:
            times[index] = estimate
            continue
        nearest = min(anchored, key=lambda other: (abs(other - index), other))
        drift = times[nearest] - float(paragraphs[nearest]["start"])
        times[index] = max(0.0, estimate + drift)
    clamped = 0
    for index in range(1, count):
        if times[index] < times[index - 1]:
            times[index] = times[index - 1]
            clamped += 1

    # 4. What the local pass has and the API text lacks — and the reverse. A
    # stretch is a run of differences; a chance match of a few common words
    # inside it ("of the") does not cut it in two.
    gaps: list[dict[str, Any]] = []
    stretches = 0
    api_only = 0
    for i1, i2, j1, j2 in _regions(opcodes):
        api_side, local_side = i2 - i1, j2 - j1
        if api_side >= VERIFY_MIN_GAP_WORDS and local_side * 3 <= api_side:
            api_only += api_side
        # The local pass has it and the API has nothing, or a few words for a
        # long local stretch (its answer resumed mid-sentence): the local text
        # is added, and the API's few words stay in their paragraph.
        if local_side >= VERIFY_MIN_GAP_WORDS and api_side * 3 <= local_side:
            stretches += 1
            gaps += _gap_paragraphs(segments, local_token, j1, j2, local_time)

    return {
        "paragraphs": [
            {**paragraph, "start": times[index], "exact": index in set(anchored)}
            for index, paragraph in enumerate(paragraphs)
        ],
        "gaps": gaps,
        "stats": {
            "paragraphs": count,
            "anchored": len(anchored),
            "shifted": count - len(anchored),
            "clamped": clamped,
            "gaps": stretches,
            "added_paragraphs": len(gaps),
            "added_words": sum(gap["words"] for gap in gaps),
            "api_words": len(api_words),
            "local_words": len(local_words),
            "matched_share": matched / len(api_words) if api_words else 0.0,
            "api_only_words": api_only,
        },
    }


def _regions(opcodes: list[tuple[str, int, int, int, int]]) -> list[list[int]]:
    """Join the differences between two long matches into one region each.

    Args:
        opcodes: ``SequenceMatcher.get_opcodes()`` over the two word lists.

    Returns:
        ``[i1, i2, j1, j2]`` per region: the API words and the local words
        between two matching runs of at least ``VERIFY_MIN_ANCHOR_WORDS``.
        Shorter matches inside a region are part of it; one at its edge is not.
    """
    regions: list[list[int]] = []
    current: list[int] | None = None
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            if i2 - i1 >= VERIFY_MIN_ANCHOR_WORDS and current is not None:
                regions.append(current)
                current = None
            continue
        if current is None:
            current = [i1, i2, j1, j2]
        else:
            # Opcodes are contiguous, so this takes in any short match held
            # between the region's last difference and this one.
            current[1], current[3] = i2, j2
    if current is not None:
        regions.append(current)
    return regions


def _gap_paragraphs(
    segments: list[dict[str, Any]],
    local_token: list[tuple[int, int]],
    first: int,
    last: int,
    local_time: Any,
) -> list[dict[str, Any]]:
    """Put the original wording of a local-only stretch back, in paragraphs.

    Args:
        segments: The local pass's segments.
        local_token: For each local word, its ``(segment, position)``.
        first: The first local word of the stretch.
        last: One past its last word.
        local_time: Function placing a local word on the timeline.

    Returns:
        Paragraphs with ``start``, ``text`` and ``words``, broken about every
        ``VERIFY_GAP_PARAGRAPH_SECONDS`` at a sentence end — or, since a
        local model can leave sentence ends unpunctuated, where a segment
        starts with a capital letter.
    """
    paragraphs: list[dict[str, Any]] = []
    tokens: list[str] = []
    words = 0
    start = local_time(first)
    seen: tuple[int, int] | None = None
    for index in range(first, last):
        place = local_token[index]
        if place == seen:
            words += 1
            continue
        seen = place
        segment_index, position = place
        token = str(segments[segment_index]["text"]).split()[position]
        moment = local_time(index)
        sentence_start = (tokens and tokens[-1].endswith(_SENTENCE_END)) or (
            position == 0 and token[:1].isupper()
        )
        if tokens and sentence_start and moment - start >= VERIFY_GAP_PARAGRAPH_SECONDS:
            paragraphs.append(
                {"start": start, "text": " ".join(tokens), "words": words}
            )
            tokens, words, start = [], 0, moment
        tokens.append(token)
        words += 1
    if tokens:
        paragraphs.append({"start": start, "text": " ".join(tokens), "words": words})
    return paragraphs


def merge(
    paragraphs: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Put the re-timed paragraphs and the added ones in reading order.

    The API order is kept; an added paragraph goes after the last API paragraph
    whose time is not later than its own (before everything when it is earlier
    than the first).

    Args:
        paragraphs: ``paragraphs`` from :func:`cross_check`.
        gaps: ``gaps`` from :func:`cross_check`.

    Returns:
        The items in order, each with ``kind`` (``"paragraph"`` or ``"gap"``)
        added to its fields.
    """
    pending = sorted(gaps, key=lambda gap: gap["start"])
    items: list[dict[str, Any]] = []
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if index == 0:
            while pending and pending[0]["start"] < paragraph["start"]:
                items.append({**pending.pop(0), "kind": "gap"})
        items.append({**paragraph, "kind": "paragraph"})
        limit = paragraphs[index + 1]["start"] if index + 1 < count else float("inf")
        while pending and paragraph["start"] <= pending[0]["start"] < limit:
            items.append({**pending.pop(0), "kind": "gap"})
    items += [{**gap, "kind": "gap"} for gap in pending]
    return items
