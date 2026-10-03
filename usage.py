"""What each paid OpenAI request actually used, and what it cost.

Every transcription chunk, described screenshot and title leaves one record:
the model, the tokens the API reported (or the audio seconds, for models billed
per minute), and the cost at the prices in :mod:`config`. Records travel with
checkpointed results, so a run that resumes still counts what its earlier
attempt paid for. Pure functions, no I/O.
"""

from typing import Any

from config import (
    CHAT_PRICE_PER_MTOK,
    TRANSCRIPTION_FALLBACK_PRICE_PER_MINUTE,
    TRANSCRIPTION_PRICE_PER_MINUTE,
    TRANSCRIPTION_PRICE_PER_MTOK,
)

_MILLION = 1_000_000


def _field(container: Any, name: str) -> Any:
    """Read a field from a dict or an SDK object (``None`` when absent)."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _int(value: Any) -> int | None:
    """Return the value if it is a whole number of tokens, else ``None``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def transcription_record(result: Any, model: str, seconds: float) -> dict[str, Any]:
    """Describe what one transcription request used and cost.

    Args:
        result: The API response (its ``usage`` is read when present).
        model: Transcription model name.
        seconds: Length of the audio that was sent.

    Returns:
        A JSON-serialisable record: ``kind``, ``model``, ``seconds``,
        ``input_tokens``, ``output_tokens``, ``cost_usd`` and ``estimated``
        (True when the cost could not be computed from reported usage).
    """
    usage = _field(result, "usage")
    input_tokens = _int(_field(usage, "input_tokens"))
    output_tokens = _int(_field(usage, "output_tokens"))
    billed_seconds = _field(usage, "seconds")
    if isinstance(billed_seconds, int | float) and not isinstance(billed_seconds, bool):
        seconds = float(billed_seconds)

    prices = TRANSCRIPTION_PRICE_PER_MTOK.get(model)
    if prices and input_tokens is not None and output_tokens is not None:
        cost = (input_tokens * prices[0] + output_tokens * prices[1]) / _MILLION
        estimated = False
    else:
        per_minute = TRANSCRIPTION_PRICE_PER_MINUTE.get(model)
        # A per-minute model is billed exactly this way; a token-billed model
        # without usage, or an unknown model, only approximately.
        estimated = per_minute is None or model in TRANSCRIPTION_PRICE_PER_MTOK
        cost = seconds / 60 * (per_minute or TRANSCRIPTION_FALLBACK_PRICE_PER_MINUTE)
    return {
        "kind": "transcription",
        "model": model,
        "seconds": round(seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "estimated": estimated,
    }


def _chat_record(response: Any, model: str, kind: str) -> dict[str, Any]:
    """Describe what one chat completion used and cost.

    Args:
        response: The chat completion (its ``usage`` is read when present).
        model: Chat model name.
        kind: What the request was for, e.g. ``"vision"`` or ``"title"``.

    Returns:
        A record like :func:`transcription_record`'s, without ``seconds``;
        ``cost_usd`` is ``None`` when the model or its usage is unknown.
    """
    usage = _field(response, "usage")
    input_tokens = _int(_field(usage, "prompt_tokens"))
    output_tokens = _int(_field(usage, "completion_tokens"))
    prices = CHAT_PRICE_PER_MTOK.get(model)
    cost = None
    if prices and input_tokens is not None and output_tokens is not None:
        cost = (input_tokens * prices[0] + output_tokens * prices[1]) / _MILLION
    return {
        "kind": kind,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "estimated": cost is None,
    }


def vision_record(response: Any, model: str) -> dict[str, Any]:
    """Describe what one screenshot description used and cost.

    Args:
        response: The chat completion (its ``usage`` is read when present).
        model: Vision model name.

    Returns:
        See :func:`_chat_record`.
    """
    return _chat_record(response, model, "vision")


def title_record(response: Any, model: str) -> dict[str, Any]:
    """Describe what one title request used and cost.

    Args:
        response: The chat completion (its ``usage`` is read when present).
        model: Title model name.

    Returns:
        See :func:`_chat_record`.
    """
    return _chat_record(response, model, "title")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Add up a list of records.

    Args:
        records: Records from :func:`transcription_record` or
            :func:`vision_record`.

    Returns:
        ``requests``, ``seconds``, ``input_tokens``, ``output_tokens``,
        ``cost_usd`` and ``estimated`` (True if any part was estimated or
        unknown; an unknown cost counts as zero in the total).
    """
    return {
        "requests": len(records),
        "seconds": round(sum(r.get("seconds") or 0 for r in records), 3),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in records),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in records),
        "cost_usd": sum(r.get("cost_usd") or 0 for r in records),
        "estimated": any(
            r.get("estimated") or r.get("cost_usd") is None for r in records
        ),
    }


def add_to_cost(
    cost: dict[str, Any], part: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Add later requests to a saved run's cost, e.g. a title asked for in History.

    Args:
        cost: A run's cost as saved with its history row (``usage_json``): part
            summaries such as ``title``, the total ``cost_usd`` and ``estimated``.
        part: The part the requests belong to.
        records: Their usage records.

    Returns:
        A new cost with that part's summary and the total grown by the records.
    """
    if not records:
        return cost
    added = summarize(records)
    before = cost.get(part) or {}
    grown = {
        key: before.get(key, 0) + added[key]
        for key in ("requests", "seconds", "input_tokens", "output_tokens", "cost_usd")
    }
    grown["estimated"] = bool(before.get("estimated")) or added["estimated"]
    return {
        **cost,
        part: grown,
        "cost_usd": (cost.get("cost_usd") or 0) + added["cost_usd"],
        "estimated": bool(cost.get("estimated")) or added["estimated"],
    }


def format_usd(cost: float, estimated: bool = False) -> str:
    """Render a cost readably at every scale, from fractions of a cent up.

    Args:
        cost: Amount in USD.
        estimated: Prefix ``≈`` when the figure is an estimate.

    Returns:
        E.g. ``$0``, ``<$0.0001``, ``$0.0042``, ``$0.07`` or ``≈ $1.23``.
    """
    if cost <= 0:
        text = "$0"
    elif cost < 0.0001:
        text = "<$0.0001"
    else:
        text = f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"
    return f"≈ {text}" if estimated else text
