"""Describing video key frames with a vision model.

Audio-only transcription misses everything that was shown rather than said:
slide headings, figures, diagrams, code on a shared screen. This module turns
the frames selected by :mod:`frames` into short factual notes that
:mod:`transcribe` interleaves into the transcript by timestamp.

Frames the model judges uninformative are dropped, so a recording of a talking
head adds nothing to the transcript even if a few frames were extracted. Each
description is checkpointed as it arrives, so a failed or interrupted run does
not pay for the same frames twice. Like the rest of the pipeline this module is
UI-agnostic: progress is reported through an optional callback.
"""

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

import checkpoints
import openai_api
import usage
from config import (
    CHAT_PRICE_PER_MTOK,
    DEFAULT_FRAME_DETAIL,
    DEFAULT_VISION_MODEL,
    VISION_OUTPUT_TOKENS_PER_FRAME,
    VISION_TOKENS_FIXED,
    VISION_TOKENS_PER_FRAME,
    VISION_TOKENS_PER_MEGAPIXEL,
)
from exceptions import VisualContextError
from logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

# Kept deliberately terse: the instruction is re-sent with every frame, so its
# length is charged once per image.
_CAPTION_PROMPT = (
    "You are captioning a still frame from a screen recording. In one short "
    "sentence, state what is displayed — slide titles, headings, figures, "
    "diagrams or code. Quote any text you can read verbatim. If the frame "
    "shows nothing informative (a face, a blank screen, a plain desktop), "
    "reply with exactly NONE."
)
_SKIP_MARKER = "NONE"
_MAX_CAPTION_TOKENS = 120
# Give up after this many frames fail in a row: the cause is then the network or
# the service, not one bad frame, and every further attempt would wait it out.
_MAX_CONSECUTIVE_FAILURES = 3
# Bump when the prompt changes, so descriptions made with the old prompt are not
# reused.
_CACHE_VERSION = "v1"


def estimate_frame_tokens(frame_count: int, detail: str = DEFAULT_FRAME_DETAIL) -> int:
    """Estimate the image tokens a run will spend, for the pre-run cost hint.

    Args:
        frame_count: Number of frames that would be described.
        detail: Image fidelity, ``"low"`` or ``"high"``.

    Returns:
        Approximate total image tokens.
    """
    per_frame = VISION_TOKENS_PER_FRAME.get(detail, VISION_TOKENS_PER_FRAME["low"])
    return frame_count * per_frame


def frame_tokens(width: int, height: int) -> int:
    """Estimate the input tokens of one frame of a given size.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tokens per request (image plus prompt), from measured values.
    """
    megapixels = width * height / 1_000_000
    return round(VISION_TOKENS_FIXED + VISION_TOKENS_PER_MEGAPIXEL * megapixels)


def estimate_frame_cost(
    frame_count: int,
    model: str = DEFAULT_VISION_MODEL,
    detail: str = DEFAULT_FRAME_DETAIL,
    tokens_per_frame: int | None = None,
) -> float | None:
    """Estimate what describing a number of frames would cost, in USD.

    Args:
        frame_count: Number of frames that would be described.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.
        tokens_per_frame: Input tokens per frame when the frame size is known
            (see :func:`frame_tokens`); otherwise a Full HD figure is assumed.

    Returns:
        The approximate cost, or ``None`` if no price is known for the model.
    """
    prices = CHAT_PRICE_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    if tokens_per_frame is None:
        input_tokens = estimate_frame_tokens(frame_count, detail)
    else:
        input_tokens = frame_count * tokens_per_frame
    input_cost = input_tokens * input_price
    output_cost = frame_count * VISION_OUTPUT_TOKENS_PER_FRAME * output_price
    return (input_cost + output_cost) / 1_000_000


def _encode_frame(frame_path: Path) -> str:
    """Read an image and return it as a base64 data URL.

    Args:
        frame_path: Path to the JPEG frame.

    Returns:
        A ``data:image/jpeg;base64,...`` URL.

    Raises:
        VisualContextError: If the frame cannot be read.
    """
    try:
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise VisualContextError(f"Cannot read frame {frame_path}: {exc}") from exc
    return f"data:image/jpeg;base64,{encoded}"


def _describe_frame(
    client: OpenAI, frame_path: Path, model: str, detail: str
) -> tuple[str | None, dict[str, Any]]:
    """Describe a single frame.

    Args:
        client: Configured OpenAI client.
        frame_path: Path to the JPEG frame.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.

    Returns:
        The description (``None`` when the frame holds nothing worth noting),
        and the usage record of the request — paid for either way.

    Raises:
        OpenAIAccountError: If the key or the account is refused.
        VisualContextError: If this request fails for any other reason.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=_MAX_CAPTION_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _CAPTION_PROMPT},
                        {
                            "type": "image_url",
                            # `detail` belongs inside image_url for this API.
                            "image_url": {
                                "url": _encode_frame(frame_path),
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
        )
    except OpenAIError as exc:
        raise openai_api.translate_error(exc, VisualContextError) from exc

    spent = usage.vision_record(response, model)
    if not response.choices:
        # Paid for all the same; the caller counts it with the failed frame.
        error = VisualContextError("OpenAI returned no answer for this frame")
        error.usage_record = spent
        raise error
    description = (response.choices[0].message.content or "").strip()
    if not description or description.upper().startswith(_SKIP_MARKER):
        return None, spent
    return description, spent


def _report(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    """Invoke the progress callback if one was provided.

    Args:
        progress_callback: Optional callback receiving a status payload.
        **payload: Status fields (e.g. ``status``, ``message``, ``progress``).
    """
    if progress_callback:
        progress_callback(payload)


def cache_dir(video_digest: str, model: str, detail: str) -> Path:
    """Return where descriptions of one video's frames are kept.

    A frame at a given time is the same picture whatever interval selected it,
    so descriptions are keyed by frame time and reused across settings.

    Args:
        video_digest: :func:`checkpoints.file_digest` of the video.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.

    Returns:
        The cache directory (created on first save).
    """
    return checkpoints.run_dir(video_digest, "frames", model, detail, _CACHE_VERSION)


def frame_name(time: float) -> str:
    """Return the cache name of the frame at ``time`` seconds."""
    return f"frame_{round(time * 1000):09d}"


def mark_billed(cache: Path, names: list[str]) -> None:
    """Mark cached descriptions as counted by a saved history row.

    A run that saves with some screenshots missing keeps the cache, so a rerun
    pays only for the missing ones — and must not count the reused ones again.
    Only the frames this row used are marked: marking the whole cache also
    marked descriptions paid by an unsaved attempt with another interval, which
    then counted in no row at all.

    Args:
        cache: Directory from :func:`cache_dir`.
        names: :func:`frame_name` of each frame the row used.
    """
    for name in names:
        saved = checkpoints.load(cache, name)
        if isinstance(saved, dict) and not saved.get("billed"):
            checkpoints.save(cache, name, {**saved, "billed": True})


def unbilled_usage(cache: Path, model: str, used: list[str]) -> list[dict[str, Any]]:
    """Return what cached descriptions outside this run cost, if no row has yet.

    Called before a complete run discards the cache: descriptions paid by an
    earlier, unsaved attempt with other settings would otherwise vanish from
    every total.

    Args:
        cache: Directory from :func:`cache_dir`.
        model: Vision model name, for descriptions saved without usage.
        used: :func:`frame_name` of each frame this run used (already counted).

    Returns:
        One usage record per such description.
    """
    if not cache.is_dir():
        return []
    skip = set(used)
    records = []
    for path in sorted(cache.glob("frame_*.json")):
        saved = checkpoints.load(cache, path.stem)
        if path.stem in skip or not isinstance(saved, dict) or saved.get("billed"):
            continue
        records.append(saved.get("usage") or usage.vision_record(None, model))
    return records


def describe_keyframes(
    frames: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    detail: str = DEFAULT_FRAME_DETAIL,
    progress_callback: ProgressCallback | None = None,
    cache: Path | None = None,
    failed: list[float] | None = None,
    collected: list[dict[str, Any]] | None = None,
    spent: list[dict[str, Any]] | None = None,
    picture: Callable[[dict[str, Any]], Path] | None = None,
) -> list[dict[str, Any]]:
    """Describe every extracted key frame, keeping only the informative ones.

    Frames are sent one per request. Batching them would save only the shared
    prompt — image tokens dominate and are billed per image either way — while
    risking the model conflating frames and losing every result on one failure.

    Args:
        frames: Frames with ``time`` (seconds) and ``path``, ordered by time.
        api_key: OpenAI API key.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.
        progress_callback: Optional callback receiving status payloads.
        cache: Directory from :func:`cache_dir`; descriptions found there are
            reused and new ones saved. ``None`` disables caching.
        failed: When given, the times of frames that could not be described
            are appended to it, so the caller can say how many were lost.
        collected: When given, each note is appended to it as it is made, so
            the notes already paid for survive an early stop.
        spent: When given, the usage record of every description used —
            including ones reused from the cache, paid by an earlier attempt —
            is appended to it.
        picture: Returns the image to send for a frame (the scan's
            :func:`frames.pictures`); called only for frames not already in the
            cache. Defaults to the frame's ``path``.

    Returns:
        Notes with ``time`` and ``description``, ordered by time. Frames the
        model found uninformative are omitted, so this may be shorter than
        ``frames`` — or empty.

    Raises:
        OpenAIAccountError: At the first frame refused for account reasons
            (bad key, no credit) — every other frame would be too.
        VisualContextError: If every frame fails, or several in a row do.
    """
    if not frames:
        return []

    client: OpenAI | None = None
    notes: list[dict[str, Any]] = collected if collected is not None else []
    failures = 0
    consecutive = 0
    total = len(frames)

    try:
        for index, frame in enumerate(frames):
            name = frame_name(frame["time"])
            saved = checkpoints.load(cache, name) if cache else None
            if saved is not None:
                description = saved.get("description")
                # Reused descriptions count once: in the first saved run that
                # used them. One without usage predates cost recording, so it
                # counts as unknown rather than free.
                if spent is not None and not saved.get("billed"):
                    spent.append(saved.get("usage") or usage.vision_record(None, model))
                # A frame described earlier is a success: it ends a run of
                # failures, or a resumed step would stop at old bad frames.
                consecutive = 0
            else:
                _report(
                    progress_callback,
                    status="progress",
                    message=f"Reading screen {index + 1} of {total}…",
                    progress=index / total,
                )
                client = client or openai_api.make_client(api_key)
                try:
                    image = picture(frame) if picture else Path(frame["path"])
                    description, record = _describe_frame(client, image, model, detail)
                except VisualContextError as exc:
                    # One unreadable frame should not cost the user the whole run.
                    # An OpenAIAccountError is not a VisualContextError, so a bad
                    # key or an empty balance passes straight through.
                    if spent is not None and exc.usage_record:
                        spent.append(exc.usage_record)
                    failures += 1
                    consecutive += 1
                    if failed is not None:
                        failed.append(frame["time"])
                    logger.warning("Skipping frame at %.1fs: %s", frame["time"], exc)
                    if consecutive >= _MAX_CONSECUTIVE_FAILURES:
                        raise VisualContextError(
                            f"{consecutive} frames in a row failed; last error: {exc}"
                        ) from exc
                    continue
                consecutive = 0
                if spent is not None:
                    spent.append(record)
                if cache:
                    checkpoints.save(
                        cache, name, {"description": description, "usage": record}
                    )
            if description:
                notes.append({"time": frame["time"], "description": description})
    finally:
        if client is not None:
            client.close()

    if failures == total:
        raise VisualContextError(
            f"Could not describe any of the {total} extracted frames"
        )

    logger.info("Kept %d on-screen notes from %d frames", len(notes), total)
    return list(notes)
