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
from config import (
    DEFAULT_FRAME_DETAIL,
    DEFAULT_VISION_MODEL,
    VISION_OUTPUT_TOKENS_PER_FRAME,
    VISION_PRICE_PER_MTOK,
    VISION_TOKENS_PER_FRAME,
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


def estimate_frame_cost(
    frame_count: int,
    model: str = DEFAULT_VISION_MODEL,
    detail: str = DEFAULT_FRAME_DETAIL,
) -> float | None:
    """Estimate what describing a number of frames would cost, in USD.

    Args:
        frame_count: Number of frames that would be described.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.

    Returns:
        The approximate cost, or ``None`` if no price is known for the model.
    """
    prices = VISION_PRICE_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    input_cost = estimate_frame_tokens(frame_count, detail) * input_price
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
) -> str | None:
    """Describe a single frame.

    Args:
        client: Configured OpenAI client.
        frame_path: Path to the JPEG frame.
        model: Vision model name.
        detail: Image fidelity, ``"low"`` or ``"high"``.

    Returns:
        The description, or ``None`` when the frame holds nothing worth noting.

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

    if not response.choices:
        raise VisualContextError("OpenAI returned no answer for this frame")
    description = (response.choices[0].message.content or "").strip()
    if not description or description.upper().startswith(_SKIP_MARKER):
        return None
    return description


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


def describe_keyframes(
    frames: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    detail: str = DEFAULT_FRAME_DETAIL,
    progress_callback: ProgressCallback | None = None,
    cache: Path | None = None,
    failed: list[float] | None = None,
    collected: list[dict[str, Any]] | None = None,
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
            name = f"frame_{round(frame['time'] * 1000):09d}"
            saved = checkpoints.load(cache, name) if cache else None
            if saved is not None:
                description = saved.get("description")
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
                    description = _describe_frame(
                        client, Path(frame["path"]), model, detail
                    )
                except VisualContextError as exc:
                    # One unreadable frame should not cost the user the whole run.
                    # An OpenAIAccountError is not a VisualContextError, so a bad
                    # key or an empty balance passes straight through.
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
                if cache:
                    checkpoints.save(cache, name, {"description": description})
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
