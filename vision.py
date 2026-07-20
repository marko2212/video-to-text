"""Describing video key frames with a vision model.

Audio-only transcription misses everything that was shown rather than said:
slide headings, figures, diagrams, code on a shared screen. This module turns
the frames selected by :mod:`frames` into short factual notes that
:mod:`transcribe` interleaves into the transcript by timestamp.

Frames the model judges uninformative are dropped, so a recording of a talking
head adds nothing to the transcript even if a few frames were extracted. Like
the rest of the pipeline this module is UI-agnostic: progress is reported
through an optional callback.
"""

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from config import (
    DEFAULT_FRAME_DETAIL,
    DEFAULT_VISION_MODEL,
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
    price = VISION_PRICE_PER_MTOK.get(model)
    if price is None:
        return None
    return estimate_frame_tokens(frame_count, detail) * price / 1_000_000


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
        VisualContextError: If the API request fails.
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
        raise VisualContextError(f"Vision request failed: {exc}") from exc

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


def describe_keyframes(
    frames: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    detail: str = DEFAULT_FRAME_DETAIL,
    progress_callback: ProgressCallback | None = None,
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

    Returns:
        Notes with ``time`` and ``description``, ordered by time. Frames the
        model found uninformative are omitted, so this may be shorter than
        ``frames`` — or empty.

    Raises:
        VisualContextError: If every frame fails.
    """
    if not frames:
        return []

    client = OpenAI(api_key=api_key)
    notes: list[dict[str, Any]] = []
    failures = 0
    total = len(frames)

    for index, frame in enumerate(frames, start=1):
        _report(
            progress_callback,
            status="progress",
            message=f"Reading screen {index}/{total}",
            progress=index / total,
        )
        try:
            description = _describe_frame(client, Path(frame["path"]), model, detail)
        except VisualContextError as exc:
            # One unreadable frame should not cost the user the whole run.
            failures += 1
            logger.warning("Skipping frame at %.1fs: %s", frame["time"], exc)
            continue
        if description:
            notes.append({"time": frame["time"], "description": description})

    if failures == total:
        raise VisualContextError(
            f"Could not describe any of the {total} extracted frames"
        )

    logger.info("Kept %d on-screen notes from %d frames", len(notes), total)
    return notes
