"""Key-frame extraction for on-screen (visual) context.

Sampling a video at a fixed interval is wasteful: a talking head produces
hundreds of near-identical images, and every one of them costs vision tokens.
Instead this module asks ffmpeg's ``select`` filter for the frames where the
picture actually changed, drops perceptual near-duplicates, and applies interval
and count guardrails so the cost of a long screen-share stays predictable.

Like :mod:`audio`, this module is UI-agnostic — it never imports Streamlit.
"""

import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import ffmpeg
from PIL import Image

from config import (
    FRAME_DUPLICATE_DISTANCE,
    FRAME_MAX_COUNT,
    FRAME_MAX_INTERVAL_SECONDS,
    FRAME_MIN_INTERVAL_SECONDS,
    FRAME_QUALITY,
    HASH_SIZE,
    SCENE_THRESHOLD,
    get_settings,
)
from exceptions import VisualContextError
from logger import get_logger

logger = get_logger(__name__)

# `showinfo` prints one line per frame it passes through; the frame's position on
# the timeline is the `pts_time:` field. The pattern is anchored on the filter's
# own log prefix because ffmpeg emits other lines containing "pts_time" — matching
# those would inject phantom timestamps and shift every frame out of alignment.
_SHOWINFO_PTS = re.compile(r"Parsed_showinfo.*?\bpts_time:(-?\d+(?:\.\d+)?)")
# Zero-padded so lexical sorting of the written files matches numeric order.
_FRAME_PATTERN = "frame_%05d.jpg"
# A clip shorter than the chosen interval would be represented by a single
# frame, so the interval tightens just enough to get this many samples out of it.
# Deliberately small: beyond avoiding that, the chosen cadence is left alone.
_MIN_SAMPLES_PER_VIDEO = 2
_MIN_SAMPLE_SECONDS = 5.0


def _ffmpeg_error_detail(exc: ffmpeg.Error) -> str:
    """Return the stderr text carried by an ffmpeg error, if any.

    Args:
        exc: The exception raised by ``ffmpeg.run``.

    Returns:
        The decoded stderr output, falling back to the exception text.
    """
    return exc.stderr.decode(errors="replace") if exc.stderr else str(exc)


def _parse_frame_times(stderr: str) -> list[float]:
    """Pull the timeline position of every emitted frame out of ffmpeg's log.

    Args:
        stderr: The captured ffmpeg stderr containing ``showinfo`` lines.

    Returns:
        Frame times in seconds, in the order the frames were written.
    """
    return [float(match) for match in _SHOWINFO_PTS.findall(stderr)]


@lru_cache
def _rate_control_kwarg() -> dict[str, str]:
    """Return the output flag that stops ffmpeg padding back to a constant rate.

    ``-fps_mode`` replaced ``-vsync`` in ffmpeg 5.1. Build strings are not
    reliably parseable (a git build reports only a date), so the flag is probed
    for directly rather than inferred from a version number.

    Returns:
        Either ``{"fps_mode": "vfr"}`` or the pre-5.1 ``{"vsync": "vfr"}``.
    """
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        # ffmpeg missing entirely; extraction will fail with a clearer message.
        return {"fps_mode": "vfr"}
    if "-fps_mode" in (probe.stdout + probe.stderr):
        return {"fps_mode": "vfr"}
    logger.info("ffmpeg predates -fps_mode; falling back to -vsync")
    return {"vsync": "vfr"}


def _run_frame_select(
    video_path: Path, output_dir: Path, threshold: float, max_interval: float
) -> str:
    """Select the frames worth looking at and return ffmpeg's stderr.

    Scene detection alone is not enough: ffmpeg's score is tuned for natural
    footage, and a measured full-screen slide change scored only 0.077. So the
    same pass also takes a frame whenever nothing has been selected for
    ``max_interval`` seconds, which is what actually guarantees coverage.
    Doing both in one pass keeps every timestamp on the same timeline and avoids
    one ffmpeg invocation per sampled second.

    Variable frame rate output is essential: without it ffmpeg pads the result
    back to a constant frame rate by duplicating frames, which silently undoes
    the whole point of selecting only some frames — and leaves the images
    misaligned with the timestamps rather than obviously broken.

    Args:
        video_path: Source video file.
        output_dir: Directory the JPEG frames are written into.
        threshold: Scene-change threshold between 0 and 1.
        max_interval: Longest stretch in seconds allowed without a frame.

    Returns:
        The captured stderr, containing one ``showinfo`` line per frame.

    Raises:
        VisualContextError: If ffmpeg cannot read the video.
    """
    # `eq(n,0)` force-includes the very first frame: its scene score is
    # undefined, so `gt(scene,...)` alone always skips it — and selecting it
    # also seeds `prev_selected_t` for the interval term. In ffmpeg expressions
    # `+` is addition and any non-zero value is true, so the sum acts as an OR.
    # The expression is passed through as-is — ffmpeg-python escapes the commas.
    select_expression = (
        f"eq(n,0)+gt(scene,{threshold})+gte(t-prev_selected_t,{max_interval})"
    )

    stream = ffmpeg.input(str(video_path))
    stream = stream.filter("select", select_expression).filter("showinfo")
    stream = ffmpeg.output(
        stream,
        str(output_dir / _FRAME_PATTERN),
        **_rate_control_kwarg(),
        **{"qscale:v": FRAME_QUALITY},
    )

    try:
        # showinfo logs at INFO level, so the log level must not be raised — and
        # on success the data arrives in the returned stderr, not an exception.
        _, stderr = ffmpeg.run(
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        logger.error("Key-frame extraction failed for %s: %s", video_path, detail)
        raise VisualContextError(f"Failed to extract key frames: {detail}") from exc

    return stderr.decode(errors="replace")


def video_duration(video_path: Path) -> float:
    """Return the duration of a video in seconds.

    Args:
        video_path: Source video file.

    Returns:
        Duration in seconds, or 0.0 when it cannot be determined.
    """
    try:
        metadata = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as exc:
        logger.debug("ffprobe failed for %s: %s", video_path, _ffmpeg_error_detail(exc))
        return 0.0
    try:
        return float(metadata["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def sample_interval(duration: float, max_interval: float) -> float:
    """Return how often to take a frame regardless of scene changes.

    Args:
        duration: Video duration in seconds; 0 when unknown.
        max_interval: The requested upper bound between samples.

    Returns:
        The requested interval, tightened only when the video is too short for
        it to fire more than once.
    """
    if duration <= 0:
        return max_interval
    return min(
        max_interval, max(duration / _MIN_SAMPLES_PER_VIDEO, _MIN_SAMPLE_SECONDS)
    )


def dhash(image_path: Path, size: int = HASH_SIZE) -> int:
    """Compute a difference hash for an image.

    Each bit records whether a pixel is brighter than the one to its right, so
    the hash tracks structure (where the edges are) and ignores overall
    brightness — which is exactly the jitter that makes two shots of the same
    slide look different byte-for-byte.

    Args:
        image_path: Path to the image file.
        size: Hash edge length; the default 8 yields a 64-bit hash.

    Returns:
        The hash as an integer.

    Raises:
        VisualContextError: If the image cannot be read.
    """
    try:
        with Image.open(image_path) as image:
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                # convert("L") drops alpha instead of compositing it, which turns
                # transparent pixels into mid-grey noise. Flatten onto white first.
                canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(canvas, image.convert("RGBA"))
            # One extra column so each row yields `size` left-to-right comparisons.
            # The resampling filter is passed explicitly: Pillow's default is
            # unspecified, and NEAREST vs LANCZOS shifts a hash by ~10 bits.
            grayscale = image.convert("L").resize(
                (size + 1, size), Image.Resampling.LANCZOS
            )
            pixels = list(grayscale.getdata())
    except OSError as exc:
        raise VisualContextError(f"Cannot read frame {image_path}: {exc}") from exc

    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            brighter = pixels[offset + column] > pixels[offset + column + 1]
            bits = (bits << 1) | int(brighter)
    return bits


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two hashes.

    Args:
        left: First hash.
        right: Second hash.

    Returns:
        The bit-level distance; 0 means the hashes are identical.
    """
    return (left ^ right).bit_count()


def apply_min_interval(
    frames: list[dict[str, Any]], min_interval: float
) -> list[dict[str, Any]]:
    """Drop frames that follow the previous kept frame too closely.

    A hard cut or a camera pan can trip scene detection several times within a
    second; keeping one of those is enough.

    Args:
        frames: Frames with a ``time`` key, ordered by time.
        min_interval: Minimum spacing in seconds between kept frames.

    Returns:
        The thinned list.
    """
    kept: list[dict[str, Any]] = []
    for frame in frames:
        if kept and frame["time"] - kept[-1]["time"] < min_interval:
            continue
        kept.append(frame)
    return kept


def drop_near_duplicates(
    frames: list[dict[str, Any]], max_distance: int = FRAME_DUPLICATE_DISTANCE
) -> list[dict[str, Any]]:
    """Remove frames that look the same as the last one kept.

    Scene detection reacts to lighting shifts and compression noise as well as
    to real changes, so a perceptual hash gives a second opinion.

    Args:
        frames: Frames with ``time`` and ``path`` keys, ordered by time.
        max_distance: Hamming distance below which two frames count as the same.

    Returns:
        The deduplicated list.
    """
    kept: list[dict[str, Any]] = []
    previous_hash: int | None = None
    for frame in frames:
        current_hash = dhash(Path(frame["path"]))
        if (
            previous_hash is not None
            and hamming_distance(previous_hash, current_hash) <= max_distance
        ):
            continue
        kept.append(frame)
        previous_hash = current_hash
    return kept


def cap_frame_count(
    frames: list[dict[str, Any]], max_frames: int = FRAME_MAX_COUNT
) -> list[dict[str, Any]]:
    """Thin the list down to at most ``max_frames``, spread evenly over time.

    Sampling evenly (rather than truncating) keeps coverage of the whole video
    when a busy recording produces far more candidates than the budget allows.

    Args:
        frames: Frames ordered by time.
        max_frames: Maximum number of frames to keep.

    Returns:
        At most ``max_frames`` frames, always including the first and last.
    """
    if max_frames <= 0:
        return []
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]

    last = len(frames) - 1
    step = last / (max_frames - 1)
    indices = sorted({round(position * step) for position in range(max_frames)})
    return [frames[index] for index in indices]


def extract_keyframes(
    video_path: str | Path,
    output_dir: Path | None = None,
    threshold: float = SCENE_THRESHOLD,
    min_interval: float = FRAME_MIN_INTERVAL_SECONDS,
    max_interval: float = FRAME_MAX_INTERVAL_SECONDS,
    max_frames: int = FRAME_MAX_COUNT,
) -> list[dict[str, Any]]:
    """Extract a small, representative set of frames from a video.

    One ffmpeg pass selects candidates (scene changes, plus a frame at least
    every ``max_interval`` seconds); the rest is thinning — minimum spacing,
    perceptual deduplication, then a hard cap. Every stage after the first only
    removes frames, so the number of images sent to a vision model is bounded.

    Args:
        video_path: Source video file.
        output_dir: Where to write frames. Defaults to ``<temp_dir>/frames``.
        threshold: Scene-change threshold between 0 and 1 (higher = fewer frames).
        min_interval: Minimum spacing in seconds between kept frames.
        max_interval: Longest stretch in seconds allowed without a frame.
        max_frames: Hard upper bound on the number of frames returned.

    Returns:
        Frames as dicts with ``time`` (seconds) and ``path``, ordered by time.

    Raises:
        VisualContextError: If ffmpeg cannot read the video.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = get_settings().temp_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Images left over from an earlier run would be picked up by the glob below
    # and pair the wrong timestamp with the wrong picture.
    for stale in output_dir.glob("*.jpg"):
        stale.unlink(missing_ok=True)

    interval = sample_interval(video_duration(video_path), max_interval)
    stderr = _run_frame_select(video_path, output_dir, threshold, interval)
    times = _parse_frame_times(stderr)
    written = sorted(output_dir.glob("frame_*.jpg"))
    if len(written) != len(times):
        # Pairing is positional, so a count mismatch means every timestamp after
        # the first discrepancy is wrong. Refuse rather than mislabel the video.
        raise VisualContextError(
            f"ffmpeg reported {len(times)} key frames but wrote {len(written)}; "
            "cannot match frames to timestamps"
        )
    frames = [
        {"time": time, "path": path} for time, path in zip(times, written, strict=True)
    ]
    logger.info("Frame selection produced %d candidates", len(frames))

    frames = apply_min_interval(frames, min_interval)
    frames = drop_near_duplicates(frames)
    frames = cap_frame_count(frames, max_frames)

    kept = {Path(frame["path"]) for frame in frames}
    for path in output_dir.glob("*.jpg"):
        if path not in kept:
            path.unlink(missing_ok=True)

    logger.info("Keeping %d key frames from %s", len(frames), video_path.name)
    return frames
