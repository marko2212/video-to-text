"""Key-frame extraction for on-screen (visual) context.

Sampling a video at a fixed interval is wasteful: a talking head produces
hundreds of near-identical images, and every one of them costs vision tokens.
Instead this module asks ffmpeg's ``select`` filter for the frames where the
picture actually changed, drops perceptual near-duplicates, and applies interval
and count guardrails so the cost of a long screen-share stays predictable.

Like :mod:`audio`, this module is UI-agnostic — it never imports Streamlit.
"""

import json
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import ffmpeg
from PIL import Image

from config import (
    FRAME_DUPLICATE_DISTANCE,
    FRAME_INTERVAL_MIN_SECONDS,
    FRAME_MAX_INTERVAL_SECONDS,
    FRAME_MIN_INTERVAL_SECONDS,
    FRAME_QUALITY,
    FRAME_UNINDEXED_FORMATS,
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
# The `metadata` filter prints each selected frame's scene score on its own line.
_SCENE_SCORE = re.compile(r"Parsed_metadata.*?lavfi\.scene_score=(\d+(?:\.\d+)?)")
# Bump when the scan's content or selection rules change, so old scans are redone.
# 2: frames record whether their JPEG is shared, and the scan its threshold and grid.
# 3: no sharing in containers without an index (their frames cannot be re-extracted
#    exactly); the length falls back to the last frame when the file has none.
_SCAN_VERSION = 3
_SCAN_INDEX = "index.json"
# Zero-padded so lexical sorting of the written files matches numeric order.
_FRAME_PATTERN = "frame_%05d.jpg"
# A clip shorter than the chosen interval would be represented by a single
# frame, so the interval tightens just enough to get this many samples out of it.
# Deliberately small: beyond avoiding that, the chosen cadence is left alone.
_MIN_SAMPLES_PER_VIDEO = 2
_MIN_SAMPLE_SECONDS = 5.0
# One scan per folder at a time in this process: every rerun during a long scan
# (a slider move, Start) used to start another full decode of the same video.
_scan_locks: dict[str, threading.Lock] = {}
_scan_locks_guard = threading.Lock()


def max_frames_setting() -> int:
    """Return the configured cap on frames per video.

    Read at call time rather than baked into a default argument, so setting
    ``FRAME_MAX_COUNT`` in ``.env`` actually takes effect.

    Returns:
        The maximum number of frames to describe for one video.
    """
    return get_settings().frame_max_count


def _ffmpeg_error_detail(exc: ffmpeg.Error) -> str:
    """Return the stderr text carried by an ffmpeg error, if any.

    Args:
        exc: The exception raised by ``ffmpeg.run``.

    Returns:
        The decoded stderr output, falling back to the exception text.
    """
    return exc.stderr.decode(errors="replace") if exc.stderr else str(exc)


def _last_lines(detail: str, count: int = 2) -> str:
    """Keep the end of an ffmpeg log, where the cause is, for a user message.

    The whole log (version banner, build flags, stream dump) ran to over two
    thousand characters in a warning; it stays in the log file.

    Args:
        detail: ffmpeg's stderr.
        count: Non-empty lines to keep.

    Returns:
        The last ``count`` non-empty lines, joined by spaces.
    """
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    return " ".join(lines[-count:])


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


def effective_interval(
    duration: float,
    max_interval: float,
    max_frames: int | None = None,
) -> float:
    """Return how often a frame is taken, ignoring scene changes.

    The requested interval is adjusted at both ends: tightened when the video is
    too short for it to fire more than once, and widened when honouring it would
    blow past the frame cap (the rough estimate used when there is no scan; the
    scan-based selection passes ``max_frames=0`` and caps afterwards).

    Args:
        duration: Video duration in seconds; 0 when unknown.
        max_interval: The requested upper bound between samples.
        max_frames: Cap on the number of frames; defaults to the setting, and
            0 means no widening.

    Returns:
        The interval the extraction will really use.
    """
    if duration <= 0:
        return max_interval
    if max_frames is None:
        max_frames = max_frames_setting()
    interval = min(
        max_interval, max(duration / _MIN_SAMPLES_PER_VIDEO, _MIN_SAMPLE_SECONDS)
    )
    if max_frames > 0:
        interval = max(interval, duration / max_frames)
    return interval


def estimate_frame_count(
    duration: float,
    max_interval: float = FRAME_MAX_INTERVAL_SECONDS,
    max_frames: int | None = None,
) -> int:
    """Estimate how many frames a video will yield, for the pre-run UI hint.

    Only the frames the interval guarantees are counted — one at the start and
    one per interval after that. Scene changes push the real number up and
    deduplication pulls it back down, so this is an indication, not a promise.

    Args:
        duration: Video duration in seconds; 0 when unknown.
        max_interval: Requested longest stretch without a frame.
        max_frames: Hard cap applied to the selection; defaults to the setting.

    Returns:
        The estimated frame count, or 0 when the duration is unknown.
    """
    if duration <= 0:
        return 0
    if max_frames is None:
        max_frames = max_frames_setting()
    interval = effective_interval(duration, max_interval, max_frames)
    return min(max_frames, int(duration // interval) + 1)


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
    """Keep one frame per burst: the last one, once the picture has settled.

    A hard cut or a camera pan can trip scene detection several times within a
    second; one of those is enough. It must be the *last* of the burst: keeping
    the first threw away the frame that showed the new slide whenever a slide
    changed less than ``min_interval`` after an interval sample — the old slide
    was kept (and later dropped as a duplicate), so the new one appeared a whole
    interval late, or not at all if it was gone by then. A burst is measured
    from its first frame, so continuous motion still yields a frame every
    ``min_interval`` seconds instead of collapsing into one.

    Args:
        frames: Frames with a ``time`` key, ordered by time.
        min_interval: Length in seconds of the window treated as one burst.

    Returns:
        The thinned list.
    """
    kept: list[dict[str, Any]] = []
    burst_start = 0.0
    for frame in frames:
        if kept and frame["time"] - burst_start < min_interval:
            kept[-1] = frame
            continue
        kept.append(frame)
        burst_start = frame["time"]
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
        # A scanned frame carries its hash; hashing again would decode every JPEG
        # on each move of the slider.
        current_hash = frame["hash"] if "hash" in frame else dhash(Path(frame["path"]))
        if (
            previous_hash is not None
            and hamming_distance(previous_hash, current_hash) <= max_distance
        ):
            continue
        kept.append(frame)
        previous_hash = current_hash
    return kept


def cap_frame_count(
    frames: list[dict[str, Any]], max_frames: int | None = None
) -> list[dict[str, Any]]:
    """Thin the list down to at most ``max_frames``, spread evenly over time.

    Sampling evenly (rather than truncating) keeps coverage of the whole video
    when a busy recording produces far more candidates than the budget allows.

    Args:
        frames: Frames ordered by time.
        max_frames: Maximum number of frames to keep; defaults to the setting.

    Returns:
        At most ``max_frames`` frames, always including the first and last.
    """
    if max_frames is None:
        max_frames = max_frames_setting()
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


def _parse_scene_scores(stderr: str) -> list[float]:
    """Pull each selected frame's scene score out of ffmpeg's log.

    Args:
        stderr: Captured stderr of a pass that ran the ``metadata`` filter.

    Returns:
        One score (0-1) per selected frame, in order.
    """
    return [float(match) for match in _SCENE_SCORE.findall(stderr)]


def _run_scan(video_path: Path, output_dir: Path, threshold: float, grid: float) -> str:
    """Write every scene change plus a frame every ``grid`` seconds, with scores.

    Scene detection alone is not enough: ffmpeg's score is tuned for natural
    footage, and a measured full-screen slide change scored only 0.077. So the
    same pass also takes a frame whenever nothing was selected for ``grid``
    seconds. Variable frame rate output is essential: without it ffmpeg pads
    the result back to a constant rate by duplicating frames, which leaves the
    images silently misaligned with the timestamps.

    Args:
        video_path: Source video file.
        output_dir: Directory the JPEG frames are written into.
        threshold: Scene-change threshold between 0 and 1.
        grid: Seconds between the frames taken regardless of scene changes.

    Returns:
        ffmpeg's stderr: a ``showinfo`` line (time) and a ``metadata`` line
        (scene score) per written frame.

    Raises:
        VisualContextError: If ffmpeg cannot read the video.
    """
    select_expression = f"eq(n,0)+gt(scene,{threshold})+gte(t-prev_selected_t,{grid})"
    stream = ffmpeg.input(str(video_path))
    stream = (
        stream.filter("select", select_expression)
        .filter("metadata", mode="print", key="lavfi.scene_score")
        .filter("showinfo")
    )
    stream = ffmpeg.output(
        stream,
        str(output_dir / _FRAME_PATTERN),
        **_rate_control_kwarg(),
        **{"qscale:v": FRAME_QUALITY},
    )
    try:
        _, stderr = ffmpeg.run(
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        logger.error("Video scan failed for %s: %s", video_path, detail)
        raise VisualContextError(
            f"Failed to scan the video: {_last_lines(detail)}"
        ) from exc
    return stderr.decode(errors="replace")


def load_scan(
    scan_dir: Path,
    threshold: float = SCENE_THRESHOLD,
    grid: float = FRAME_INTERVAL_MIN_SECONDS,
) -> dict[str, Any] | None:
    """Return a finished scan from disk without ever starting one.

    Args:
        scan_dir: The scan's folder.
        threshold: Scene-change threshold the scan must have been made with.
        grid: Grid step the scan must have been made with.

    Returns:
        The scan, or ``None`` when there is no complete, current one.
    """
    return _load_scan(scan_dir, threshold, grid)


def _probe(video_path: Path) -> dict[str, Any]:
    """Read what the scan needs to know about a file before decoding it.

    Args:
        video_path: Source video file.

    Returns:
        ``has_video``, ``duration`` (0.0 when the file does not say) and
        ``indexed`` (False for containers that cannot be seeked exactly).

    Raises:
        VisualContextError: If ffprobe cannot read the file.
    """
    try:
        metadata = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        logger.error("ffprobe failed for %s: %s", video_path, detail)
        raise VisualContextError(
            f"Cannot read the video: {_last_lines(detail)}"
        ) from exc
    container = metadata.get("format", {})
    try:
        duration = float(container.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    names = set(str(container.get("format_name", "")).split(","))
    return {
        "has_video": any(
            stream.get("codec_type") == "video"
            for stream in metadata.get("streams", [])
        ),
        "duration": duration,
        "indexed": not names & set(FRAME_UNINDEXED_FORMATS),
    }


def _load_scan(scan_dir: Path, threshold: float, grid: float) -> dict[str, Any] | None:
    """Return a finished scan from disk, or ``None`` if it is missing or stale."""
    try:
        scan = json.loads((scan_dir / _SCAN_INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (scan.get("version"), scan.get("threshold"), scan.get("grid")) != (
        _SCAN_VERSION,
        threshold,
        grid,
    ):
        return None
    for frame in scan["frames"]:
        frame["path"] = scan_dir / frame["file"]
        if not frame["path"].exists():
            return None
    return scan


def scan_video(
    video_path: str | Path,
    scan_dir: Path,
    threshold: float = SCENE_THRESHOLD,
    grid: float = FRAME_INTERVAL_MIN_SECONDS,
) -> dict[str, Any]:
    """Look through a video once, so the screenshot count can be known up front.

    The count used to be estimated from the interval alone, and was wrong both
    ways: on four real recordings a static meeting estimated at 231 screenshots
    described 10, and a busy 16-minute screen share estimated at 4 described 28 —
    because what is described is the number of *distinct* screens. One pass
    keeps every scene change plus a frame every ``grid`` seconds (the finest
    interval the UI offers) and hashes them; :func:`select_from_scan` then gives
    the exact selection for any interval, and the run describes those very
    frames instead of decoding the video a second time.

    Args:
        video_path: Source video file.
        scan_dir: Where the frames and ``index.json`` live; reused when complete.
        threshold: Scene-change threshold between 0 and 1.
        grid: Seconds between frames taken regardless of scene changes.

    Returns:
        ``duration``, ``width`` and ``height`` of the frames, and ``frames``:
        dicts with ``time``, ``path``, ``scene`` (a scene change, not a grid
        sample), ``hash`` and ``shared`` (``path`` is an earlier frame's
        near-identical picture — :func:`pictures` gives the frame's own).

    Raises:
        VisualContextError: If ffmpeg cannot read the video or its log does not
            match the frames it wrote.
    """
    video_path = Path(video_path)
    with _scan_locks_guard:
        lock = _scan_locks.setdefault(str(scan_dir.resolve()), threading.Lock())
    # A second caller waits for the first scan and then loads it.
    with lock:
        existing = _load_scan(scan_dir, threshold, grid)
        if existing is not None:
            return existing
        # A folder without a readable index is an interrupted or outdated scan.
        shutil.rmtree(scan_dir, ignore_errors=True)
        return _scan(video_path, scan_dir, threshold, grid)


def _scan(
    video_path: Path, scan_dir: Path, threshold: float, grid: float
) -> dict[str, Any]:
    """Run the scan pass and store its result (see :func:`scan_video`)."""
    info = _probe(video_path)
    if not info["has_video"]:
        raise VisualContextError(
            "This file has no picture, so there is nothing to describe."
        )
    scan_dir.parent.mkdir(parents=True, exist_ok=True)
    # Built in a folder of its own and renamed when complete, so a scan that is
    # interrupted — or runs in two tabs at once — never leaves a half index.
    work = Path(tempfile.mkdtemp(prefix=f"{scan_dir.name}-", dir=scan_dir.parent))
    try:
        stderr = _run_scan(video_path, work, threshold, grid)
        times = _parse_frame_times(stderr)
        scores = _parse_scene_scores(stderr)
        written = sorted(work.glob("frame_*.jpg"))
        if not len(times) == len(scores) == len(written):
            raise VisualContextError(
                f"ffmpeg reported {len(times)} frames and {len(scores)} scores "
                f"but wrote {len(written)}; cannot match frames to timestamps"
            )
        width = height = 0
        if written:
            with Image.open(written[0]) as first:
                width, height = first.size
        scanned = []
        anchor: dict[str, Any] | None = None
        for time, score, path in zip(times, scores, written, strict=True):
            frame = {
                "time": time,
                "file": path.name,
                "scene": score > threshold,
                "hash": dhash(path),
                "shared": False,
            }
            # Most frames of a static screen are the same picture: keep one image
            # per run of near-identical frames (a two-hour meeting went from
            # 253 MB of frames to a few), while every frame keeps its own time
            # and hash, so the selection is unchanged. "Near-identical" is only
            # the hash's opinion — two slides of one template can hash alike —
            # so a shared frame is marked, and the run describes its own picture.
            # That picture is re-extracted by seeking, which is exact only in a
            # container with an index: in MPEG-TS or MPEG-PS the seek landed up to
            # a keyframe interval later, so there every frame keeps its own JPEG.
            if (
                info["indexed"]
                and anchor is not None
                and (
                    hamming_distance(anchor["hash"], frame["hash"])
                    <= FRAME_DUPLICATE_DISTANCE
                )
            ):
                frame["file"] = anchor["file"]
                frame["shared"] = True
                path.unlink(missing_ok=True)
            else:
                anchor = frame
            scanned.append(frame)
        scan = {
            "version": _SCAN_VERSION,
            "threshold": threshold,
            "grid": grid,
            # A browser recording (WebM) often has no length in its header;
            # its last frame is the best figure, at most one grid step short.
            "duration": info["duration"] or (times[-1] if times else 0.0),
            "width": width,
            "height": height,
            "frames": scanned,
        }
        (work / _SCAN_INDEX).write_text(json.dumps(scan), encoding="utf-8")
        try:
            work.rename(scan_dir)
        except OSError:
            # Another tab finished the same scan first; use that one.
            shutil.rmtree(work, ignore_errors=True)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    logger.info("Scanned %s: %d candidate frames", video_path.name, len(scan["frames"]))
    loaded = _load_scan(scan_dir, threshold, grid)
    if loaded is None:
        raise VisualContextError("The video scan could not be read back")
    return loaded


def candidates_from_scan(
    scan: dict[str, Any],
    max_interval: float,
    min_interval: float = FRAME_MIN_INTERVAL_SECONDS,
) -> list[dict[str, Any]]:
    """Pick the frames worth describing from a finished scan, before the cap.

    Every scene change, plus a frame when nothing was taken for
    ``max_interval`` seconds (the first grid frame at or after that point, so at
    most one grid step late), then one frame per burst and near-duplicates
    dropped. Pure and fast — it runs on every move of the slider.

    Args:
        scan: Result of :func:`scan_video`.
        max_interval: Requested longest stretch without a frame.
        min_interval: Length of the window treated as one burst.

    Returns:
        Frames with ``time``, ``path`` and ``hash``, ordered by time.
    """
    # Only tightened for short videos, never widened for the cap: grid frames
    # are 5 s apart, so a widened 5.1 s snapped to 10 s and gave half the
    # frames the cap allows. The cap thins the result evenly instead.
    interval = effective_interval(scan["duration"], max_interval, max_frames=0)
    chosen: list[dict[str, Any]] = []
    last: float | None = None
    for frame in scan["frames"]:
        # A small tolerance: grid times come from ffmpeg as rounded decimals.
        if last is None or frame["scene"] or frame["time"] - last >= interval - 1e-3:
            chosen.append(frame)
            last = frame["time"]
    chosen = apply_min_interval(chosen, min_interval)
    return drop_near_duplicates(chosen)


def select_from_scan(
    scan: dict[str, Any],
    max_interval: float,
    max_frames: int | None = None,
    min_interval: float = FRAME_MIN_INTERVAL_SECONDS,
) -> list[dict[str, Any]]:
    """Pick the frames a run describes: :func:`candidates_from_scan`, capped.

    The run uses exactly this result, and the count shown before it is its
    length.

    Args:
        scan: Result of :func:`scan_video`.
        max_interval: Requested longest stretch without a frame.
        max_frames: Hard cap; defaults to the setting.
        min_interval: Length of the window treated as one burst.

    Returns:
        Frames with ``time``, ``path`` and ``hash``, ordered by time.
    """
    candidates = candidates_from_scan(scan, max_interval, min_interval)
    return cap_frame_count(candidates, max_frames)


def extract_frame(video_path: Path, time: float, output: Path) -> Path:
    """Write the frame shown at ``time`` as a JPEG, by seeking to it.

    Args:
        video_path: Source video file.
        time: Position in seconds, as the scan reported it.
        output: JPEG file to write.

    Returns:
        ``output``.

    Raises:
        VisualContextError: If ffmpeg fails or writes nothing (e.g. past the end).
    """
    # The scan's times are rounded to microseconds; seeking a hair earlier makes
    # sure the frame itself is not skipped for starting a fraction later.
    stream = ffmpeg.input(str(video_path), ss=f"{max(time - 0.0005, 0.0):.6f}")
    stream = ffmpeg.output(
        stream, str(output), vframes=1, **{"qscale:v": FRAME_QUALITY}
    )
    try:
        ffmpeg.run(
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        logger.warning("Frame extraction at %.1f s failed: %s", time, detail)
        raise VisualContextError(
            f"Could not extract the frame at {time:.1f} s: {_last_lines(detail)}"
        ) from exc
    if not output.is_file():
        raise VisualContextError(f"ffmpeg wrote no frame at {time:.1f} s")
    return output


@contextmanager
def pictures(video_path: Path) -> Iterator[Callable[[dict[str, Any]], Path]]:
    """Give each selected frame its own picture, for the time it is labelled with.

    The scan keeps one JPEG per run of frames whose hashes are within
    ``FRAME_DUPLICATE_DISTANCE`` bits, but hashes that close can still be
    different screens — two slides of one template, a few more lines of code.
    Describing the shared JPEG put an earlier screen under a later time, and a
    slide that only ever shared a picture was never described at all. So a
    frame whose JPEG is shared is taken from the video again, at its own time,
    when it is about to be described (a fraction of a second each, and only for
    frames not described before).

    Args:
        video_path: The scanned video.

    Yields:
        A function returning the picture of a selected frame; the extracted
        files are deleted when the block exits.
    """
    temp_dir = get_settings().temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="frames-", dir=temp_dir))

    def picture(frame: dict[str, Any]) -> Path:
        if not frame.get("shared"):
            return Path(frame["path"])
        output = work / f"frame_{round(frame['time'] * 1000):09d}.jpg"
        if not output.is_file():
            extract_frame(Path(video_path), frame["time"], output)
        # A check on the seek: the frame must hash like the one the scan saw.
        # If not, the shared JPEG — within a few bits of it — is the better guess.
        if hamming_distance(dhash(output), frame["hash"]) > FRAME_DUPLICATE_DISTANCE:
            logger.warning(
                "The frame at %.1f s came back different from the scan's; "
                "describing the shared picture instead",
                frame["time"],
            )
            return Path(frame["path"])
        return output

    try:
        yield picture
    finally:
        shutil.rmtree(work, ignore_errors=True)
