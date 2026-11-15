"""The app from the command line: the same run as the page, without the page.

Made for scripts and AI agents (Claude Code): one command per recording, one
JSON object on standard output, progress on standard error.

    python cli.py transcribe "C:/Meetings/2026-11-05 14-03-25.mkv"
    python cli.py estimate   "C:/Meetings/2026-11-05 14-03-25.mkv"
    python cli.py latest     "C:/Meetings"
    python cli.py pending    "C:/Meetings"

``transcribe`` does what Start does on the page (:func:`runs.run`): the same
engines, saved parts, guards, cost record and History row. The transcript is
then put beside the recording as ``transcript_<its name>.txt``. Exit codes: 0
done, 1 failed, 3 stopped part-way (run the same command again — saved parts
are not paid for twice), 4 not started (the transcript exists, or the estimate
is over the limit and ``--yes`` was not given); 2 is argparse's for a bad
command line.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import audio
import checkpoints
import db
import recordings
import runs
from config import (
    CLI_ENGINES,
    CLI_LATEST_COUNT,
    CLI_MAX_COST_USD,
    DEFAULT_FRAME_DETAIL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TITLE_MODEL,
    DEFAULT_VERIFY_MODEL,
    DEFAULT_VISION_MODEL,
    FRAME_INTERVAL_MAX_SECONDS,
    FRAME_INTERVAL_MIN_SECONDS,
    FRAME_MAX_INTERVAL_SECONDS,
    LOCAL_MODELS,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    RECORDING_PROBE_LIMIT,
    TIMESTAMP_MODELS,
    TITLE_MODE_APPEND,
    TITLE_MODE_OFF,
    TITLE_MODELS,
    TITLE_MODES,
    TRANSCRIPTION_MODELS,
    VISION_MODELS,
    get_settings,
)
from exceptions import AppError, CommandError
from logger import enable_file_log, get_logger

logger = get_logger(__name__)

EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_PARTIAL = 3
EXIT_NOT_STARTED = 4


def _say(message: str) -> None:
    """Write one line for whoever watches the run (standard error)."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _emit(payload: dict[str, Any]) -> None:
    """Write the command's one result object (standard output)."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def _seconds(text: str) -> int:
    """Read a moment of a recording: seconds, ``M:SS`` or ``H:MM:SS``.

    Args:
        text: What was typed after ``--from`` or ``--to``.

    Returns:
        The moment in whole seconds.

    Raises:
        argparse.ArgumentTypeError: If the text is none of those.
    """
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a time: write seconds, M:SS or H:MM:SS"
        )
    total = 0
    for part in parts:
        total = total * 60 + int(part)
    return total


def _parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser, with one sub-command per thing the app can be asked.
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Transcribe recordings without the page; answers in JSON.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("recording", type=Path, help="a video or audio file")
        command.add_argument(
            "--engine",
            choices=list(CLI_ENGINES),
            default="openai",
            help="openai (paid, default) or local (free, on this computer)",
        )
        command.add_argument(
            "--model",
            help=f"default: {DEFAULT_MODEL} for openai, "
            f"{DEFAULT_LOCAL_MODEL} for local",
        )
        command.add_argument(
            "--from", dest="start", type=_seconds, help="leave out what is before"
        )
        command.add_argument(
            "--to", dest="end", type=_seconds, help="leave out what is after"
        )
        command.add_argument(
            "--screens",
            action="store_true",
            help="describe what a video shows (OpenAI, paid) beside the speech",
        )
        command.add_argument(
            "--screens-every",
            type=int,
            default=int(FRAME_MAX_INTERVAL_SECONDS),
            metavar="SECONDS",
            help="look at the screen at least this often (default: %(default)s)",
        )
        command.add_argument(
            "--screens-model", choices=VISION_MODELS, default=DEFAULT_VISION_MODEL
        )

    transcribe = commands.add_parser(
        "transcribe", help="transcribe one recording and put the text beside it"
    )
    add_run_options(transcribe)
    transcribe.add_argument(
        "--srt", action="store_true", help="also write subtitles (.srt)"
    )
    transcribe.add_argument(
        "--verify",
        nargs="?",
        const=DEFAULT_VERIFY_MODEL,
        metavar="LOCAL_MODEL",
        help="check an OpenAI transcript against a local Whisper pass",
    )
    transcribe.add_argument(
        "--batched", action="store_true", help="decode the local pass in batches"
    )
    transcribe.add_argument(
        "--title",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ask for an AI title (default: the setting chosen on the page)",
    )
    transcribe.add_argument(
        "--out-dir", type=Path, help="default: the recording's own folder"
    )
    transcribe.add_argument(
        "--overwrite", action="store_true", help="replace an existing transcript"
    )
    transcribe.add_argument(
        "--max-cost",
        type=float,
        default=CLI_MAX_COST_USD,
        metavar="USD",
        help="refuse an OpenAI run estimated above this (default: %(default)s)",
    )
    transcribe.add_argument(
        "--yes", action="store_true", help="run whatever the estimate says"
    )

    estimate = commands.add_parser(
        "estimate", help="say how long a recording is and what a run would cost"
    )
    add_run_options(estimate)

    latest = commands.add_parser("latest", help="list a folder's newest recordings")
    latest.add_argument("folder", type=Path)
    latest.add_argument("--count", type=int, default=CLI_LATEST_COUNT)

    pending = commands.add_parser(
        "pending", help="list a folder's recordings that have no transcript yet"
    )
    pending.add_argument("folder", type=Path)
    return parser


def _local_installed() -> bool:
    """Return True when the offline engine (faster-whisper) is installed."""
    return importlib.util.find_spec("faster_whisper") is not None


def _existing_recording(path: Path) -> Path:
    """Return the recording's full path, or refuse a file the app cannot take."""
    recording = path.expanduser().resolve()
    if not recording.is_file():
        raise CommandError(f"There is no file {recording}.")
    if runs.source_type(recording) is None:
        raise CommandError(f"{recording.name} is not a supported video or audio file.")
    return recording


def _engine_and_model(args: argparse.Namespace) -> tuple[str, str]:
    """Return the engine and model asked for, checked against what exists."""
    provider = CLI_ENGINES[args.engine]
    if provider == PROVIDER_LOCAL:
        offered, default = list(LOCAL_MODELS), DEFAULT_LOCAL_MODEL
        if not _local_installed():
            raise CommandError(
                "The local engine is not installed: run `uv sync --extra local`."
            )
    else:
        offered, default = TRANSCRIPTION_MODELS, DEFAULT_MODEL
    model = args.model or default
    if model not in offered:
        raise CommandError(
            f"{model!r} is not a model of the {args.engine} engine: "
            f"choose from {', '.join(offered)}."
        )
    return provider, model


def _span(args: argparse.Namespace, length: float | None) -> tuple[int, int] | None:
    """Return the stretch of the recording asked for, or ``None`` for all of it."""
    if args.start is None and args.end is None:
        return None
    if args.end is None and length is None:
        raise CommandError(
            "The recording does not say how long it is: give --to as well."
        )
    start = args.start or 0
    end = args.end if args.end is not None else int(length or 0) + 1
    if length is not None:
        end = min(end, int(length) + 1)
    if start >= end:
        raise CommandError("--from must be earlier than --to and inside the recording.")
    return start, end


def _visual(args: argparse.Namespace, kind: str) -> dict[str, Any] | None:
    """Return the on-screen context settings asked for, if any."""
    if not args.screens:
        return None
    if kind != "video":
        raise CommandError("--screens needs a video: an audio file shows nothing.")
    if (
        not FRAME_INTERVAL_MIN_SECONDS
        <= args.screens_every
        <= (FRAME_INTERVAL_MAX_SECONDS)
    ):
        raise CommandError(
            f"--screens-every takes {FRAME_INTERVAL_MIN_SECONDS} to "
            f"{FRAME_INTERVAL_MAX_SECONDS} seconds."
        )
    return {
        "model": args.screens_model,
        "detail": DEFAULT_FRAME_DETAIL,
        "interval": float(args.screens_every),
    }


def _plan(args: argparse.Namespace, to_run: bool) -> dict[str, Any]:
    """Work out what a run of this command line would do, before any of it.

    Args:
        args: The parsed command line.
        to_run: True when the run follows, so a missing OpenAI key is refused
            here; an estimate needs none.

    Returns:
        ``recording``, ``kind``, ``provider``, ``model``, ``length`` (seconds
        or ``None``), ``span``, ``visual`` and ``estimate`` (USD or ``None``).
    """
    missing = audio.missing_tools()
    if missing:
        raise CommandError(audio.missing_tools_message(missing))
    recording = _existing_recording(args.recording)
    kind = runs.source_type(recording) or "audio"
    provider, model = _engine_and_model(args)
    length = recordings.duration(recording)
    span = _span(args, length)
    visual = _visual(args, kind)
    key = get_settings().openai_api_key
    if to_run and (provider == PROVIDER_OPENAI or visual) and not key:
        raise CommandError(
            "No OpenAI API key: set OPENAI_API_KEY in the app's .env file, "
            "or use --engine local."
        )
    seconds = float(span[1] - span[0]) if span else length
    return {
        "recording": recording,
        "kind": kind,
        "provider": provider,
        "model": model,
        "length": length,
        "span": span,
        "visual": visual,
        "seconds": seconds,
        "estimate": runs.estimate_cost(seconds, provider, model, visual),
    }


def _plan_summary(plan: dict[str, Any], engine: str) -> dict[str, Any]:
    """Return the part of a plan that every answer about a run carries."""
    return {
        "recording": str(plan["recording"]),
        "source_type": plan["kind"],
        "engine": engine,
        "model": plan["model"],
        "duration_seconds": plan["length"],
        "span": list(plan["span"]) if plan["span"] else None,
        "screens": plan["visual"],
        "estimated_cost_usd": plan["estimate"],
    }


def _title_setting(choice: bool | None) -> tuple[str, str]:
    """Return the AI title mode and model for a run from the command line.

    Args:
        choice: ``--title`` (True), ``--no-title`` (False) or neither (the
            setting chosen on the page, kept in the history database).

    Returns:
        The mode (a ``TITLE_MODES`` key) and the chat model.
    """
    stored = db.get_preferences()
    mode = stored.get("title_mode")
    if mode not in TITLE_MODES:
        mode = TITLE_MODE_OFF
    model = stored.get("title_model")
    if model not in TITLE_MODELS:
        model = DEFAULT_TITLE_MODEL
    if choice is False:
        mode = TITLE_MODE_OFF
    elif choice and mode == TITLE_MODE_OFF:
        mode = TITLE_MODE_APPEND
    return mode, model


def _progress() -> runs.ProgressCallback:
    """Return a progress hook that writes each new message once."""
    last: list[str] = []

    def report(info: dict[str, Any]) -> None:
        message = str(info.get("message", ""))
        if message and last != [message]:
            last[:] = [message]
            _say(message)

    return report


@contextmanager
def _stage(message: str) -> Iterator[None]:
    """Say what a long step is doing (the page shows a spinner there)."""
    _say(message)
    yield


def _recording_entry(path: Path, length: float | None) -> dict[str, Any]:
    """Describe one recording of a folder listing."""
    transcript = runs.sibling_transcript(path)
    try:
        stat = path.stat()
        size_mb: float | None = round(stat.st_size / (1024 * 1024), 1)
        saved: str | None = datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        )
    except OSError:
        size_mb = saved = None
    return {
        "recording": str(path),
        "saved": saved,
        "size_mb": size_mb,
        "duration_seconds": length,
        "transcript": str(transcript) if transcript.exists() else None,
    }


def _list_folder(folder: Path, count: int | None, only_pending: bool) -> dict[str, Any]:
    """List a folder's recordings, newest first.

    Args:
        folder: The recordings folder.
        count: How many to list at most, or ``None`` for all.
        only_pending: Leave out recordings that have a transcript beside them.

    Returns:
        ``folder`` and ``recordings`` (see :func:`_recording_entry`).
    """
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise CommandError(f"There is no folder {folder}.")
    try:
        found = recordings.list_recordings(folder)
    except OSError as exc:
        raise CommandError(f"{folder} cannot be read: {exc}") from exc
    if only_pending:
        found = [p for p in found if not runs.sibling_transcript(p).exists()]
    if count is not None:
        found = found[: max(0, count)]
    lengths = recordings.durations(found[:RECORDING_PROBE_LIMIT])
    return {
        "folder": str(folder),
        "recordings": [_recording_entry(path, lengths.get(path)) for path in found],
    }


def _transcribe(args: argparse.Namespace) -> int:
    """Run one recording through the app and put the transcript beside it.

    Returns:
        The exit code (see the module docstring).
    """
    plan = _plan(args, to_run=True)
    recording: Path = plan["recording"]
    provider, model = plan["provider"], plan["model"]
    answer = _plan_summary(plan, args.engine)
    if args.srt and provider == PROVIDER_OPENAI and model not in TIMESTAMP_MODELS:
        raise CommandError(
            f"{model} returns no timestamps: for --srt use --model "
            f"{sorted(TIMESTAMP_MODELS)[0]} or --engine local."
        )
    verify = args.verify
    if verify:
        if provider != PROVIDER_OPENAI or model in TIMESTAMP_MODELS:
            raise CommandError(
                "--verify is for OpenAI models without timestamps "
                f"({DEFAULT_MODEL}); the others have exact times already."
            )
        if verify not in LOCAL_MODELS or not _local_installed():
            raise CommandError(
                "--verify needs the local engine (`uv sync --extra local`) and "
                f"one of its models: {', '.join(LOCAL_MODELS)}."
            )
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else None
    if out_dir is not None and not out_dir.is_dir():
        raise CommandError(f"There is no folder {out_dir}.")
    target = runs.sibling_transcript(recording, out_dir)
    if target.exists() and not args.overwrite:
        _emit(
            {
                **answer,
                "status": "exists",
                "transcript": str(target),
                "message": "This recording already has a transcript; "
                "--overwrite replaces it.",
            }
        )
        return EXIT_NOT_STARTED
    estimate = plan["estimate"]
    paid = provider == PROVIDER_OPENAI or plan["visual"]
    if paid and not args.yes and (estimate is None or estimate > args.max_cost):
        _emit(
            {
                **answer,
                "status": "needs_confirmation",
                "max_cost_usd": args.max_cost,
                "message": "Nothing was sent. The estimate is over the limit"
                if estimate is not None
                else "Nothing was sent. The recording does not say how long it "
                "is, so no estimate can be made",
            }
        )
        return EXIT_NOT_STARTED

    settings = get_settings()
    stem = runs.safe_stem(recording.name)
    work = Path(tempfile.mkdtemp(prefix="cli-", dir=settings.temp_dir))
    results = SimpleNamespace(
        transcript_path=None,
        srt_path=None,
        elapsed_seconds=None,
        cost=None,
        partial=None,
        title=None,
        row_id=None,
        notices=[],
    )
    keep_work = False
    try:
        with checkpoints.active_run():
            if plan["kind"] == "video":
                _say("Extracting audio from video...")
            audio_path, video_path, kind = runs.prepare_source(
                recording, work / f"{stem}.wav"
            )
            request = runs.RunRequest(
                filename=recording.name,
                source_type=kind,
                audio_path=audio_path,
                video_path=video_path,
                transcript_path=work / f"transcript_{stem}.txt",
                srt_path=work / f"transcript_{stem}.srt",
                provider=provider,
                model=model,
                with_timestamps=args.srt,
                visual=plan["visual"],
                span=plan["span"],
                verify=verify,
                batched=args.batched,
                api_key=settings.openai_api_key,
                local_available=_local_installed(),
            )
            title = _title_setting(args.title)
            hooks = runs.RunHooks(
                progress=_progress(),
                stage=_stage,
                load_local_model=runs.load_local_model,
                title_setting=lambda: title,
            )
            runs.run(request, results, hooks)

        cost = results.cost or {}
        answer |= {
            "title": results.title,
            "row_id": results.row_id,
            "elapsed_seconds": results.elapsed_seconds,
            "cost_usd": cost.get("cost_usd"),
            "cost_estimated": cost.get("estimated"),
            "notices": [
                {"level": level, "message": message}
                for level, message in results.notices or []
            ],
        }
        if results.row_id is not None:
            shutil.copyfile(request.transcript_path, target)
            srt_target = None
            if results.srt_path:
                srt_target = target.with_suffix(".srt")
                shutil.copyfile(results.srt_path, srt_target)
            _emit(
                {
                    **answer,
                    "status": "done",
                    "transcript": str(target),
                    "srt": str(srt_target) if srt_target else None,
                }
            )
            return EXIT_DONE
        if results.partial:
            # The partial transcript stays in temp/: beside the recording it
            # would pass for a finished one.
            keep_work = True
            _emit(
                {
                    **answer,
                    "status": "partial",
                    "parts": results.partial,
                    "partial_transcript": str(results.transcript_path),
                    "message": "Run the same command again to transcribe only "
                    "the rest; saved parts are not sent or paid for again.",
                }
            )
            return EXIT_PARTIAL
        errors = [m for level, m in results.notices or [] if level == "error"]
        _emit({**answer, "status": "failed", "message": " ".join(errors) or None})
        return EXIT_FAILED
    finally:
        if keep_work:
            # Only the partial transcript is worth keeping, not the audio.
            for leftover in work.glob("*.wav"):
                with suppress(OSError):
                    leftover.unlink()
        else:
            shutil.rmtree(work, ignore_errors=True)


def _estimate(args: argparse.Namespace) -> int:
    """Say how long a recording is and what the run asked for would cost."""
    plan = _plan(args, to_run=False)
    transcript = runs.sibling_transcript(plan["recording"])
    _emit(
        {
            **_plan_summary(plan, args.engine),
            "status": "estimate",
            "seconds_to_transcribe": plan["seconds"],
            "max_cost_usd": CLI_MAX_COST_USD,
            "transcript": str(transcript) if transcript.exists() else None,
        }
    )
    return EXIT_DONE


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: The arguments, without the program name (default: the process's).

    Returns:
        The exit code (see the module docstring).
    """
    for stream in (sys.stdout, sys.stderr):
        # A pipe on Windows gets the ANSI code page, which cannot write "š".
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    try:
        if args.command == "latest":
            _emit({"status": "listed", **_list_folder(args.folder, args.count, False)})
            return EXIT_DONE
        if args.command == "pending":
            _emit({"status": "listed", **_list_folder(args.folder, None, True)})
            return EXIT_DONE
        if args.command == "estimate":
            return _estimate(args)
        enable_file_log()
        db.init_db()
        return _transcribe(args)
    except AppError as exc:
        _emit({"status": "failed", "message": str(exc)})
        return EXIT_FAILED
    except Exception as exc:
        # Whatever went wrong, the caller reads JSON, not a traceback.
        logger.exception("Command failed")
        _emit({"status": "failed", "message": f"Unexpected error: {exc}"})
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
