"""Demo material for the README screenshots and GIF: recordings, history, crop.

Used by ``run.ps1`` next to it. Everything it writes goes to a folder outside
the repo: made-up recordings, a copy of the app and that copy's own database.
``prefs`` and ``history`` refuse a data folder inside the repo, so the real
history (confidential meetings) can never get demo rows.

    python demo.py copy-app <repo> <dest>
    python demo.py media <out>          # needs <out>/tts/line*.wav and ffmpeg
    python demo.py prefs <recordings>   # with the app copy on PYTHONPATH
    python demo.py history              # likewise
    python demo.py crop <shots> <target>
    python demo.py gif <manifest.json> <target.gif>   # frames from shoot.mjs video
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import wave
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

from PIL import Image, ImageChops, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
RATE = 22050
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
# The recording that is transcribed: the meeting ends after about 1:36, the
# recorder runs on to 9:25.
MEETING = "Sprint planning.mkv"
MEETING_SECONDS = 9 * 60 + 25
MEETING_START = "2026-10-20 10:02"
# The rest of the folder list: silence with believable lengths and dates.
OTHERS = {
    "Client onboarding call.mkv": (47 * 60 + 12, "2026-10-14 11:05"),
    "Voice memo - budget ideas.m4a": (3 * 60 + 41, "2026-10-17 18:44"),
    "Weekly sync.mp4": (32 * 60 + 5, "2026-10-09 15:31"),
    "Product demo.mkv": (62 * 60 + 37, "2026-10-02 14:10"),
}
SHOTS = ("home.png", "transcribe.png", "history.png")


def _log(message: str) -> None:
    sys.stdout.write(message + "\n")


def _ffmpeg(*args: str) -> None:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise SystemExit("ffmpeg is not on PATH")
    subprocess.run([binary, "-y", "-loglevel", "error", *args], check=True)


def copy_app(repo: Path, dest: Path) -> None:
    """Copy the repo's tracked files (working-tree versions) into a new folder.

    Untracked files stay behind — ``.env`` with the API key, ``data/``,
    ``temp/``, ``uploads/`` — so the copy runs with no key and no history.

    Args:
        repo: The repo root.
        dest: The folder to create.
    """
    listed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, check=True
    ).stdout.decode()
    for name in filter(None, listed.split("\0")):
        source = repo / name
        if source.is_file():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if (dest / ".env").exists():
        raise SystemExit(".env must not be in the app copy")
    _log(f"app copied to {dest}")


def _noise(seconds: float, level: int = 40) -> bytes:
    count = int(seconds * RATE)
    samples = (int(random.gauss(0, level)) for _ in range(count))
    return b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)


def _slide(path: Path, title: str, lines: list[str], colors: tuple[str, str]) -> None:
    image = Image.new("RGB", (1280, 720), colors[0])
    draw = ImageDraw.Draw(image)
    draw.text((90, 90), title, font=ImageFont.truetype(BOLD, 54), fill=colors[1])
    for row, text in enumerate(lines):
        font = ImageFont.truetype(FONT, 36)
        draw.text((110, 210 + row * 70), text, font=font, fill=colors[1])
    image.save(path)


def media(out: Path) -> None:
    """Make the demo recordings in ``<out>/Recordings`` and their dates.

    The meeting is the spoken lines in ``<out>/tts`` (Windows text-to-speech,
    made by ``run.ps1``) with room noise between them and a quiet tail, over
    two plain slides. ``<out>/times.json`` lists when each recording started
    and was saved; ``run.ps1`` sets those on the files, since the list shows
    when a recording was started.

    Args:
        out: The working folder.
    """
    random.seed(7)
    recordings, work = out / "Recordings", out / "work"
    recordings.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    frames = [_noise(3.0)]
    for index, part in enumerate(sorted((out / "tts").glob("line*.wav"))):
        with wave.open(str(part), "rb") as spoken:
            if (spoken.getframerate(), spoken.getnchannels()) != (RATE, 1):
                raise SystemExit(f"{part.name}: expected {RATE} Hz mono")
            frames.append(spoken.readframes(spoken.getnframes()))
        frames.append(_noise(0.9 if index % 2 == 0 else 1.3))
    speech = sum(len(chunk) for chunk in frames) / 2 / RATE
    frames.append(_noise(MEETING_SECONDS - speech))
    audio = work / "meeting.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(RATE)
        for chunk in frames:
            target.writeframes(chunk)
    _log(f"speech ends at {speech:.1f} s of {MEETING_SECONDS} s")

    _slide(
        work / "slide1.png",
        "Sprint planning — week 43",
        [
            "•  Done: subtitle export, resumable runs",
            "•  Almost ready: screenshots on change",
            "•  Next: faster history page",
            "•  Next: clearer messages when credit runs out",
            "•  Next: new screenshots in the docs",
        ],
        ("#f4f6fb", "#1f2a44"),
    )
    _slide(
        work / "slide2.png",
        "Meeting ended",
        ["The recording is still running…"],
        ("#20242c", "#c9ced8"),
    )
    first = round(speech + 2)
    _ffmpeg(
        *("-loop", "1", "-t", str(first), "-i", str(work / "slide1.png")),
        *("-loop", "1", "-t", str(MEETING_SECONDS - first)),
        *("-i", str(work / "slide2.png"), "-i", str(audio)),
        *("-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,fps=2,format=yuv420p[v]"),
        *("-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-tune", "stillimage"),
        *("-crf", "30", "-c:a", "aac", "-b:a", "96k", str(recordings / MEETING)),
    )
    for name, (seconds, _) in OTHERS.items():
        _ffmpeg(
            *("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(seconds)),
            *("-c:a", "aac", "-b:a", "16k", str(recordings / name)),
        )

    times = {}
    for name, (seconds, start) in {
        MEETING: (MEETING_SECONDS, MEETING_START),
        **OTHERS,
    }.items():
        started = datetime.strptime(start, "%Y-%m-%d %H:%M")
        saved = started + timedelta(seconds=seconds)
        times[name] = {"created": started.isoformat(), "saved": saved.isoformat()}
    (out / "times.json").write_text(json.dumps(times, indent=1), encoding="utf-8")
    _log(f"{len(times)} recordings in {recordings}")


def _isolated_db() -> ModuleType:
    """Return the app copy's ``db`` module, refusing a data folder in the repo."""
    # From the app copy on PYTHONPATH (run.ps1 sets it), never the repo.
    import db
    from config import get_settings

    data = get_settings().data_dir.resolve()
    if data.is_relative_to(REPO) or Path(db.__file__).resolve().is_relative_to(REPO):
        raise SystemExit(f"refusing to write demo rows: {data} is in the repo")
    db.init_db()
    return db


def prefs(recordings: Path) -> None:
    """Start the page on the folder source, showing the demo recordings.

    Args:
        recordings: The demo recordings folder.
    """
    db = _isolated_db()
    db.set_preference("source", "folder")
    db.set_preference("recordings_folder", str(recordings))
    db.set_preference("title_mode", "off")


_ONBOARDING = """\
🖥️ (0:04) A shared screen shows the slide "Kickoff — onboarding plan" with the \
agenda: introductions, data import, training, go-live.

(~0:06) Thanks for joining, everyone. Today we want to agree on the onboarding \
plan, so your team can start working in the new system by the end of next month.

(~1:12) We already have the export from your old system. It holds about forty \
thousand customer records and the order history for the last three years.

🖥️ (2:31) A spreadsheet titled "Field mapping" lists the old column names next \
to the new ones; three rows are highlighted in yellow.

(~2:35) These three fields have no match yet — the delivery notes, the second \
phone number and the customer category. We need your decision on those before \
the first test import.

(~4:02) For training we suggest two sessions: one for the sales team and one \
for the warehouse. Each takes about an hour, and we record both.

(~5:40) Go-live would be on the third of November, with the old system kept \
read-only for one more month."""

_ROWS = [
    {
        "filename": "Product demo.mkv",
        "source_type": "video",
        "model": "gpt-4o-transcribe",
        "provider": "OpenAI API",
        "with_timestamps": False,
        "file_size_mb": 114.7,
        "duration_minutes": 62.62,
        "elapsed_seconds": 288.4,
        "cost_usd": 0.31,
        "title": "Walkthrough of the new reporting dashboard",
        "transcript": "(~0:03) Let me share my screen and show you the new reports.",
        "created_at": "2026-10-02 15:20:41",
    },
    {
        "filename": "Weekly sync.mp4",
        "source_type": "video",
        "model": "whisper-1",
        "provider": "OpenAI API",
        "with_timestamps": True,
        "file_size_mb": 61.6,
        "duration_minutes": 32.08,
        "elapsed_seconds": 118.2,
        "cost_usd": 0.19,
        "title": "Release dates and the QA backlog",
        "transcript": "(0:01) Okay, let's go around the table.",
        "srt": "1\n00:00:01,000 --> 00:00:03,200\nOkay, let's go around the table.\n",
        "created_at": "2026-10-09 16:09:12",
    },
    {
        "filename": "Client onboarding call.mkv",
        "source_type": "video",
        "model": "gpt-4o-transcribe",
        "provider": "OpenAI API",
        "with_timestamps": False,
        "file_size_mb": 90.6,
        "duration_minutes": 47.2,
        "elapsed_seconds": 251.0,
        "cost_usd": 0.29,
        "title": "Onboarding plan and the data import",
        "transcript": _ONBOARDING,
        "created_at": "2026-10-14 12:01:37",
    },
    {
        "filename": "Voice memo - budget ideas.m4a",
        "source_type": "audio",
        "model": "small",
        "provider": "Local (offline)",
        "with_timestamps": False,
        "file_size_mb": 7.1,
        "duration_minutes": 3.68,
        "elapsed_seconds": 41.3,
        "cost_usd": 0.0002,
        "title": "Ideas for next year's budget",
        "transcript": "A few ideas for next year's budget, before I forget them.",
        "created_at": "2026-10-17 18:52:10",
    },
]


def history() -> None:
    """Add made-up older rows next to the real demo run, and turn titles on."""
    db = _isolated_db()
    for template in _ROWS:
        row = dict(template)
        created = row.pop("created_at")
        cost = {"provider": row["provider"], "cost_usd": row["cost_usd"]}
        record_id = db.add_transcription(
            **row, usage_json=json.dumps({**cost, "estimated": False})
        )
        # A demo date: the app always stamps the current time.
        with closing(db._connect()) as conn, conn:
            conn.execute(
                "UPDATE transcriptions SET created_at = ? WHERE id = ?",
                (created, record_id),
            )
    db.set_preference("title_mode", "append")
    _log(f"{len(db.list_transcriptions())} rows in the demo history")


def crop(shots: Path, target: Path) -> None:
    """Cut the empty bottom off each screenshot and save it into ``target``.

    Args:
        shots: Folder with the full-height screenshots.
        target: ``docs/screenshots``.
    """
    for name in SHOTS:
        image = Image.open(shots / name).convert("RGB")
        width, height = image.size
        main = image.crop((310, 0, width, height))  # right of the sidebar
        blank = Image.new("RGB", main.size, (255, 255, 255))
        box = ImageChops.difference(main, blank).point(lambda v: 255 * (v > 8))
        bottom = box.getbbox()
        cropped = image.crop((0, 0, width, min(height, bottom[3] + 40)))
        cropped.save(target / name, optimize=True)
        _log(f"{name}: {cropped.size[0]}×{cropped.size[1]}")


GIF_WIDTH = 960
GIF_FPS = 12


def gif(manifest: Path, target: Path) -> None:
    """Join the video frames into an MP4 and a looping GIF.

    Each frame lasts until the next one was taken, divided by the manifest's
    ``speed``; the stretches it lists as ``squeezed`` are timed apart: waits
    (the recording being prepared, the page redrawing, the transcription
    running) shortened to at most their given length, pauses to read (no
    length) at real speed. The MP4 stays next to the manifest, for uploading
    through GitHub's web editor if wanted.

    Args:
        manifest: ``shoot.mjs video``'s JSON (frame files and times).
        target: The GIF to write, e.g. ``docs/demo.gif``.
    """
    data = json.loads(manifest.read_text(encoding="utf-8"))
    frames, speed = data["frames"], data["speed"]
    squeezed = [
        (begin, end, 1.0 if longest is None else max(1.0, (end - begin) / longest))
        for begin, end, longest in data["squeezed"]
    ]
    folder = manifest.parent / "frames"
    lines = ["ffconcat version 1.0"]
    total = 0.0
    for index, frame in enumerate(frames):
        until = frames[index + 1]["t"] if index + 1 < len(frames) else data["end"]
        length = max(0.04, until - frame["t"])
        factor = next((f for b, e, f in squeezed if b <= frame["t"] < e), speed)
        length /= factor
        total += length
        picture = (folder / frame["file"]).as_posix()
        lines += [f"file '{picture}'", f"duration {length:.4f}"]
    lines.append(f"file '{(folder / frames[-1]['file']).as_posix()}'")
    listing = manifest.parent / "frames.ffconcat"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ffmpeg(
        *("-f", "concat", "-safe", "0", "-i", str(listing)),
        *("-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-crf", "20"),
        str(manifest.parent / "demo.mp4"),
    )
    palette = (
        f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    _ffmpeg(
        *("-f", "concat", "-safe", "0", "-i", str(listing)),
        *("-filter_complex", palette, "-loop", "0", str(target)),
    )
    size = target.stat().st_size / (1024 * 1024)
    _log(f"{target.name}: {total:.1f} s, {len(frames)} frames, {size:.1f} MB")


def main() -> None:
    """Run one step (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    step = steps.add_parser("copy-app")
    step.add_argument("repo", type=Path)
    step.add_argument("dest", type=Path)
    steps.add_parser("media").add_argument("out", type=Path)
    steps.add_parser("prefs").add_argument("recordings", type=Path)
    steps.add_parser("history")
    step = steps.add_parser("crop")
    step.add_argument("shots", type=Path)
    step.add_argument("target", type=Path)
    step = steps.add_parser("gif")
    step.add_argument("manifest", type=Path)
    step.add_argument("target", type=Path)
    args = parser.parse_args()
    if args.step == "copy-app":
        copy_app(args.repo, args.dest)
    elif args.step == "media":
        media(args.out)
    elif args.step == "prefs":
        prefs(args.recordings)
    elif args.step == "history":
        history()
    elif args.step == "crop":
        crop(args.shots, args.target)
    else:
        gif(args.manifest, args.target)


if __name__ == "__main__":
    main()
