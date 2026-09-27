"""Rewrite Serbian Cyrillic in saved transcripts as Latin (opt-in, one-off).

New transcripts are transliterated only when SERBIAN_LATIN=true (off by
default). This converts transcripts already saved, where a meeting switched
script from chunk to chunk.

    uv run python scripts/serbian_latin_backfill.py           # report only
    uv run python scripts/serbian_latin_backfill.py --apply   # back up, then fix

Only rows that are recognisably Serbian are touched (see ``serbian.py``), and
only their transcript and subtitles. ``--apply`` first copies the database with
SQLite's backup API next to it and checks the copy, and changes nothing if that
fails. Prints ids and letter counts, never transcript text.
"""

import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

# Run from the repo root or from scripts/: the app modules live one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serbian
from config import get_settings
from logger import get_logger

logger = get_logger("backfill")


def _cyrillic_letters(text: str | None) -> int:
    """Count Cyrillic letters in a text (0 for ``None``)."""
    return sum(1 for char in text or "" if "\u0400" <= char <= "\u04ff")


def _backup(db_path: Path) -> Path:
    """Copy the database with SQLite's backup API and verify the copy.

    Args:
        db_path: The live database.

    Returns:
        Path to the verified copy.

    Raises:
        SystemExit: If the copy is incomplete or damaged.
    """
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    target = db_path.with_name(f"{db_path.name}-backup-{stamp}-serbian-latin")
    with (
        closing(sqlite3.connect(db_path)) as source,
        closing(sqlite3.connect(target)) as copy,
    ):
        source.backup(copy)
        rows = source.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
        copied = copy.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
        integrity = copy.execute("PRAGMA integrity_check").fetchone()[0]
    if rows != copied or integrity != "ok":
        raise SystemExit(f"Backup check failed ({copied}/{rows} rows, {integrity})")
    return target


def backfill(db_path: Path, apply: bool) -> list[int]:
    """Find, and with ``apply`` fix, saved transcripts with Serbian Cyrillic.

    Args:
        db_path: The history database.
        apply: Change the rows (after a verified backup) instead of reporting.

    Returns:
        The ids of the rows that need (or got) the change.
    """
    # Read-only until a change is actually requested and backed up.
    readonly = f"{db_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(readonly, uri=True)) as conn:
        rows = conn.execute("SELECT id, transcript, srt FROM transcriptions").fetchall()
        changes = []
        for record_id, transcript, srt in rows:
            if not serbian.is_serbian_cyrillic(transcript or ""):
                continue
            new_transcript = serbian.cyrillic_to_latin(transcript)
            new_srt = serbian.cyrillic_to_latin(srt) if srt else srt
            changes.append((record_id, transcript, new_transcript, new_srt))
            logger.info(
                "id %d: %d Cyrillic letters -> %d",
                record_id,
                _cyrillic_letters(transcript),
                _cyrillic_letters(new_transcript),
            )

    if not changes:
        logger.info("Nothing to change.")
        return []
    if not apply:
        logger.info(
            "%d row(s) would change. Run with --apply to fix them.", len(changes)
        )
        return [record_id for record_id, *_ in changes]

    backup = _backup(db_path)
    logger.info("Backup: %s", backup)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        for record_id, _, new_transcript, new_srt in changes:
            conn.execute(
                "UPDATE transcriptions SET transcript = ?, srt = ? WHERE id = ?",
                (new_transcript, new_srt, record_id),
            )
    logger.info("%d row(s) rewritten in Latin script.", len(changes))
    return [record_id for record_id, *_ in changes]


def main() -> None:
    """Parse arguments and run the backfill on the configured database."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="back up the database, then fix rows"
    )
    args = parser.parse_args()
    db_path = get_settings().data_dir / "transcriptions.db"
    if not db_path.exists():
        raise SystemExit(f"No database at {db_path}")
    backfill(db_path, apply=args.apply)


if __name__ == "__main__":
    main()
