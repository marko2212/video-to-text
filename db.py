"""SQLite persistence layer for transcription history.

Uses the stdlib ``sqlite3`` module with short-lived connections (one per
operation), which is safe for Streamlit's rerun model. The database lives in the
configured ``data_dir`` and survives the app's "Clean temporary files" action
(which only clears ``temp/`` and ``uploads/``).
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from config import get_settings

_DB_FILENAME = "transcriptions.db"


def _db_path() -> Path:
    """Return the path to the SQLite database file."""
    return get_settings().data_dir / _DB_FILENAME


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection with pragmas tuned for the Streamlit model.

    Returns:
        A connection with ``Row`` row factory and WAL/timeout/foreign-key pragmas.
    """
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create the data directory and ``transcriptions`` table if needed."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                filename         TEXT NOT NULL,
                source_type      TEXT NOT NULL,
                model            TEXT NOT NULL,
                provider         TEXT,
                with_timestamps  INTEGER NOT NULL,
                transcript       TEXT NOT NULL,
                srt              TEXT,
                audio_path       TEXT,
                file_size_mb     REAL,
                duration_minutes REAL,
                elapsed_seconds  REAL,
                created_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        # Add columns introduced after a database may already have been created.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(transcriptions)")}
        for name, column_type in (("provider", "TEXT"), ("elapsed_seconds", "REAL")):
            if name not in columns:
                # Both values are literals from the tuple above, never user input.
                conn.execute(
                    f"ALTER TABLE transcriptions ADD COLUMN {name} {column_type}"
                )


def add_transcription(
    filename: str,
    source_type: str,
    model: str,
    with_timestamps: bool,
    transcript: str,
    srt: str | None = None,
    audio_path: str | None = None,
    file_size_mb: float | None = None,
    duration_minutes: float | None = None,
    provider: str | None = None,
    elapsed_seconds: float | None = None,
) -> int:
    """Insert a transcription record.

    Args:
        filename: Original uploaded file name.
        source_type: Either ``"audio"`` or ``"video"``.
        model: Transcription model used.
        with_timestamps: Whether timestamps/subtitles were generated.
        transcript: The full transcript text.
        srt: Optional SRT subtitle text.
        audio_path: Optional best-effort path to the source audio.
        file_size_mb: Optional audio size in megabytes.
        duration_minutes: Optional audio duration in minutes.
        provider: Engine used ("OpenAI API" or "Local (offline)").
        elapsed_seconds: Optional wall-clock time the run took.

    Returns:
        The id of the newly inserted row.
    """
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """
            INSERT INTO transcriptions (
                filename, source_type, model, provider, with_timestamps,
                transcript, srt, audio_path, file_size_mb, duration_minutes,
                elapsed_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                source_type,
                model,
                provider,
                int(with_timestamps),
                transcript,
                srt,
                audio_path,
                file_size_mb,
                duration_minutes,
                elapsed_seconds,
            ),
        )
        return cursor.lastrowid


def list_transcriptions() -> list[sqlite3.Row]:
    """Return lightweight rows (no large text fields), newest first.

    Returns:
        Rows with metadata columns, ordered by ``created_at`` descending.
    """
    with closing(_connect()) as conn:
        return conn.execute(
            """
            SELECT id, filename, source_type, model, provider, with_timestamps,
                   file_size_mb, duration_minutes, elapsed_seconds, created_at
            FROM transcriptions
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()


def get_transcription(record_id: int) -> sqlite3.Row | None:
    """Return the full record for a given id.

    Args:
        record_id: The transcription id.

    Returns:
        The full row, or ``None`` if no record matches.
    """
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?", (record_id,)
        ).fetchone()


def delete_transcription(record_id: int) -> None:
    """Delete a single transcription record.

    Args:
        record_id: The transcription id to delete.
    """
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM transcriptions WHERE id = ?", (record_id,))
