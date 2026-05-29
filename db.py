"""SQLite persistence layer for transcription history.

Uses the stdlib ``sqlite3`` module with short-lived connections (one per
operation), which is safe for Streamlit's rerun model. The database lives in
``data/transcriptions.db`` and survives the app's "Clean temporary files"
action (which only clears ``temp/`` and ``uploads/``).
"""

import os
import sqlite3
from contextlib import closing

DB_PATH = os.path.join("data", "transcriptions.db")


def _connect():
    """Opens a SQLite connection with pragmas tuned for the Streamlit model."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Creates the data directory and transcriptions table if needed."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                filename         TEXT NOT NULL,
                source_type      TEXT NOT NULL,
                model            TEXT NOT NULL,
                with_timestamps  INTEGER NOT NULL,
                transcript       TEXT NOT NULL,
                srt              TEXT,
                audio_path       TEXT,
                file_size_mb     REAL,
                duration_minutes REAL,
                created_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )


def add_transcription(
    filename,
    source_type,
    model,
    with_timestamps,
    transcript,
    srt=None,
    audio_path=None,
    file_size_mb=None,
    duration_minutes=None,
):
    """Inserts a transcription record and returns its new id."""
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """
            INSERT INTO transcriptions (
                filename, source_type, model, with_timestamps,
                transcript, srt, audio_path, file_size_mb, duration_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                source_type,
                model,
                1 if with_timestamps else 0,
                transcript,
                srt,
                audio_path,
                file_size_mb,
                duration_minutes,
            ),
        )
        return cursor.lastrowid


def list_transcriptions():
    """Returns lightweight rows (no large text fields), newest first."""
    with closing(_connect()) as conn:
        return conn.execute(
            """
            SELECT id, filename, source_type, model, with_timestamps,
                   file_size_mb, duration_minutes, created_at
            FROM transcriptions
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()


def get_transcription(record_id):
    """Returns the full record for a given id, or None if not found."""
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?", (record_id,)
        ).fetchone()


def delete_transcription(record_id):
    """Deletes a single transcription record by id."""
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM transcriptions WHERE id = ?", (record_id,))
