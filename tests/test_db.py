"""Tests for the SQLite history layer."""

import db


def test_add_and_get_roundtrip():
    db.init_db()
    record_id = db.add_transcription(
        filename="call.mp3",
        source_type="audio",
        model="whisper-1",
        with_timestamps=True,
        transcript="hello world",
        srt="1\n00:00:00,000 --> 00:00:01,000\nhello",
        file_size_mb=1.23,
    )

    record = db.get_transcription(record_id)
    assert record is not None
    assert record["transcript"] == "hello world"
    assert record["srt"].startswith("1")
    assert record["with_timestamps"] == 1  # bool stored as int
    assert record["file_size_mb"] == 1.23
    assert record["created_at"]  # auto timestamp


def test_list_is_newest_first_and_lightweight():
    db.init_db()
    first = db.add_transcription("a.mp3", "audio", "whisper-1", False, "first")
    second = db.add_transcription(
        "b.mkv", "video", "gpt-4o-transcribe", False, "second"
    )

    rows = db.list_transcriptions()
    # `in` on a sqlite3.Row checks values, so compare against the key list.
    columns = rows[0].keys()
    assert [r["id"] for r in rows] == [second, first]  # DESC order
    assert "transcript" not in columns  # large columns omitted


def test_delete_removes_record():
    db.init_db()
    record_id = db.add_transcription("x.wav", "audio", "whisper-1", False, "text")

    db.delete_transcription(record_id)

    assert db.get_transcription(record_id) is None
    assert db.list_transcriptions() == []


def test_elapsed_seconds_roundtrips():
    db.init_db()
    record_id = db.add_transcription(
        filename="meeting.mkv",
        source_type="video",
        model="whisper-1",
        with_timestamps=False,
        transcript="text",
        elapsed_seconds=154.5,
    )

    assert db.get_transcription(record_id)["elapsed_seconds"] == 154.5
    # The history list renders it, so it must survive the lightweight query too.
    assert db.list_transcriptions()[0]["elapsed_seconds"] == 154.5


def test_init_db_migrates_a_database_without_the_newer_columns():
    # Recreate the original schema: no `provider`, no `elapsed_seconds`.
    conn = db._connect()
    try:
        conn.execute(
            """
            CREATE TABLE transcriptions (
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
        conn.execute(
            "INSERT INTO transcriptions "
            "(filename, source_type, model, with_timestamps, transcript) "
            "VALUES ('old.mp3', 'audio', 'whisper-1', 0, 'existing history')"
        )
        conn.commit()
    finally:
        conn.close()

    db.init_db()

    rows = db.list_transcriptions()
    assert len(rows) == 1, "pre-existing history must survive the migration"
    assert rows[0]["elapsed_seconds"] is None
    assert db.get_transcription(rows[0]["id"])["transcript"] == "existing history"
    # And the migrated database still accepts new rows with the new columns.
    new_id = db.add_transcription(
        "new.mp3", "audio", "whisper-1", False, "fresh", elapsed_seconds=12.0
    )
    assert db.get_transcription(new_id)["elapsed_seconds"] == 12.0


def test_pragmas_are_applied():
    conn = db._connect()
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    finally:
        conn.close()
    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1
