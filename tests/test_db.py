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


def test_pragmas_are_applied():
    conn = db._connect()
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    finally:
        conn.close()
    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1
