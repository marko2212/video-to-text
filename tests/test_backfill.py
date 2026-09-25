"""The one-off Serbian Latin backfill of saved transcripts (isolated database)."""

import importlib.util
import sqlite3

import pytest

import config
import db

_SCRIPT = config.BASE_DIR / "scripts" / "serbian_latin_backfill.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("serbian_latin_backfill", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history():
    db.init_db()
    ids = {
        "mixed": db.add_transcription(
            filename="meeting.mkv",
            source_type="video",
            model="gpt-4o-transcribe",
            with_timestamps=True,
            transcript="Dobar dan. Данас причамо о буџету.",
            srt="1\n00:00:00,000 --> 00:00:02,000\nЋао\n",
        ),
        "latin": db.add_transcription(
            filename="call.amr",
            source_type="audio",
            model="gpt-4o-transcribe",
            with_timestamps=False,
            transcript="Sve je već latinicom.",
        ),
        "russian": db.add_transcription(
            filename="talk.mp4",
            source_type="video",
            model="whisper-1",
            with_timestamps=False,
            transcript="Привет, как дела?",
        ),
    }
    return config.get_settings().data_dir / "transcriptions.db", ids


def test_a_dry_run_reports_without_changing_anything(caplog):
    db_path, ids = _history()
    before = db_path.read_bytes()

    with caplog.at_level("INFO"):
        changed = _load_script().backfill(db_path, apply=False)

    assert changed == [ids["mixed"]]
    assert db_path.read_bytes() == before
    output = caplog.text
    assert "would change" in output
    assert "буџет" not in output  # never prints transcript text


def test_apply_backs_up_first_and_fixes_only_serbian_rows():
    db_path, ids = _history()

    _load_script().backfill(db_path, apply=True)

    fixed = db.get_transcription(ids["mixed"])
    assert fixed["transcript"] == "Dobar dan. Danas pričamo o budžetu."
    assert "Ćao" in fixed["srt"]
    assert db.get_transcription(ids["russian"])["transcript"] == "Привет, как дела?"

    backups = list(db_path.parent.glob("transcriptions.db-backup-*-serbian-latin"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as copy:
        original = copy.execute(
            "SELECT transcript FROM transcriptions WHERE id = ?", (ids["mixed"],)
        ).fetchone()[0]
    assert original == "Dobar dan. Данас причамо о буџету."


def test_nothing_changes_when_the_backup_fails(monkeypatch):
    db_path, _ = _history()
    before = db_path.read_bytes()
    script = _load_script()

    def failed_backup(path):
        raise SystemExit("Backup check failed")

    monkeypatch.setattr(script, "_backup", failed_backup)
    with pytest.raises(SystemExit):
        script.backfill(db_path, apply=True)
    assert db_path.read_bytes() == before
