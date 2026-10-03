"""Tests for the on-disk checkpoints of paid work."""

import os
import time

import audio
import checkpoints
import config


def test_digest_depends_on_content_not_name(tmp_path):
    first = tmp_path / "call.wav"
    same = tmp_path / "renamed.wav"
    other = tmp_path / "other.wav"
    first.write_bytes(b"recording one")
    same.write_bytes(b"recording one")
    other.write_bytes(b"recording two")

    assert checkpoints.file_digest(first) == checkpoints.file_digest(same)
    assert checkpoints.file_digest(first) != checkpoints.file_digest(other)


def test_saved_results_round_trip_and_count(tmp_path):
    directory = checkpoints.run_dir("abc123", "transcribe", "gpt-4o-transcribe")
    assert directory.parent == config.get_settings().temp_dir / "checkpoints"
    assert checkpoints.count(directory) == 0
    assert checkpoints.load(directory, "chunk_000") is None

    checkpoints.save(directory, "chunk_000", {"text": "Dobar dan", "n": 1})
    checkpoints.save(directory, "chunk_001", {"text": "Zdravo"})

    assert checkpoints.load(directory, "chunk_000") == {"text": "Dobar dan", "n": 1}
    assert checkpoints.count(directory, "chunk_") == 2

    checkpoints.discard(directory)
    assert checkpoints.count(directory) == 0


def test_a_corrupt_checkpoint_is_ignored(tmp_path):
    directory = checkpoints.run_dir("abc123", "frames")
    directory.mkdir(parents=True)
    (directory / "frame_1.json").write_text("{not json", encoding="utf-8")

    assert checkpoints.load(directory, "frame_1") is None


def test_directory_names_are_safe(tmp_path):
    directory = checkpoints.run_dir("abc", "gpt-4o/transcribe", "10 min")
    assert directory.name == "abc_gpt-4o-transcribe_10-min"


def test_a_save_that_cannot_write_is_logged_not_raised(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    checkpoints.save(blocker, "chunk_000", {"text": "kept in memory"})
    assert checkpoints.load(blocker, "chunk_000") is None


def test_old_checkpoints_are_pruned_and_recent_ones_kept():
    old = checkpoints.run_dir("old", "transcribe")
    new = checkpoints.run_dir("new", "transcribe")
    checkpoints.save(old, "chunk_000", {"text": "old"})
    checkpoints.save(new, "chunk_000", {"text": "new"})
    long_ago = time.time() - 30 * 86_400
    os.utime(old / "chunk_000.json", (long_ago, long_ago))

    assert checkpoints.prune(max_age_days=14) == 1
    assert not old.exists()
    assert checkpoints.load(new, "chunk_000") == {"text": "new"}


def test_upload_names_cannot_leave_their_folder():
    backslash = chr(92)
    assert audio.safe_name("x/D:evil.mp4") == "D_evil.mp4"
    assert audio.safe_name(f"C:{backslash}Users{backslash}call.wav") == "call.wav"
    assert audio.safe_name("dir/../x.amr") == "x.amr"
    assert audio.safe_name("..") == "upload"
    assert audio.safe_name("call.wav") == "call.wav"


def test_a_running_transcription_is_visible_to_every_session():
    assert not checkpoints.any_active_run()
    with checkpoints.active_run():
        assert checkpoints.any_active_run()
    assert not checkpoints.any_active_run()


def test_the_run_is_released_even_when_it_is_stopped():
    class Stop(BaseException):  # like Streamlit's StopException
        pass

    try:
        with checkpoints.active_run():
            raise Stop
    except Stop:
        pass
    assert not checkpoints.any_active_run()


def test_scratch_folders_of_a_killed_run_are_pruned():
    temp_dir = config.get_settings().temp_dir
    old = temp_dir / "segments-old"
    fresh = temp_dir / "frames-fresh"
    for folder in (old, fresh):
        folder.mkdir(parents=True)
        (folder / "chunk.mp3").write_bytes(b"audio")
    two_days_ago = time.time() - 48 * 3600
    os.utime(old / "chunk.mp3", (two_days_ago, two_days_ago))
    os.utime(old, (two_days_ago, two_days_ago))

    with checkpoints.active_run():
        assert checkpoints.prune_scratch(max_age_hours=24) == 0  # never mid-run
    assert checkpoints.prune_scratch(max_age_hours=24) == 1
    assert not old.exists()
    assert (fresh / "chunk.mp3").exists()


def _aged(path, hours=48, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    then = time.time() - hours * 3600
    os.utime(path, (then, then))
    return path


def test_working_copies_a_day_old_are_removed_but_not_transcripts():
    settings = config.get_settings()
    uploads, temp = settings.upload_dir, settings.temp_dir
    old_upload = _aged(uploads / "Sprint planning.mkv")
    old_call = _aged(uploads / "phone call.amr")
    old_wav = _aged(temp / "Sprint planning.wav")
    old_preview = _aged(temp / "Sprint planning_preview.MP3")
    # Kept: transcripts (the owner's call, for later), saved parts of an
    # unfinished run (their own 14-day rule), and anything recent.
    old_text = _aged(temp / "transcript_Sprint planning.txt")
    old_subtitles = _aged(temp / "transcript_Sprint planning.srt")
    old_part = _aged(temp / "checkpoints" / "abc-gpt" / "chunk_000.json")
    fresh_upload = _aged(uploads / "today.mkv", hours=2)
    fresh_wav = _aged(temp / "today.wav", hours=2)

    removed = checkpoints.prune_working_copies(max_age_hours=24)

    assert removed == 4
    for gone in (old_upload, old_call, old_wav, old_preview):
        assert not gone.exists(), gone
    for kept in (old_text, old_subtitles, old_part, fresh_upload, fresh_wav):
        assert kept.exists(), kept


def test_working_copies_stay_while_a_run_is_active():
    old_wav = _aged(config.get_settings().temp_dir / "meeting.wav")

    with checkpoints.active_run():
        assert checkpoints.prune_working_copies(max_age_hours=24) == 0

    assert old_wav.exists()


def test_the_history_is_never_a_working_copy(monkeypatch, tmp_path):
    # DATA_DIR may point at the uploads folder: the database must survive, so
    # nothing in that folder is treated as a working copy.
    shared = tmp_path / "shared"
    monkeypatch.setenv("DATA_DIR", str(shared))
    monkeypatch.setenv("UPLOAD_DIR", str(shared))
    config.get_settings.cache_clear()
    database = _aged(shared / "transcriptions.db")
    upload = _aged(shared / "call.amr")

    checkpoints.prune_working_copies(max_age_hours=24)

    assert database.exists()
    # Everything in a folder the history lives in is left alone.
    assert upload.exists()


def test_a_copy_that_cannot_be_deleted_is_skipped_not_fatal(monkeypatch):
    temp = config.get_settings().temp_dir
    held = _aged(temp / "held open.wav")
    other = _aged(temp / "other.wav")
    real_unlink = type(held).unlink

    def unlink(self, missing_ok=False):
        if self.name == held.name:
            raise PermissionError(13, "The file is being used by another process")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(held), "unlink", unlink)

    assert checkpoints.prune_working_copies(max_age_hours=24) == 1
    assert held.exists()
    assert not other.exists()


def test_the_working_files_size_is_what_clean_up_would_free(monkeypatch, tmp_path):
    settings = config.get_settings()
    _aged(settings.upload_dir / "call.amr", content=b"a" * 1_000)
    _aged(settings.temp_dir / "call.wav", content=b"b" * 300)
    part = settings.temp_dir / "checkpoints" / "abc-gpt" / "chunk_000.json"
    _aged(part, content=b"c" * 20)

    assert checkpoints.working_files_size() == 1_320

    # The history and the model folder are left alone by the button, so they
    # are not counted either — even inside temp/.
    monkeypatch.setenv("DATA_DIR", str(settings.temp_dir / "data"))
    config.get_settings.cache_clear()
    _aged(config.get_settings().data_dir / "transcriptions.db", content=b"d" * 5_000)

    assert checkpoints.working_files_size() == 1_320


def test_a_folder_named_twice_is_counted_once(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    monkeypatch.setenv("TEMP_DIR", str(shared))
    monkeypatch.setenv("UPLOAD_DIR", str(shared))
    config.get_settings.cache_clear()
    _aged(shared / "call.amr", content=b"a" * 700)

    assert checkpoints.working_files_size() == 700
