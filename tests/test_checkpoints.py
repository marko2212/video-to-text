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
