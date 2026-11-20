"""The command line: one JSON answer per command, the same run as the page."""

import argparse
import json
from pathlib import Path

import pytest

import cli
import config
import db
import recordings
import runs
import transcribe
from exceptions import IncompleteTranscriptionError, TranscriptionError

_NAME = "2026-11-05 14-03-25 - daily - šema.wav"


def _call(capsys, *argv: str) -> tuple[int, dict]:
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def recording(tmp_path, monkeypatch) -> Path:
    """A two-minute recording in a Meeting folder (its length is not probed)."""
    folder = tmp_path / "Meeting"
    folder.mkdir()
    path = folder / _NAME
    path.write_bytes(b"daily audio")
    monkeypatch.setattr(recordings, "duration", lambda path: 120.0)
    return path


@pytest.fixture
def engine(monkeypatch) -> list[dict]:
    """Replace the OpenAI pipeline; each call writes a transcript of the input."""
    calls = []

    def fake(input_file, output_file, api_key, **kwargs):
        calls.append(kwargs)
        kwargs["progress_callback"]({"status": "progress", "message": "part 1 of 1"})
        text = f"Transcript of {Path(input_file).read_bytes().decode()}"
        Path(output_file).write_text(text, encoding="utf-8")
        if kwargs["srt_output_file"]:
            Path(kwargs["srt_output_file"]).write_text("1\n", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    return calls


def _work_folders() -> list[Path]:
    return list(config.get_settings().temp_dir.glob("cli-*"))


def test_transcribe_puts_the_text_beside_the_recording_and_into_history(
    capsys, recording, engine
):
    code, answer = _call(capsys, "transcribe", str(recording))

    target = recording.parent / "transcript_2026-11-05 14-03-25 - daily - šema.txt"
    assert code == cli.EXIT_DONE
    assert answer["status"] == "done"
    assert Path(answer["transcript"]) == target
    assert target.read_text(encoding="utf-8") == "Transcript of daily audio"
    assert answer["estimated_cost_usd"] == pytest.approx(0.012)
    assert (answer["engine"], answer["model"]) == ("openai", config.DEFAULT_MODEL)
    [row] = db.list_transcriptions()
    assert answer["row_id"] == row["id"]
    assert row["filename"] == _NAME
    assert _work_folders() == []


def test_progress_goes_to_standard_error_and_the_answer_is_one_object(
    capsys, recording, engine
):
    cli.main(["transcribe", str(recording)])

    captured = capsys.readouterr()
    assert "part 1 of 1" in captured.err
    assert json.loads(captured.out)["status"] == "done"


@pytest.mark.parametrize(
    ("installed", "argv", "fill"),
    [
        (True, [], config.DEFAULT_LOCAL_FILL_MODEL),
        (False, [], None),
        (True, ["--verify"], None),
    ],
)
def test_incomplete_parts_are_filled_locally_as_on_the_page(
    capsys, recording, engine, monkeypatch, installed, argv, fill
):
    monkeypatch.setattr(cli, "_local_installed", lambda: installed)
    monkeypatch.setattr(runs, "load_local_model", lambda name: f"model:{name}")

    code, _ = _call(capsys, "transcribe", str(recording), *argv)

    assert code == cli.EXIT_DONE
    offered = engine[-1]["fill"]
    assert (offered and offered[1]) == fill


def test_a_recording_with_a_transcript_is_left_alone(capsys, recording, engine):
    target = runs.sibling_transcript(recording)
    target.write_text("Corrected by hand.", encoding="utf-8")

    code, answer = _call(capsys, "transcribe", str(recording))

    assert (code, answer["status"]) == (cli.EXIT_NOT_STARTED, "exists")
    assert engine == []
    assert target.read_text(encoding="utf-8") == "Corrected by hand."

    code, answer = _call(capsys, "transcribe", str(recording), "--overwrite")

    assert (code, answer["status"]) == (cli.EXIT_DONE, "done")
    assert target.read_text(encoding="utf-8") == "Transcript of daily audio"


def test_a_run_over_the_limit_sends_nothing_until_it_is_confirmed(
    capsys, recording, engine, monkeypatch
):
    monkeypatch.setattr(recordings, "duration", lambda path: 3 * 3600.0)

    code, answer = _call(capsys, "transcribe", str(recording))

    assert (code, answer["status"]) == (cli.EXIT_NOT_STARTED, "needs_confirmation")
    assert answer["estimated_cost_usd"] == pytest.approx(1.08)
    assert answer["max_cost_usd"] == config.CLI_MAX_COST_USD
    assert engine == []
    assert db.list_transcriptions() == []

    code, answer = _call(capsys, "transcribe", str(recording), "--yes")

    assert (code, answer["status"]) == (cli.EXIT_DONE, "done")
    assert len(engine) == 1


def test_a_recording_of_unknown_length_needs_a_yes(
    capsys, recording, engine, monkeypatch
):
    monkeypatch.setattr(recordings, "duration", lambda path: None)

    code, answer = _call(capsys, "transcribe", str(recording))

    assert (code, answer["status"]) == (cli.EXIT_NOT_STARTED, "needs_confirmation")
    assert answer["estimated_cost_usd"] is None
    assert engine == []


def test_a_local_run_asks_for_no_key_and_no_confirmation(
    capsys, recording, monkeypatch
):
    def fake_local(input_file, output_file, model, **kwargs):
        Path(output_file).write_text(f"Local, {model}.", encoding="utf-8")

    monkeypatch.setattr(recordings, "duration", lambda path: 10 * 3600.0)
    monkeypatch.setattr(cli, "_local_installed", lambda: True)
    monkeypatch.setattr(runs, "load_local_model", lambda name: f"model:{name}")
    monkeypatch.setattr(transcribe, "transcribe_local", fake_local)
    monkeypatch.delenv("OPENAI_API_KEY")
    config.get_settings.cache_clear()

    code, answer = _call(
        capsys, "transcribe", str(recording), "--engine", "local", "--model", "small"
    )

    assert (code, answer["status"]) == (cli.EXIT_DONE, "done")
    assert answer["cost_usd"] == 0
    target = Path(answer["transcript"])
    assert target.read_text(encoding="utf-8") == "Local, model:small."


def test_a_run_that_stops_part_way_keeps_its_text_out_of_the_folder(
    capsys, recording, monkeypatch
):
    def stops(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("First part. ⚠️", encoding="utf-8")
        raise IncompleteTranscriptionError("no credit left.", completed=1, total=3)

    monkeypatch.setattr(transcribe, "transcribe_openai", stops)

    code, answer = _call(capsys, "transcribe", str(recording))

    assert (code, answer["status"]) == (cli.EXIT_PARTIAL, "partial")
    assert answer["parts"] == "1 of 3"
    assert (
        Path(answer["partial_transcript"])
        .read_text(encoding="utf-8")
        .startswith("First part.")
    )
    # Beside the recording a partial transcript would pass for a finished one.
    assert not runs.sibling_transcript(recording).exists()
    assert db.list_transcriptions() == []


def test_a_failed_run_says_why_in_json(capsys, recording, monkeypatch):
    def fails(*args, **kwargs):
        raise TranscriptionError("the file could not be read.")

    monkeypatch.setattr(transcribe, "transcribe_openai", fails)

    code, answer = _call(capsys, "transcribe", str(recording))

    assert (code, answer["status"]) == (cli.EXIT_FAILED, "failed")
    assert "the file could not be read." in answer["message"]
    assert _work_folders() == []
    assert not runs.sibling_transcript(recording).exists()


def test_subtitles_are_written_beside_the_transcript(capsys, recording, engine):
    code, answer = _call(
        capsys, "transcribe", str(recording), "--model", "whisper-1", "--srt"
    )

    assert code == cli.EXIT_DONE
    assert Path(answer["srt"]) == runs.sibling_transcript(recording).with_suffix(".srt")
    assert Path(answer["srt"]).exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--srt"], "whisper-1"),
        (["--model", "gpt-9"], "not a model of the openai engine"),
        (["--screens"], "--screens needs a video"),
        (["--from", "5:00"], "--from must be earlier"),
        (["--out-dir", "Z:/no/such/folder"], "There is no folder"),
    ],
)
def test_what_cannot_be_done_is_refused_before_anything_starts(
    capsys, recording, engine, argv, message
):
    code, answer = _call(capsys, "transcribe", str(recording), *argv)

    assert (code, answer["status"]) == (cli.EXIT_FAILED, "failed")
    assert message in answer["message"]
    assert engine == []


def test_a_missing_file_and_a_missing_key_are_said_plainly(
    capsys, recording, engine, monkeypatch
):
    code, answer = _call(capsys, "transcribe", str(recording.parent / "gone.mkv"))
    assert code == cli.EXIT_FAILED
    assert "There is no file" in answer["message"]

    monkeypatch.delenv("OPENAI_API_KEY")
    config.get_settings.cache_clear()
    code, answer = _call(capsys, "transcribe", str(recording))
    assert code == cli.EXIT_FAILED
    assert "No OpenAI API key" in answer["message"]
    assert engine == []


def test_a_stretch_of_the_recording_is_priced_and_sent_as_a_span(
    capsys, recording, engine
):
    code, answer = _call(
        capsys, "transcribe", str(recording), "--from", "0:30", "--to", "1:30"
    )

    assert code == cli.EXIT_DONE
    assert answer["span"] == [30, 90]
    assert answer["estimated_cost_usd"] == pytest.approx(0.006)
    assert engine[0]["span"] == (30, 90)


def test_the_title_follows_the_setting_chosen_on_the_page(
    capsys, recording, engine, monkeypatch
):
    import titles

    monkeypatch.setattr(titles, "make_title", lambda *args: ("Daily", []))
    db.init_db()
    db.set_preference("title_mode", config.TITLE_MODE_APPEND)

    _, answer = _call(capsys, "transcribe", str(recording))
    assert answer["title"] == "Daily"

    _, answer = _call(capsys, "transcribe", str(recording), "--overwrite", "--no-title")
    assert answer["title"] is None


def test_estimate_starts_nothing_and_needs_no_key(
    capsys, recording, engine, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY")
    config.get_settings.cache_clear()

    code, answer = _call(capsys, "estimate", str(recording), "--engine", "openai")

    assert (code, answer["status"]) == (cli.EXIT_DONE, "estimate")
    assert answer["duration_seconds"] == 120.0
    assert answer["estimated_cost_usd"] == pytest.approx(0.012)
    assert answer["transcript"] is None
    assert engine == []
    assert _work_folders() == []


def test_latest_and_pending_list_a_folder_newest_first(capsys, recording):
    older = recording.parent / "phone_20261006-091106.amr"
    older.write_bytes(b"call")
    (recording.parent / "notes_2026-11-05.txt").write_text("notes", encoding="utf-8")
    runs.sibling_transcript(older).write_text("Call.", encoding="utf-8")

    code, answer = _call(capsys, "latest", str(recording.parent), "--count", "5")

    assert code == cli.EXIT_DONE
    listed = {Path(entry["recording"]).name: entry for entry in answer["recordings"]}
    assert set(listed) == {_NAME, older.name}
    assert listed[older.name]["transcript"].endswith(
        "transcript_phone_20261006-091106.txt"
    )
    assert listed[_NAME]["transcript"] is None
    assert listed[_NAME]["duration_seconds"] == 120.0

    code, answer = _call(capsys, "pending", str(recording.parent))

    assert [Path(entry["recording"]).name for entry in answer["recordings"]] == [_NAME]


def test_a_folder_that_is_not_there_is_an_answer_not_a_traceback(capsys, tmp_path):
    code, answer = _call(capsys, "pending", str(tmp_path / "nowhere"))

    assert (code, answer["status"]) == (cli.EXIT_FAILED, "failed")
    assert "There is no folder" in answer["message"]


def test_times_are_read_as_seconds_or_clock():
    assert cli._seconds("90") == 90
    assert cli._seconds("1:30") == 90
    assert cli._seconds("1:02:03") == 3723
    with pytest.raises(argparse.ArgumentTypeError):
        cli._seconds("half past")


def _row(row_id: int):
    return db.get_transcription(row_id)


def test_rename_gives_a_moved_recording_its_new_name_in_history(
    capsys, recording, engine, tmp_path
):
    bare = recording.with_name("2026-11-16 13-54-36.wav")
    recording.rename(bare)
    code, answer = _call(capsys, "transcribe", str(bare))
    row_id = answer["row_id"]
    before = _row(row_id)
    project = tmp_path / "Project" / "Meeting"
    project.mkdir(parents=True)
    moved = project / "2026-11-16 13-54-36 - daily - schema.wav"
    bare.rename(moved)

    code, answer = _call(capsys, "rename", str(bare), str(moved))

    assert code == cli.EXIT_DONE
    assert answer["status"] == "renamed"
    assert (answer["from"], answer["to"]) == (bare.name, moved.name)
    assert [entry["row_id"] for entry in answer["entries"]] == [row_id]
    after = _row(row_id)
    assert after["filename"] == moved.name
    # An audio file is transcribed where it lies: its stored path follows it.
    assert before["audio_path"] == str(bare)
    assert after["audio_path"] == str(moved)
    for column in ("transcript", "title", "cost_usd", "created_at", "usage_json"):
        assert after[column] == before[column]


def test_rename_by_name_keeps_a_video_rows_extracted_audio_path(capsys):
    db.init_db()
    row_id = db.add_transcription(
        filename="2026-11-16 11-01-49.mkv",
        source_type="video",
        model="gpt-4o-transcribe",
        with_timestamps=False,
        transcript="Text.",
        audio_path="C:/temp/cli-x/2026-11-16 11-01-49.wav",
    )

    code, answer = _call(
        capsys,
        "rename",
        "2026-11-16 11-01-49.mkv",
        "2026-11-16 11-01-49 - architecture - MPN.mkv",
    )

    assert (code, answer["status"]) == (cli.EXIT_DONE, "renamed")
    row = _row(row_id)
    assert row["filename"] == "2026-11-16 11-01-49 - architecture - MPN.mkv"
    assert row["audio_path"] == "C:/temp/cli-x/2026-11-16 11-01-49.wav"


def _add(name: str) -> int:
    db.init_db()
    return db.add_transcription(
        filename=name,
        source_type="video",
        model="gpt-4o-transcribe",
        with_timestamps=False,
        transcript=f"Text of {name}.",
    )


def test_every_run_of_the_recording_is_renamed_unless_one_row_is_named(capsys):
    first, second = _add("a.mkv"), _add("a.mkv")
    other = _add("b.mkv")

    code, answer = _call(capsys, "rename", "a.mkv", "a - one.mkv", "--id", str(first))

    assert code == cli.EXIT_DONE
    assert [entry["row_id"] for entry in answer["entries"]] == [first]
    assert [_row(i)["filename"] for i in (first, second, other)] == [
        "a - one.mkv",
        "a.mkv",
        "b.mkv",
    ]

    code, answer = _call(capsys, "rename", "a.mkv", "a - two.mkv")

    assert [entry["row_id"] for entry in answer["entries"]] == [second]
    assert _row(second)["filename"] == "a - two.mkv"

    code, answer = _call(capsys, "rename", "b.mkv", "b - x.mkv", "--id", str(first))

    assert (code, answer["status"]) == (cli.EXIT_FAILED, "failed")
    assert f"no entry {first} named b.mkv" in answer["message"]
    assert _row(other)["filename"] == "b.mkv"


def test_a_dry_run_and_an_unknown_name_change_nothing(capsys):
    row_id = _add("a.mkv")

    code, answer = _call(capsys, "rename", "a.mkv", "a - one.mkv", "--dry-run")

    assert (code, answer["status"]) == (cli.EXIT_DONE, "dry_run")
    assert [entry["row_id"] for entry in answer["entries"]] == [row_id]
    assert _row(row_id)["filename"] == "a.mkv"

    code, answer = _call(capsys, "rename", "c.mkv", "c - one.mkv")

    assert (code, answer["status"]) == (cli.EXIT_NOT_STARTED, "not_found")
    assert answer["entries"] == []


@pytest.mark.parametrize(
    ("new", "reason"),
    [
        ("transcript_a - one.txt", "not a video or audio file name"),
        ("a.mkv", "are the same"),
        ("{missing}", "There is no file"),
    ],
)
def test_a_rename_that_looks_like_a_mistake_is_refused(capsys, tmp_path, new, reason):
    row_id = _add("a.mkv")
    new = new.replace("{missing}", str(tmp_path / "Meeting" / "a - one.mkv"))

    code, answer = _call(capsys, "rename", "a.mkv", new)

    assert (code, answer["status"]) == (cli.EXIT_FAILED, "failed")
    assert reason in answer["message"]
    assert _row(row_id)["filename"] == "a.mkv"


def test_history_lists_entries_newest_first_and_filters_by_name(capsys):
    old = _add("2026-11-13 14-12-54.mkv")
    new = _add("2026-11-16 13-01-12.mkv")
    db.set_title(new, "API engine", None, None)

    code, answer = _call(capsys, "history")

    assert (code, answer["status"]) == (cli.EXIT_DONE, "listed")
    assert [entry["row_id"] for entry in answer["entries"]] == [new, old]
    assert answer["entries"][0]["recording"] == "2026-11-16 13-01-12.mkv"
    assert answer["entries"][0]["title"] == "API engine"
    assert "transcript" not in answer["entries"][0]

    code, answer = _call(capsys, "history", "--name", "2026-11-13")

    assert [entry["row_id"] for entry in answer["entries"]] == [old]

    code, answer = _call(capsys, "history", "--count", "1")

    assert [entry["row_id"] for entry in answer["entries"]] == [new]
