"""One run without a page: ``runs.run`` with fake engines and a plain sink."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import audio
import checkpoints
import config
import db
import runs
import titles
import transcribe
from exceptions import (
    AudioProcessingError,
    IncompleteTranscriptionError,
    TranscriptionError,
)

_SPENT = {
    "requests": 1,
    "seconds": 60.0,
    "input_tokens": 100,
    "output_tokens": 20,
    "cost_usd": 0.01,
    "estimated": False,
    "audio_seconds": 60.0,
    "chunks": [],
}


def _sink() -> SimpleNamespace:
    return SimpleNamespace(
        transcript_path=None,
        srt_path=None,
        elapsed_seconds=None,
        cost=None,
        partial="left by an earlier run",
        title=None,
        row_id=None,
        notices=None,
    )


def _request(tmp_path, **changes) -> runs.RunRequest:
    recording = tmp_path / "talk.wav"
    recording.write_bytes(b"talk audio")
    fields = {
        "filename": "talk.wav",
        "source_type": "audio",
        "audio_path": recording,
        "transcript_path": tmp_path / "transcript_talk.txt",
        "srt_path": tmp_path / "transcript_talk.srt",
        "provider": config.PROVIDER_OPENAI,
        "model": config.DEFAULT_MODEL,
        "api_key": "test-key",
    }
    return runs.RunRequest(**{**fields, **changes})


@pytest.fixture
def engine(monkeypatch):
    """Replace the OpenAI pipeline: it saves one part and writes a transcript."""
    saved = []

    def fake(input_file, output_file, api_key, **kwargs):
        digest = checkpoints.file_digest(Path(input_file))
        parts = transcribe.checkpoint_dir(digest, kwargs["model"])
        parts.mkdir(parents=True, exist_ok=True)
        (parts / "chunk_000.json").write_text("{}", encoding="utf-8")
        saved.append(parts)
        Path(output_file).write_text("Transcript.", encoding="utf-8")
        return dict(_SPENT)

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    db.init_db()
    return saved


def test_a_finished_run_is_saved_and_its_parts_move_beside_the_history(
    tmp_path, engine
):
    sink = _sink()

    runs.run(_request(tmp_path), sink)

    [row] = db.list_transcriptions()
    assert sink.row_id == row["id"]
    assert row["filename"] == "talk.wav"
    assert db.get_transcription(row["id"])["transcript"] == "Transcript."
    assert row["cost_usd"] == pytest.approx(0.01)
    assert sink.cost["transcription"]["requests"] == 1
    assert sink.transcript_path.read_text(encoding="utf-8") == "Transcript."
    assert sink.partial is None
    assert sink.notices == []
    [parts] = engine
    assert not parts.exists()
    assert (config.get_settings().runs_dir / str(row["id"])).is_dir()


def test_nothing_is_set_between_the_history_row_and_the_discard(
    tmp_path, engine, monkeypatch
):
    # For the page every result set is a Streamlit call, where a Stop takes
    # effect: one between the row and the discard would leave the saved parts
    # of a saved transcript, and the next Start would save it a second time.
    events = []

    class Sink:
        def __setattr__(self, name, value):
            events.append(name)

    real_add, real_discard = db.add_transcription, checkpoints.discard
    monkeypatch.setattr(
        db, "add_transcription", lambda **row: events.append("row") or real_add(**row)
    )
    monkeypatch.setattr(
        checkpoints,
        "discard",
        lambda folder: events.append("discard") or real_discard(folder),
    )

    runs.run(_request(tmp_path), Sink())

    assert events[events.index("row") :] == ["row", "discard", "row_id", "notices"]
    # And everything the page shows is set before the row, where a Stop still
    # leaves the parts for a free resume.
    assert {"transcript_path", "elapsed_seconds", "cost"} <= set(
        events[: events.index("row")]
    )


def test_without_a_key_nothing_is_sent(tmp_path, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("the engine was called without a key")

    monkeypatch.setattr(transcribe, "transcribe_openai", never)
    sink = _sink()

    runs.run(_request(tmp_path, api_key=None), sink)

    assert sink.notices == [("error", runs.NO_KEY_NOTICE)]
    assert sink.row_id is None


def test_a_run_that_stops_part_way_is_reported_not_saved(tmp_path, monkeypatch):
    def stops(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("First part. ⚠️ stopped", encoding="utf-8")
        raise IncompleteTranscriptionError("no credit left.", completed=1, total=3)

    monkeypatch.setattr(transcribe, "transcribe_openai", stops)
    db.init_db()
    sink = _sink()

    runs.run(_request(tmp_path), sink)

    assert sink.partial == "1 of 3"
    assert sink.row_id is None
    assert sink.cost["partial"] is True
    assert sink.transcript_path.exists()
    assert db.list_transcriptions() == []
    levels = [level for level, _ in sink.notices]
    assert levels == ["error", "info"]
    assert "Stopped after part 1 of 3: no credit left." in sink.notices[0][1]
    assert "No finished parts are saved" in sink.notices[1][1]


def test_a_failed_run_leaves_an_error_and_no_row(tmp_path, monkeypatch):
    def fails(*args, **kwargs):
        raise TranscriptionError("the file could not be read.")

    def never_asked() -> tuple[str, str]:
        raise AssertionError("the title setting was read for a failed run")

    monkeypatch.setattr(transcribe, "transcribe_openai", fails)
    db.init_db()
    sink = _sink()

    runs.run(_request(tmp_path), sink, runs.RunHooks(title_setting=never_asked))

    assert sink.notices == [
        ("error", "Transcription error: the file could not be read.")
    ]
    assert sink.row_id is None
    assert sink.elapsed_seconds is None
    assert db.list_transcriptions() == []


def test_the_title_is_written_after_the_transcript_and_saved_with_the_row(
    tmp_path, engine, monkeypatch
):
    record = {
        "kind": "title",
        "model": "gpt-5.4-nano",
        "input_tokens": 3_000,
        "output_tokens": 5,
        "cost_usd": 0.001,
        "estimated": False,
    }
    asked = []

    def make_title(transcript, api_key, model):
        asked.append((transcript, model))
        return "Weekly planning", [record]

    monkeypatch.setattr(titles, "make_title", make_title)
    sink = _sink()
    hooks = runs.RunHooks(
        title_setting=lambda: (config.TITLE_MODE_APPEND, "gpt-5.4-nano")
    )

    runs.run(_request(tmp_path), sink, hooks)

    assert asked == [("Transcript.", "gpt-5.4-nano")]
    [row] = db.list_transcriptions()
    assert row["title"] == sink.title == "Weekly planning"
    assert sink.cost["cost_usd"] == pytest.approx(0.011)


def test_a_local_run_needs_no_key_and_costs_nothing(tmp_path, monkeypatch):
    loaded = []

    def fake_local(input_file, output_file, model, **kwargs):
        assert model == "model:small"
        assert kwargs["batched"] is True
        Path(output_file).write_text("Local transcript.", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_local", fake_local)
    db.init_db()
    sink = _sink()
    hooks = runs.RunHooks(
        load_local_model=lambda name: loaded.append(name) or f"model:{name}"
    )
    request = _request(
        tmp_path,
        provider=config.PROVIDER_LOCAL,
        model="small",
        api_key=None,
        batched=True,
    )

    runs.run(request, sink, hooks)

    assert loaded == ["small"]
    [row] = db.list_transcriptions()
    assert row["cost_usd"] == 0
    assert sink.cost["local_batched"] is True
    assert sink.cost["provider"] == config.PROVIDER_LOCAL


def test_a_file_is_hashed_once_per_run(tmp_path, engine):
    hashed = []

    def digest(path: Path) -> str:
        hashed.append(path)
        return checkpoints.file_digest(path)

    request = _request(tmp_path)

    runs.run(request, _sink(), runs.RunHooks(digest=digest))

    assert hashed == [request.audio_path]


def test_the_estimate_prices_minutes_and_screenshots():
    openai = config.PROVIDER_OPENAI
    assert runs.estimate_cost(600, openai, "gpt-4o-transcribe") == pytest.approx(0.06)
    assert runs.estimate_cost(600, openai, "whisper-1") == pytest.approx(0.06)
    assert runs.estimate_cost(600, config.PROVIDER_LOCAL, "base") == 0
    assert runs.estimate_cost(None, openai, "gpt-4o-transcribe") is None

    visual = {"model": "gpt-5.4-nano", "detail": "low", "interval": 30.0}
    with_screens = runs.estimate_cost(600, config.PROVIDER_LOCAL, "base", visual)
    # 21 looks in ten minutes; a local run pays for nothing else.
    assert with_screens == pytest.approx(
        21 * (2519 * 0.20 + config.VISION_OUTPUT_TOKENS_PER_FRAME * 1.25) / 1_000_000
    )


def test_an_audio_file_is_used_where_it_lies_and_a_video_is_extracted(
    tmp_path, monkeypatch
):
    call = tmp_path / "phone_20261006-091106.amr"
    video = tmp_path / "2026-11-05 14-03-25.mkv"
    notes = tmp_path / "notes.txt"
    for path in (call, video, notes):
        path.write_bytes(b"x")
    wav = tmp_path / "work" / "meeting.wav"
    monkeypatch.setattr(audio, "to_wav", lambda source, target: target)

    assert runs.prepare_source(call, wav) == (call, None, "audio")
    assert runs.prepare_source(video, wav) == (wav, video, "video")
    with pytest.raises(AudioProcessingError, match="not a supported"):
        runs.prepare_source(notes, wav)


def test_a_transcript_is_named_after_its_recording(tmp_path):
    recording = tmp_path / "2026-11-05 14-03-25 - daily - šema posmatranja.mkv"

    assert runs.sibling_transcript(recording) == (
        tmp_path / "transcript_2026-11-05 14-03-25 - daily - šema posmatranja.txt"
    )
    assert runs.sibling_transcript(recording, tmp_path / "out").parent == (
        tmp_path / "out"
    )


def test_the_saved_parts_are_counted_not_promised(tmp_path, engine):
    request = _request(tmp_path)
    digest = checkpoints.file_digest(request.audio_path)
    parts = transcribe.checkpoint_dir(digest, request.model)

    nothing = runs.saved_parts_hint(request.audio_path, request.model, 3)
    assert nothing.startswith("No finished parts are saved")

    parts.mkdir(parents=True)
    (parts / "chunk_000.json").write_text("{}", encoding="utf-8")
    some = runs.saved_parts_hint(
        request.audio_path, request.model, 3, retry="press Start", retry_verb="Start"
    )
    assert some.startswith("1 of 3 part(s) are saved — press Start to transcribe")


# --- the parts an OpenAI run leaves incomplete, filled by local Whisper ------------


def test_the_local_fill_is_on_with_the_backend_and_off_with_a_whole_check(
    monkeypatch,
):
    from pydantic import ValidationError

    assert runs.fill_model(local_available=True) == "medium"
    assert runs.fill_model(local_available=False) is None
    # A check of the whole recording fills those parts already.
    assert runs.fill_model(local_available=True, verify="base") is None

    monkeypatch.setenv("LOCAL_FILL_MODEL", "small")
    config.get_settings.cache_clear()
    assert runs.fill_model(local_available=True) == "small"

    monkeypatch.setenv("LOCAL_FILL", "false")
    config.get_settings.cache_clear()
    assert runs.fill_model(local_available=True) is None

    monkeypatch.setenv("LOCAL_FILL_MODEL", "huge")
    config.get_settings.cache_clear()
    with pytest.raises(ValidationError, match="LOCAL_FILL_MODEL must be one of"):
        config.get_settings()


def test_the_pipeline_gets_the_local_model_to_load_only_when_needed(
    tmp_path, monkeypatch
):
    offered = []

    def fake(input_file, output_file, api_key, **kwargs):
        offered.append(kwargs["fill"])
        Path(output_file).write_text("Transcript.", encoding="utf-8")
        return dict(_SPENT)

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    db.init_db()
    loaded = []
    hooks = runs.RunHooks(
        load_local_model=lambda name: loaded.append(name) or f"model:{name}"
    )

    runs.run(_request(tmp_path, local_available=True), _sink(), hooks)

    load, name = offered[-1]
    assert name == config.DEFAULT_LOCAL_FILL_MODEL
    assert loaded == []  # not before the pipeline asks for it
    assert load() == "model:medium"

    runs.run(_request(tmp_path, local_available=False), _sink(), hooks)
    runs.run(_request(tmp_path, local_available=True, verify="base"), _sink(), hooks)
    assert offered[1:] == [None, None]


def _checked(part, start, end, rate, **check):
    return {
        "index": int(part[:3]),
        "part": part,
        "start": start,
        "end": end,
        "words_per_min": rate,
        "typical_words_per_min": 160,
        "thin": True,
        "capped": False,
        "local_check": {"model": "medium", **check},
    }


def test_the_notices_say_what_local_whisper_filled_and_what_still_lacks_speech():
    chunks = [
        _checked("004a", 1200.5, 1349.1, 9.5, local_words=330, added_words=312),
        _checked("005a", 1505.4, 1654.6, 3.6, local_words=12, added_words=0),
    ]
    filled = {
        "model": "medium",
        "seconds": 192.0,
        "parts": 2,
        "filled": 1,
        "added_words": 312,
        "failed": 0,
    }

    notices = runs.run_warnings({"chunks": chunks, "filled": filled})

    assert notices == [
        (
            "warning",
            "Speech may be missing in 1 part(s) — 25:05–27:34 (4 words/min against "
            "about 160, local Whisper (medium) heard 12 words there, nothing more). "
            "The transcript marks them ⚠️.",
        ),
        (
            "info",
            "Local Whisper (medium) filled 1 of 2 part(s) that came back incomplete "
            "from OpenAI, in 3:12: 20:00–22:29 (312 words), marked 🧩.",
        ),
    ]
    cost = runs.run_cost(config.PROVIDER_OPENAI, {**_SPENT, "filled": filled}, [])
    assert "2 incomplete part(s) checked with local Whisper (medium, free)" in (
        runs.describe_cost(cost)
    )


def test_a_local_check_that_could_not_run_is_a_warning_and_the_marks_stay():
    error = {"error": "no space left for the model"}
    chunks = [
        _checked("004a", 1200.5, 1349.1, 9.5, **error),
        _checked("005a", 1505.4, 1654.6, 3.6, **error),
    ]
    filled = {
        "model": "medium",
        "seconds": 0.4,
        "parts": 2,
        "filled": 0,
        "added_words": 0,
        "failed": 2,
        "error": "no space left for the model",
    }

    notices = runs.run_warnings({"chunks": chunks, "filled": filled})

    assert [level for level, _ in notices] == ["warning", "warning"]
    assert notices[0][1].startswith("Speech may be missing in 2 part(s) — ")
    assert notices[1][1] == (
        "Local Whisper (medium) could not check 2 of the 2 part(s) that came back "
        "incomplete: no space left for the model They stay marked ⚠️."
    )
    cost = runs.run_cost(config.PROVIDER_OPENAI, {**_SPENT, "filled": filled}, [])
    assert "local Whisper" not in runs.describe_cost(cost)
