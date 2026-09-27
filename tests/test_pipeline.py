"""The OpenAI transcription pipeline end to end, with a fake client and fake audio.

No network and no ffmpeg: the recording is generated with pydub, chunk export
writes a small marker file instead of an MP3, and the client is a stand-in that
answers per chunk.
"""

from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import checkpoints
import config
import openai_api
import transcribe
from exceptions import (
    IncompleteTranscriptionError,
    OpenAIAccountError,
    TranscriptionError,
)

MINUTE = 60_000


_SECOND = Sine(440, sample_rate=16000, bit_depth=16).to_audio_segment(duration=1000)


def _tone(ms: int) -> AudioSegment:
    # One generated second repeated: generating minutes sample by sample is slow.
    return (_SECOND * (ms // 1000 + 1))[:ms]


def _fake_export(segment, temp_folder: Path, name: str) -> Path:
    path = temp_folder / f"{name}.mp3"
    path.write_text(f"{name}|{len(segment)}", encoding="utf-8")
    return path


def _rate_limit_error(code: str) -> openai.RateLimitError:
    body = {"message": "no", "type": code, "code": code}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.openai.com/v1/x"), json=body
    )
    return openai.RateLimitError("Error code: 429", response=response, body=body)


def _server_error() -> openai.InternalServerError:
    response = httpx.Response(
        500, request=httpx.Request("POST", "https://api.openai.com/v1/x")
    )
    return openai.InternalServerError(
        "Error code: 500", response=response, body={"message": "boom"}
    )


class FakeClient:
    """Answers each chunk with ``answer(name, length_ms)`` and records the calls."""

    def __init__(self, answer):
        self.calls: list[tuple[str, int]] = []
        self._answer = answer
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

    def create(self, model, file, response_format=None):
        name, length = file.read().decode().split("|")
        self.calls.append((name, int(length)))
        return self._answer(name, int(length))

    def close(self):
        pass


def _text(text: str, tokens: int = 100, segments=None):
    return SimpleNamespace(
        text=text,
        usage={"input_tokens": 1_000, "output_tokens": tokens},
        segments=segments or [],
    )


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Run transcribe_openai on generated audio; returns a runner and the paths."""
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"stands in for the audio file (hashed for checkpoints)")
    output = tmp_path / "out.txt"
    monkeypatch.setattr(transcribe, "_export_chunk", _fake_export)

    spent = []

    def run(audio: AudioSegment, client: FakeClient, **kwargs):
        monkeypatch.setattr(transcribe, "_load_audio", lambda path, folder: audio)
        monkeypatch.setattr(openai_api, "make_client", lambda key: client)
        spent.append(transcribe.transcribe_openai(source, output, "test-key", **kwargs))
        return output.read_text(encoding="utf-8")

    return SimpleNamespace(run=run, source=source, output=output, spent=spent)


def _saved_chunks(source: Path, model: str = config.DEFAULT_MODEL) -> int:
    return transcribe.saved_chunk_count(checkpoints.file_digest(source), model)


def test_chunk_length_depends_on_the_model():
    assert transcribe.segment_minutes("gpt-4o-transcribe") == 5
    assert transcribe.segment_minutes("whisper-1") == 10
    assert transcribe.segment_minutes("some-future-model") == 5


def test_chunk_length_can_be_set_for_every_model(monkeypatch):
    monkeypatch.setenv("SEGMENT_DURATION_MINUTES", "3")
    config.get_settings.cache_clear()
    assert transcribe.segment_minutes("whisper-1") == 3


def test_cuts_move_to_a_nearby_pause():
    # Speech with a pause from 4:52 to 4:54, i.e. 7 s before the 5:00 mark.
    audio = _tone(292_000) + AudioSegment.silent(2_000, 16000) + _tone(200_000)
    bounds = transcribe._chunk_bounds(audio, 5 * MINUTE)

    first_cut = bounds[0][1]
    assert 292_000 <= first_cut <= 294_000
    assert bounds[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in pairwise(bounds))


def test_a_tiny_tail_joins_the_previous_chunk():
    # 10:05 in 5-minute chunks: the 5 s tail must not become its own request.
    bounds = transcribe._chunk_bounds(_tone(10 * MINUTE + 5_000), 5 * MINUTE)
    assert len(bounds) == 2
    assert bounds[-1][1] == 10 * MINUTE + 5_000


def test_gpt4o_audio_goes_out_in_five_minute_chunks(pipeline):
    client = FakeClient(lambda name, ms: _text(f"Part {name}."))
    text = pipeline.run(_tone(12 * MINUTE), client)

    assert [name for name, _ in client.calls] == ["chunk_000", "chunk_001", "chunk_002"]
    assert all(ms <= 5 * MINUTE + 10_000 for _, ms in client.calls)
    assert "Part chunk_000." in text and "Part chunk_002." in text


def test_an_answer_cut_off_by_the_output_cap_is_split_and_redone(pipeline):
    def answer(name, ms):
        # Anything longer than 3 minutes "fills" the 2,000-token cap.
        if ms > 3 * MINUTE:
            return _text(f"Truncated {name}", tokens=2048)
        return _text(f"Complete {name}.", tokens=900)

    client = FakeClient(answer)
    text = pipeline.run(_tone(5 * MINUTE), client)

    assert [name for name, _ in client.calls] == [
        "chunk_000",
        "chunk_000a",
        "chunk_000b",
    ]
    assert "Truncated" not in text
    assert "Complete chunk_000a." in text and "Complete chunk_000b." in text


def test_a_failed_chunk_keeps_the_paid_parts_and_a_rerun_resumes(pipeline):
    audio = _tone(12 * MINUTE)

    def failing(name, ms):
        if name == "chunk_002":
            raise _server_error()
        return _text(f"Said in {name}.")

    first = FakeClient(failing)
    with pytest.raises(IncompleteTranscriptionError) as stopped:
        pipeline.run(audio, first)

    assert (stopped.value.completed, stopped.value.total) == (2, 3)
    partial = pipeline.output.read_text(encoding="utf-8")
    assert "Said in chunk_000." in partial and "Said in chunk_001." in partial
    assert "⚠️" in partial and "Not transcribed" in partial
    assert _saved_chunks(pipeline.source) == 2

    second = FakeClient(lambda name, ms: _text(f"Said in {name}."))
    text = pipeline.run(audio, second)

    # Only the missing chunk is sent (and paid for) again.
    assert [name for name, _ in second.calls] == ["chunk_002"]
    assert "Said in chunk_000." in text and "Said in chunk_002." in text
    assert "⚠️" not in text

    # The checkpoints outlive a successful run until the caller has stored the
    # result: a run stopped right after its last chunk can still finish free.
    assert _saved_chunks(pipeline.source) == 3
    third = FakeClient(lambda name, ms: _text("never sent"))
    assert pipeline.run(audio, third) == text
    assert third.calls == []


def test_no_credit_stops_at_the_first_request(pipeline):
    def refused(name, ms):
        raise _rate_limit_error("insufficient_quota")

    client = FakeClient(refused)
    with pytest.raises(OpenAIAccountError, match="credit exhausted"):
        pipeline.run(_tone(12 * MINUTE), client)

    assert len(client.calls) == 1
    assert _saved_chunks(pipeline.source) == 0


def test_whisper_segments_are_placed_on_the_recording_timeline(pipeline):
    def answer(name, ms):
        segment = SimpleNamespace(start=1.0, end=2.0, text=f" Hello from {name}. ")
        return _text("unused", segments=[segment])

    client = FakeClient(answer)
    text = pipeline.run(_tone(12 * MINUTE), client, model="whisper-1")

    # whisper-1 uses 10-minute chunks, so the second starts at about 10:00.
    assert "(0:01) Hello from chunk_000." in text
    assert "(10:0" in text and "Hello from chunk_001." in text


def test_serbian_cyrillic_comes_out_in_latin(pipeline):
    client = FakeClient(lambda name, ms: _text("Данас причамо о буџету."))
    text = pipeline.run(_tone(2 * MINUTE), client)
    assert "Danas pričamo o budžetu." in text


def test_serbian_can_stay_in_cyrillic(pipeline, monkeypatch):
    monkeypatch.setenv("SERBIAN_LATIN", "false")
    config.get_settings.cache_clear()
    client = FakeClient(lambda name, ms: _text("Данас причамо о буџету."))
    text = pipeline.run(_tone(2 * MINUTE), client)
    assert "Данас причамо о буџету." in text


def test_an_oversized_chunk_fails_before_any_request(tmp_path, monkeypatch):
    class BigSegment:
        def export(self, path, **kwargs):
            Path(path).write_bytes(b"x" * 2048)

    monkeypatch.setattr(transcribe, "MAX_SEGMENT_SIZE_MB", 0.001)
    with pytest.raises(transcribe.TranscriptionError, match="too large"):
        transcribe._export_chunk(BigSegment(), tmp_path, "chunk_000")
    assert not (tmp_path / "chunk_000.mp3").exists()


def test_untimed_text_gets_approximate_times_and_notes_in_place():
    pieces = [
        {"start": 0.0, "duration": 300.0, "text": "A. B. C. D. E. F. G. H."},
        {"start": 300.0, "duration": 300.0, "text": "Second chunk starts."},
    ]
    notes = [
        {"time": 10.0, "description": "Title slide"},
        {"time": 200.0, "description": "Budget slide"},
        {"time": 310.0, "description": "Roadmap slide"},
    ]
    out = transcribe.build_untimed_transcript(pieces, visual_notes=notes)

    assert out.split("\n\n") == [
        "(~0:00) A. B. C. D.",
        "🖥️ (0:10) Title slide",
        "(~2:30) E. F. G. H.",
        "🖥️ (3:20) Budget slide",
        "(~5:00) Second chunk starts.",
        "🖥️ (5:10) Roadmap slide",
    ]


def test_untimed_partial_transcript_marks_where_it_stops():
    pieces = [{"start": 0.0, "duration": 300.0, "text": "Done part."}]
    notes = [{"time": 400.0, "description": "Later slide"}]
    out = transcribe.build_untimed_transcript(
        pieces, visual_notes=notes, missing=(300.0, 720.0, "OpenAI credit exhausted.")
    )
    lines = out.split("\n\n")

    assert lines[0] == "(~0:00) Done part."
    assert lines[1].startswith("⚠️ (5:00–12:00) Not transcribed: OpenAI credit")
    assert lines[2] == "🖥️ (6:40) Later slide"


def test_a_failed_half_keeps_the_other_half_and_resumes_there(pipeline):
    audio = _tone(5 * MINUTE)

    def first_try(name, ms):
        if name == "chunk_000":
            return _text("Truncated", tokens=2048)
        if name == "chunk_000b":
            raise _server_error()
        return _text(f"Complete {name}.")

    # No chunk is complete yet, so this is a plain failure, not a partial run.
    with pytest.raises(TranscriptionError):
        pipeline.run(audio, FakeClient(first_try))

    second = FakeClient(lambda name, ms: _text(f"Complete {name}."))
    text = pipeline.run(audio, second)

    # Neither the capped whole nor the finished first half is paid for again.
    assert [name for name, _ in second.calls] == ["chunk_000b"]
    assert "Complete chunk_000a." in text and "Complete chunk_000b." in text


def test_an_answer_that_keeps_hitting_the_cap_is_split_only_once(pipeline):
    client = FakeClient(lambda name, ms: _text(f"Loop {name}", tokens=2048))
    text = pipeline.run(_tone(5 * MINUTE), client)

    assert [name for name, _ in client.calls] == [
        "chunk_000",
        "chunk_000a",
        "chunk_000b",
    ]
    # What still ends at the cap is said, not silently kept.
    assert text.count("stopped writing at its length limit") == 2


def test_an_unexpected_failure_still_leaves_a_partial_transcript(pipeline, monkeypatch):
    def broken_export(segment, temp_folder, name):
        if name == "chunk_001":
            raise RuntimeError("encoder crashed")
        return _fake_export(segment, temp_folder, name)

    monkeypatch.setattr(transcribe, "_export_chunk", broken_export)
    with pytest.raises(IncompleteTranscriptionError, match="encoder crashed"):
        pipeline.run(_tone(12 * MINUTE), FakeClient(lambda n, ms: _text("Said.")))
    assert "Not transcribed" in pipeline.output.read_text(encoding="utf-8")


def test_a_partial_run_keeps_the_account_error_as_its_cause(pipeline):
    def refused_later(name, ms):
        if name == "chunk_001":
            raise _rate_limit_error("credit_balance_exhausted")
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError) as stopped:
        pipeline.run(_tone(12 * MINUTE), FakeClient(refused_later))
    assert isinstance(stopped.value.__cause__, OpenAIAccountError)


def test_timed_partial_puts_later_notes_below_the_marker(pipeline):
    def answer(name, ms):
        if name == "chunk_001":
            raise _server_error()
        segment = SimpleNamespace(start=1.0, end=2.0, text=" Hello. ")
        return _text("unused", segments=[segment])

    notes = [
        {"time": 30.0, "description": "Early slide"},
        {"time": 700.0, "description": "Late slide"},
    ]
    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(
            _tone(12 * MINUTE),
            FakeClient(answer),
            model="whisper-1",
            visual_notes=notes,
        )
    lines = pipeline.output.read_text(encoding="utf-8").strip().split("\n\n")

    assert lines[0] == "(0:01) Hello."
    assert lines[1] == "🖥️ (0:30) Early slide"
    assert lines[2].startswith("⚠️")
    assert lines[3] == "🖥️ (11:40) Late slide"


def test_a_failing_checkpoint_save_does_not_fail_the_run(pipeline, monkeypatch):
    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(checkpoints.tempfile, "mkstemp", full_disk)
    text = pipeline.run(_tone(2 * MINUTE), FakeClient(lambda n, ms: _text("Said.")))
    assert "Said." in text


def test_runs_use_their_own_scratch_folders(pipeline):
    seen = []

    def answer(name, ms):
        seen.append(name)
        return _text("Said.")

    pipeline.run(_tone(2 * MINUTE), FakeClient(answer))
    temp_dir = config.get_settings().temp_dir
    # Nothing shared is left behind (two tabs used to share temp/segments).
    assert not list(temp_dir.glob("segments*"))


def test_progress_counts_finished_chunks(pipeline):
    reports = []
    pipeline.run(
        _tone(12 * MINUTE),
        FakeClient(lambda n, ms: _text("Said.")),
        progress_callback=reports.append,
    )
    progress = [r["progress"] for r in reports if r["status"] == "progress"]
    # Reported before each request: nothing is finished when part 1 starts.
    assert progress == [0.0, 1 / 3, 2 / 3]
    assert reports[-1]["status"] == "complete"


def test_a_run_reports_the_tokens_and_cost_it_used(pipeline):
    pipeline.run(
        _tone(12 * MINUTE), FakeClient(lambda n, ms: _text("Said.", tokens=200))
    )
    spent = pipeline.spent[-1]

    assert spent["requests"] == 3
    assert (spent["input_tokens"], spent["output_tokens"]) == (3_000, 600)
    # gpt-4o-transcribe: $2.50 / 1M input, $10 / 1M output.
    assert spent["cost_usd"] == pytest.approx((3_000 * 2.5 + 600 * 10) / 1e6)
    assert spent["estimated"] is False
    assert spent["seconds"] == pytest.approx(12 * 60, abs=1)


def test_the_capped_answer_that_was_split_is_counted_too(pipeline):
    def answer(name, ms):
        if name == "chunk_000":
            return _text("Truncated", tokens=2048)
        return _text("Complete.", tokens=500)

    pipeline.run(_tone(5 * MINUTE), FakeClient(answer))
    spent = pipeline.spent[-1]

    assert spent["requests"] == 3
    assert spent["output_tokens"] == 2048 + 500 + 500


def test_a_resumed_run_counts_what_its_first_attempt_paid(pipeline):
    audio = _tone(12 * MINUTE)

    def failing(name, ms):
        if name == "chunk_002":
            raise _server_error()
        return _text("Said.", tokens=100)

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(audio, FakeClient(failing))
    pipeline.run(audio, FakeClient(lambda n, ms: _text("Said.", tokens=100)))

    # Two chunks paid in the first attempt, one in the second: the transcript
    # cost all three.
    assert pipeline.spent[-1]["requests"] == 3
    assert pipeline.spent[-1]["output_tokens"] == 300


def test_the_recording_length_is_reported_even_when_a_chunk_is_resent(pipeline):
    def answer(name, ms):
        return _text("Capped", tokens=2048) if name == "chunk_000" else _text("Ok.")

    pipeline.run(_tone(5 * MINUTE), FakeClient(answer))
    spent = pipeline.spent[-1]

    assert spent["audio_seconds"] == pytest.approx(300, abs=1)
    assert spent["seconds"] == pytest.approx(600, abs=2)  # audio sent: 5:00 + 2 halves


def test_a_chunk_saved_before_cost_recording_is_estimated_not_free(pipeline):
    audio = _tone(12 * MINUTE)

    def failing(name, ms):
        if name == "chunk_002":
            raise _server_error()
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(audio, FakeClient(failing))
    # Strip the usage from the saved chunks, as an older version wrote them.
    checkpoint = transcribe.checkpoint_dir(
        checkpoints.file_digest(pipeline.source), config.DEFAULT_MODEL
    )
    for path in checkpoint.glob("chunk_*.json"):
        saved = checkpoints.load(checkpoint, path.stem)
        for piece in saved["pieces"]:
            piece.pop("usage", None)
        checkpoints.save(checkpoint, path.stem, saved)

    pipeline.run(audio, FakeClient(lambda n, ms: _text("Said.")))
    spent = pipeline.spent[-1]

    assert spent["requests"] == 3
    assert spent["estimated"] is True
    assert spent["cost_usd"] > 0.02  # two restored 5-minute chunks at ~$0.006/min


def test_a_partial_run_reports_what_it_paid_including_a_split_chunk(pipeline):
    def answer(name, ms):
        if name == "chunk_001":
            return _text("Capped", tokens=2048)
        if name == "chunk_001b":
            raise _server_error()
        return _text("Said.", tokens=100)

    with pytest.raises(IncompleteTranscriptionError) as stopped:
        pipeline.run(_tone(12 * MINUTE), FakeClient(answer))

    # chunk_000, the capped chunk_001 and its finished first half were paid.
    assert stopped.value.spent["requests"] == 3
    assert stopped.value.spent["output_tokens"] == 100 + 2048 + 100


def test_the_script_the_model_returned_is_kept_by_default(monkeypatch):
    # Macedonian colleagues use the app too; their Cyrillic must not be
    # rewritten in Serbian Latin unless someone asks for it.
    monkeypatch.delenv("SERBIAN_LATIN", raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().serbian_latin is False


def test_a_split_saved_before_cost_recording_is_estimated_on_resume(pipeline):
    audio = _tone(12 * MINUTE)

    def first_try(name, ms):
        if name == "chunk_001":
            return _text("Capped", tokens=2048)
        if name == "chunk_001b":
            raise _server_error()
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(audio, FakeClient(first_try))
    # Strip the usage from the split marker and its finished half, as the
    # version before cost recording wrote them.
    checkpoint = transcribe.checkpoint_dir(
        checkpoints.file_digest(pipeline.source), config.DEFAULT_MODEL
    )
    for name in ("part_001", "part_001a"):
        saved = checkpoints.load(checkpoint, name)
        saved.pop("usage", None)
        for piece in saved.get("pieces", []):
            piece.pop("usage", None)
        checkpoints.save(checkpoint, name, saved)

    pipeline.run(audio, FakeClient(lambda n, ms: _text("Said.")))
    spent = pipeline.spent[-1]

    # chunk_000, the capped chunk_001 and half a (both estimated), then half b
    # and chunk_002.
    assert spent["requests"] == 5
    assert spent["estimated"] is True


def test_a_partial_run_reports_the_length_it_covers_not_the_audio_sent(pipeline):
    def answer(name, ms):
        if name == "chunk_001":
            return _text("Capped", tokens=2048)
        if name == "chunk_001b":
            raise _server_error()
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError) as stopped:
        pipeline.run(_tone(12 * MINUTE), FakeClient(answer))

    # The partial transcript ends where chunk_001 starts, at 5:00.
    assert stopped.value.spent["audio_seconds"] == pytest.approx(300, abs=1)


def test_a_first_chunk_that_fails_after_a_split_still_reports_what_it_paid(pipeline):
    def answer(name, ms):
        if name == "chunk_000":
            return _text("Capped", tokens=2048)
        if name == "chunk_000b":
            raise _server_error()
        return _text("Said.")

    with pytest.raises(TranscriptionError) as stopped:
        pipeline.run(_tone(5 * MINUTE), FakeClient(answer))

    assert not isinstance(stopped.value, IncompleteTranscriptionError)
    assert stopped.value.spent["requests"] == 2  # the capped answer and half a
