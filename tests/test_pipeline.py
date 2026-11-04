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


# --- only part of the recording ----------------------------------------------------


def test_a_chosen_stretch_is_the_only_audio_sent(pipeline):
    client = FakeClient(lambda name, ms: _text(f"Said in {name}."))

    text = pipeline.run(_tone(10 * MINUTE), client, span=(120.0, 420.0))

    # Five minutes from 2:00 to 7:00: one chunk, on the recording's timeline.
    assert client.calls == [("chunk_000", 5 * MINUTE)]
    assert text.startswith(
        "⏭️ Transcribed 2:00–7:00 of the recording; the rest was left out."
    )
    assert "(~2:00) Said in chunk_000." in text
    assert pipeline.spent[-1]["audio_seconds"] == 300.0
    # Its saved parts are kept apart from the whole recording's.
    digest = checkpoints.file_digest(pipeline.source)
    assert transcribe.saved_chunk_count(digest, config.DEFAULT_MODEL, (120, 420)) == 1
    assert transcribe.saved_chunk_count(digest, config.DEFAULT_MODEL) == 0


def test_a_stretch_is_cut_within_its_own_ends():
    audio = _tone(12 * MINUTE)

    bounds = transcribe._chunk_bounds(audio, 5 * MINUTE, (MINUTE, 11 * MINUTE))

    assert bounds[0][0] == MINUTE and bounds[-1][1] == 11 * MINUTE
    assert len(bounds) == 2
    assert all(end > start for start, end in bounds)
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


def test_the_pause_search_stops_at_the_stretch_s_end():
    # Digital silence just past the end of the stretch would be the quietest
    # moment near the five-minute mark (a cut there was absorbed by the tail
    # merge, but it was sound the owner left out deciding the cut); the cut
    # must fall on the pause inside the stretch instead.
    audio = (
        _tone(291_500)
        + (_tone(1000) - 30)
        + _tone(12_500)
        + AudioSegment.silent(duration=5000)
        + _tone(290_000)
    )

    bounds = transcribe._chunk_bounds(audio, 5 * MINUTE, (0, 305_000))

    assert len(bounds) == 2
    assert abs(bounds[0][1] - 292_000) <= 500
    assert bounds[-1][1] == 305_000


def test_parts_of_the_whole_recording_serve_a_stretch_with_the_same_start(pipeline):
    # A run of the whole recording stopped at its third part; the owner leaves
    # the tail out and starts again: the two finished parts have the same
    # boundaries and must not be paid for twice.
    def failing(name, ms):
        if name == "chunk_002":
            raise _rate_limit_error("insufficient_quota")
        return _text(f"Said in {name}.")

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(_tone(15 * MINUTE), FakeClient(failing))

    again = FakeClient(lambda name, ms: _text("Paid again."))
    text = pipeline.run(_tone(15 * MINUTE), again, span=(0.0, 660.0))

    # Only the stretch's own last part, 10:00–11:00, is sent.
    assert [name for name, _ in again.calls] == ["chunk_002"]
    assert again.calls[0][1] == pytest.approx(MINUTE, abs=250)
    assert "Said in chunk_000." in text and "Said in chunk_001." in text
    assert "Paid again." in text
    assert pipeline.spent[-1]["requests"] == 3  # the two the first run paid, plus one
    digest = checkpoints.file_digest(pipeline.source)
    assert transcribe.saved_chunk_count(digest, config.DEFAULT_MODEL, (0, 660)) == 3
    assert transcribe.saved_chunk_count(digest, config.DEFAULT_MODEL) == 2
    assert transcribe.saved_parts_within(digest, config.DEFAULT_MODEL, (0, 660)) == 2


def test_a_stretch_with_another_start_borrows_nothing(pipeline):
    def failing(name, ms):
        if name == "chunk_002":
            raise _rate_limit_error("insufficient_quota")
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(_tone(15 * MINUTE), FakeClient(failing))

    shifted = FakeClient(lambda name, ms: _text("Said."))
    pipeline.run(_tone(15 * MINUTE), shifted, span=(60.0, 660.0))

    # 1:00–11:00 is cut at 6:00, not 5:00 and 10:00: nothing matches.
    assert [ms for _, ms in shifted.calls] == pytest.approx(
        [5 * MINUTE, 5 * MINUTE], abs=250
    )
    digest = checkpoints.file_digest(pipeline.source)
    # One of the whole recording's parts lies within the stretch, so the hint
    # may mention saved parts; the run itself reuses only matching boundaries.
    assert transcribe.saved_parts_within(digest, config.DEFAULT_MODEL, (60, 660)) == 1


def test_an_empty_stretch_is_refused_before_any_request(pipeline):
    client = FakeClient(lambda name, ms: _text("x"))

    with pytest.raises(TranscriptionError, match="Nothing to transcribe"):
        pipeline.run(_tone(MINUTE), client, span=(30.0, 30.5))

    assert client.calls == []


def test_a_partial_stretch_counts_from_its_own_start(pipeline):
    def failing(name, ms):
        if name == "chunk_001":
            raise _rate_limit_error("insufficient_quota")
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError) as stopped:
        pipeline.run(_tone(22 * MINUTE), FakeClient(failing), span=(60.0, 1260.0))

    # One five-minute chunk (cut in a pause near its end) of the stretch is done.
    assert stopped.value.spent["audio_seconds"] == pytest.approx(300.0, abs=11)
    text = pipeline.output.read_text(encoding="utf-8")
    assert text.startswith("⏭️ Transcribed 1:00–21:00 of the recording")
    assert "–21:00)" in text  # the missing part ends where the stretch does


def test_the_local_engine_cuts_the_stretch_out_and_keeps_the_recording_s_times(
    tmp_path, monkeypatch
):
    cuts = []
    seen = []

    def fake_to_wav(source, target, start=None, duration=None):
        cuts.append((start, duration))
        target.write_bytes(b"the cut")
        return target

    class Model:
        def transcribe(self, path, vad_filter):
            seen.append(Path(path).read_bytes())
            segments = [SimpleNamespace(start=0.0, end=2.0, text=" Hello there.")]
            return iter(segments), SimpleNamespace(duration=120.0)

    monkeypatch.setattr(transcribe, "to_wav", fake_to_wav)
    output = tmp_path / "out.txt"

    transcribe.transcribe_local(
        tmp_path / "meeting.wav", output, Model(), span=(600.0, 720.0)
    )

    assert cuts == [(600.0, 120.0)]
    assert seen == [b"the cut"]
    text = output.read_text(encoding="utf-8")
    assert text.startswith("⏭️ Transcribed 10:00–12:00 of the recording")
    assert "(10:00) Hello there." in text
    assert not list(config.get_settings().temp_dir.glob("stretch-*.wav"))


# --- a thin answer: part of a chunk missing, no error ----------------------------


def _speech(words: int, start: int = 0) -> str:
    # Distinct words, so a sentence can be told apart from a repeat of it.
    return " ".join(f"w{start + i}" for i in range(words)) + "."


def _thin_answer(name, ms):
    # Two full chunks at 120 words/min; the third comes back almost empty, as
    # chunk 08 of the owner's 52-minute presentation did (1 word/min).
    if name == "chunk_002":
        return _text("almost nothing here.", tokens=5)
    if name.startswith("chunk_002"):
        return _text(_speech(300), tokens=400)
    return _text(_speech(600), tokens=800)


def test_a_thin_answer_is_split_and_the_halves_replace_it(pipeline):
    client = FakeClient(_thin_answer)
    text = pipeline.run(_tone(15 * MINUTE), client)

    assert [name for name, _ in client.calls] == [
        "chunk_000",
        "chunk_001",
        "chunk_002",
        "chunk_002a",
        "chunk_002b",
    ]
    assert "almost nothing here" not in text
    assert "⚠️" not in text
    records = pipeline.spent[-1]["chunks"]
    assert [r["part"] for r in records] == ["000", "001", "002a", "002b"]
    assert [r["index"] for r in records] == [0, 1, 2, 2]
    assert not any(r["thin"] or r["capped"] for r in records)
    assert records[0]["words_per_min"] == pytest.approx(120, abs=5)
    assert transcribe.describe_flagged(records) is None
    # The thin answer was paid for and is counted with the halves.
    assert pipeline.spent[-1]["requests"] == 5


def test_a_half_that_is_still_thin_is_kept_and_marked(pipeline):
    def still_thin(name, ms):
        if name.startswith("chunk_002"):
            return _text("almost nothing here.", tokens=5)
        return _text(_speech(600), tokens=800)

    client = FakeClient(still_thin)
    text = pipeline.run(_tone(15 * MINUTE), client)

    assert [name for name, _ in client.calls][-3:] == [
        "chunk_002",
        "chunk_002a",
        "chunk_002b",
    ]
    # Both halves are kept, each with a line that says it may be incomplete.
    assert text.count("almost nothing here.") == 2
    assert text.count("came back with far fewer words than expected") == 2
    assert "against about 120 in the rest" in text
    records = pipeline.spent[-1]["chunks"]
    assert [r["thin"] for r in records] == [False, False, True, True]
    assert records[2]["typical_words_per_min"] == 120
    warning = transcribe.describe_flagged(records)
    assert warning.startswith("Speech may be missing in 2 part(s) — ")
    assert warning.count("(1 words/min against about 120)") == 2
    assert warning.endswith("The transcript marks them ⚠️.")


def test_the_first_chunk_cannot_be_thin_and_a_quiet_recording_never_is(pipeline):
    # No earlier chunk to compare with: the first answer is taken as it is.
    first = FakeClient(lambda n, ms: _text("short.", tokens=5))
    pipeline.run(_tone(5 * MINUTE), first)
    assert [name for name, _ in first.calls] == ["chunk_000"]

    # A recording with little speech (2 words/min throughout) has no typical
    # rate worth a comparison, so an empty chunk is not re-sent either.
    quiet = FakeClient(
        lambda n, ms: _text("" if n == "chunk_002" else _speech(10), tokens=20)
    )
    pipeline.run(_tone(15 * MINUTE), quiet)
    assert [name for name, _ in quiet.calls] == ["chunk_000", "chunk_001", "chunk_002"]
    assert "⚠️" not in pipeline.output.read_text(encoding="utf-8")


def test_a_thin_split_is_resumed_without_paying_for_the_thin_answer_again(pipeline):
    audio = _tone(15 * MINUTE)

    def first_try(name, ms):
        if name == "chunk_002b":
            raise _server_error()
        return _thin_answer(name, ms)

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(audio, FakeClient(first_try))

    second = FakeClient(_thin_answer)
    pipeline.run(audio, second)

    assert [name for name, _ in second.calls] == ["chunk_002b"]
    # Two full chunks, the thin answer, and its two halves.
    assert pipeline.spent[-1]["requests"] == 5


def test_every_answered_part_leaves_one_log_line_and_the_plan_is_logged_once(
    pipeline, caplog
):
    import logging

    def answer(name, ms):
        result = _text(_speech(600), tokens=800)
        result._request_id = f"req_{name}"
        return result

    with caplog.at_level(logging.INFO, logger="transcribe"):
        pipeline.run(_tone(10 * MINUTE), FakeClient(answer))

    messages = [record.getMessage() for record in caplog.records]
    plans = [m for m in messages if m.startswith("Chunk plan for meeting.wav with")]
    assert len(plans) == 1
    assert "gpt-4o-transcribe: 2 part(s): 0:00–" in plans[0]
    lines = [m for m in messages if m.startswith("Part 00")]
    assert len(lines) == 2
    assert lines[0].startswith("Part 000 (0:00–")
    assert "answered in" in lines[0]
    assert "800 output tokens" in lines[0]
    assert "600 words, 120 words/min, request req_chunk_000" in lines[0]
    assert "thin" not in lines[0] and "capped" not in lines[0]
    records = pipeline.spent[-1]["chunks"]
    assert records[0]["request_id"] == "req_chunk_000"
    assert records[0]["output_tokens"] == 800
    assert records[0]["chars"] == len(_speech(600))


def test_a_capped_answer_is_logged_and_recorded_as_such(pipeline, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="transcribe"):
        pipeline.run(
            _tone(5 * MINUTE),
            FakeClient(lambda name, ms: _text(f"Loop {name}", tokens=2048)),
        )

    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("Part")
    ]
    assert any(
        line.startswith("Part 000 (") and ", capped, request" in line for line in lines
    )
    records = pipeline.spent[-1]["chunks"]
    assert [r["capped"] for r in records] == [True, True]
    assert "stopped at the output cap" in transcribe.describe_flagged(records)


def test_parts_saved_by_an_older_version_still_get_a_record():
    pieces = [
        {
            "start": 300.0,
            "duration": 300.0,
            "text": _speech(500),
            "usage": [{"output_tokens": 640}],
            "thin": True,
        }
    ]

    [record] = transcribe.chunk_records(pieces)

    assert record["index"] == 0 and record["part"] is None
    assert (record["start"], record["end"]) == (300.0, 600.0)
    assert record["output_tokens"] == 640
    assert record["words"] == 500 and record["words_per_min"] == 100.0
    assert record["request_id"] is None
    assert record["thin"] and not record["capped"]


def test_a_thin_whisper_1_part_is_marked_among_its_timed_segments(pipeline):
    def answer(name, ms):
        if name.startswith("chunk_001"):
            return _text(
                "x", segments=[SimpleNamespace(start=1.0, end=2.0, text=" x ")]
            )
        words = [
            SimpleNamespace(start=i * 0.5, end=i * 0.5 + 0.4, text=f" w{i} ")
            for i in range(1200)
        ]
        return _text(_speech(1200), segments=words)

    text = pipeline.run(_tone(20 * MINUTE), FakeClient(answer), model="whisper-1")

    # The second 10-minute part answered with one word: split, both halves
    # still thin, each marked at its end among the timed lines.
    assert text.count("came back with far fewer words") == 2
    assert "(10:0" in text


# --- the cross-check against a local Whisper pass ---------------------------------


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel: fixed timed segments."""

    def __init__(self, segments, duration=120.0):
        self.segments = segments
        self.duration = duration
        self.calls = 0

    def transcribe(self, path, vad_filter):
        self.calls += 1
        return iter(self.segments), SimpleNamespace(duration=self.duration)


_SAID = [
    (0.0, 4.0, "Good morning everyone and welcome to the quarterly review."),
    (4.0, 8.0, "Today we look at the numbers for the third quarter."),
    (60.0, 64.0, "Let us move on to the product roadmap for next year."),
    (64.0, 68.0, "The migration to the new platform starts in January."),
    (68.0, 72.0, "Every team gets a dedicated contact for the transition."),
    (72.0, 76.0, "Questions about the timeline go to the programme office."),
    (100.0, 104.0, "That is all for today, thank you for joining."),
]


def _whisper():
    return FakeWhisper([SimpleNamespace(start=s, end=e, text=t) for s, e, t in _SAID])


def test_the_local_pass_re_times_the_transcript_and_fills_what_the_api_lacked(
    pipeline,
):
    # The API answer lacks the roadmap part; its second paragraph is stamped
    # by character share near 1:00 although it was spoken at 1:40.
    api_text = (
        "Good morning everyone and welcome to the quarterly review. Today we "
        "look at the numbers for the third quarter. That is all for today, "
        "thank you for joining."
    )
    whisper = _whisper()
    client = FakeClient(lambda n, ms: _text(api_text, tokens=60))

    text = pipeline.run(_tone(2 * MINUTE), client, verify=(whisper, "base"))

    assert whisper.calls == 1
    lines = text.strip().split("\n\n")
    assert lines[0].startswith("🔍 Checked against a local Whisper pass (base): ")
    assert lines[1].startswith("(0:00) Good morning everyone")
    assert lines[2].startswith("🧩 (1:00) Let us move on to the product roadmap")
    assert "(~" not in text
    verified = pipeline.spent[-1]["verified"]
    assert verified["model"] == "base"
    assert verified["gaps"] == 1 and verified["added_words"] == 38
    assert verified["anchored"] == verified["paragraphs"] == 1
    assert verified["seconds"] >= 0
    assert "chunks" in pipeline.spent[-1]


def test_a_failed_local_pass_leaves_the_paid_transcript_as_it_is(pipeline):
    class Broken:
        def transcribe(self, path, vad_filter):
            raise RuntimeError("model file corrupt")

    client = FakeClient(lambda n, ms: _text("Said in the meeting.", tokens=10))
    text = pipeline.run(_tone(2 * MINUTE), client, verify=(Broken(), "base"))

    assert "(~0:00) Said in the meeting." in text
    assert "🔍" not in text
    verified = pipeline.spent[-1]["verified"]
    assert verified == {"model": "base", "error": "model file corrupt"}


def test_a_model_with_its_own_times_is_not_checked(pipeline):
    whisper = _whisper()
    segment = SimpleNamespace(start=1.0, end=2.0, text=" Hello. ")
    client = FakeClient(lambda n, ms: _text("unused", segments=[segment]))

    text = pipeline.run(
        _tone(2 * MINUTE), client, model="whisper-1", verify=(whisper, "base")
    )

    assert whisper.calls == 0
    assert "(0:01) Hello." in text
    assert "verified" not in pipeline.spent[-1]


def test_a_partial_run_skips_the_check(pipeline):
    whisper = _whisper()

    def failing(name, ms):
        if name == "chunk_001":
            raise _server_error()
        return _text("Said.")

    with pytest.raises(IncompleteTranscriptionError):
        pipeline.run(_tone(12 * MINUTE), FakeClient(failing), verify=(whisper, "base"))

    assert whisper.calls == 0


# --- batched local decoding (opt-in) ---------------------------------------------


def _word(start: float, end: float, word: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, word=word)


class FakeBatched:
    """Stands in for faster_whisper.BatchedInferencePipeline."""

    def __init__(self, segments, duration=120.0):
        self.segments = segments
        self.duration = duration
        self.kwargs = []

    def transcribe(self, path, **kwargs):
        self.kwargs.append(kwargs)
        return iter(self.segments), SimpleNamespace(duration=self.duration)


def test_a_batched_local_run_splits_each_piece_into_sentences(tmp_path, monkeypatch):
    # One piece of up to 30 s, as batched decoding returns it, with word times.
    piece = SimpleNamespace(
        start=0.0,
        end=7.0,
        text=" Hello there. How are you? Fine thanks",
        words=[
            _word(0.0, 0.5, " Hello"),
            _word(0.6, 1.2, " there."),
            _word(1.5, 2.0, " How"),
            _word(2.1, 2.4, " are"),
            _word(2.5, 3.0, " you?"),
            _word(6.0, 6.5, " Fine"),  # after a pause of three seconds
            _word(6.6, 7.0, " thanks"),
        ],
    )
    batched = FakeBatched([piece])
    wrapped = []

    def wrap(model):
        wrapped.append(model)
        return batched

    monkeypatch.setattr(transcribe, "_batched_pipeline", wrap)
    model = object()
    output, srt = tmp_path / "out.txt", tmp_path / "out.srt"

    transcribe.transcribe_local(
        tmp_path / "meeting.wav",
        output,
        model,
        with_timestamps=True,
        srt_output_file=srt,
        batched=True,
    )

    assert wrapped == [model]
    # Word times, not the pieces' own timestamps: with those, faster-whisper
    # 1.2.1 left out a piece's last sentence.
    assert batched.kwargs == [
        {
            "vad_filter": True,
            "batch_size": config.LOCAL_BATCH_SIZE,
            "word_timestamps": True,
        }
    ]
    cues = [
        cue.splitlines()[1:] for cue in srt.read_text(encoding="utf-8").split("\n\n")
    ]
    assert cues == [
        ["00:00:00,000 --> 00:00:01,200", "Hello there."],
        ["00:00:01,500 --> 00:00:03,000", "How are you?"],
        ["00:00:06,000 --> 00:00:07,000", "Fine thanks"],
    ]


def test_a_long_batched_piece_without_pauses_is_cut_into_short_lines():
    words = [_word(i * 0.5, i * 0.5 + 0.4, f" w{i}") for i in range(50)]  # 25 s

    lines = transcribe._split_by_words(
        SimpleNamespace(start=0.0, end=25.0, text="", words=words)
    )

    assert len(lines) > 2
    assert all(end - start <= 10.0 for start, end, _ in lines)
    said = " ".join(text for *_, text in lines).split()
    assert said == [f"w{i}" for i in range(50)]


def test_a_batched_piece_without_word_times_is_kept_as_it_is():
    piece = SimpleNamespace(start=1.0, end=2.0, text=" Hi.", words=None)

    assert transcribe._split_by_words(piece) == [(1.0, 2.0, " Hi.")]


def test_a_batched_cross_check_goes_through_the_batched_pipeline(pipeline, monkeypatch):
    api_text = (
        "Good morning everyone and welcome to the quarterly review. Today we "
        "look at the numbers for the third quarter. That is all for today, "
        "thank you for joining."
    )
    whisper = _whisper()
    batched = FakeBatched(whisper.segments)
    monkeypatch.setattr(transcribe, "_batched_pipeline", lambda model: batched)
    client = FakeClient(lambda n, ms: _text(api_text, tokens=60))

    text = pipeline.run(
        _tone(2 * MINUTE), client, verify=(whisper, "base"), verify_batched=True
    )

    assert whisper.calls == 0
    assert len(batched.kwargs) == 1
    assert text.startswith("🔍 Checked against a local Whisper pass (base): ")
    assert "🧩 (1:00) Let us move on to the product roadmap" in text
