"""Tests for the vision step: cost estimates and describing frames (no network)."""

import httpx
import openai
import pytest

import checkpoints
import config
import openai_api
import vision
from exceptions import OpenAIAccountError, VisualContextError


def test_estimate_frame_tokens_scales_with_frames():
    single = vision.estimate_frame_tokens(1, "low")
    assert vision.estimate_frame_tokens(10, "low") == single * 10


def test_estimate_frame_tokens_high_detail_costs_more():
    assert vision.estimate_frame_tokens(5, "high") > vision.estimate_frame_tokens(
        5, "low"
    )


def test_estimate_frame_tokens_falls_back_for_unknown_detail():
    assert vision.estimate_frame_tokens(3, "enormous") == vision.estimate_frame_tokens(
        3, "low"
    )


def test_estimate_frame_cost_uses_the_model_price():
    cheap = vision.estimate_frame_cost(10, "gpt-5.4-nano", "low")
    dearer = vision.estimate_frame_cost(10, "gpt-5.4-mini", "low")
    assert cheap is not None and dearer is not None
    assert dearer > cheap


def test_estimate_frame_cost_is_none_for_an_unpriced_model():
    assert vision.estimate_frame_cost(10, "some-future-model") is None


def test_every_offered_vision_model_has_a_price():
    # The UI shows a cost hint per model; a missing price silently hides it.
    for model in config.VISION_MODELS:
        assert model in config.VISION_PRICE_PER_MTOK


# --- describe_keyframes with a fake client ----------------------------------


class FakeVisionClient:
    """Answers each frame with ``answer(call_number)`` and counts the calls."""

    def __init__(self, answer):
        self.calls = 0
        self._answer = answer
        completions = type("Completions", (), {"create": self.create})()
        self.chat = type("Chat", (), {"completions": completions})()

    def create(self, **kwargs):
        self.calls += 1
        result = self._answer(self.calls)
        if isinstance(result, Exception):
            raise result
        if result is None:
            return type("Response", (), {"choices": []})()
        message = type("Message", (), {"content": result})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    def close(self):
        pass


def _frames(tmp_path, count):
    frames = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append({"time": index * 10.0, "path": path})
    return frames


def _status_error(cls, status, body):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.openai.com/v1/x"), json=body
    )
    return cls(f"Error code: {status}", response=response, body=body)


def _use(monkeypatch, client):
    monkeypatch.setattr(openai_api, "make_client", lambda key: client)


def test_no_credit_stops_at_the_first_frame(tmp_path, monkeypatch):
    body = {"message": "no", "code": "insufficient_quota", "type": "insufficient_quota"}
    client = FakeVisionClient(lambda n: _status_error(openai.RateLimitError, 429, body))
    _use(monkeypatch, client)

    with pytest.raises(OpenAIAccountError, match="credit exhausted"):
        vision.describe_keyframes(_frames(tmp_path, 5), "test-key")
    assert client.calls == 1


def test_three_failures_in_a_row_stop_the_run(tmp_path, monkeypatch):
    client = FakeVisionClient(
        lambda n: _status_error(openai.InternalServerError, 500, {"message": "x"})
    )
    _use(monkeypatch, client)

    with pytest.raises(VisualContextError, match="3 frames in a row"):
        vision.describe_keyframes(_frames(tmp_path, 10), "test-key")
    assert client.calls == 3


def test_an_isolated_failure_only_skips_that_frame(tmp_path, monkeypatch):
    def answer(n):
        if n == 2:
            return _status_error(openai.BadRequestError, 400, {"message": "bad"})
        return "NONE" if n == 3 else f"Slide {n}"

    _use(monkeypatch, FakeVisionClient(answer))
    notes = vision.describe_keyframes(_frames(tmp_path, 4), "test-key")

    assert [note["description"] for note in notes] == ["Slide 1", "Slide 4"]


def test_descriptions_are_reused_from_the_cache(tmp_path, monkeypatch):
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 3)

    first = FakeVisionClient(lambda n: "NONE" if n == 2 else f"Slide {n}")
    _use(monkeypatch, first)
    notes = vision.describe_keyframes(frames, "test-key", cache=cache)

    second = FakeVisionClient(lambda n: "should not be asked")
    _use(monkeypatch, second)
    again = vision.describe_keyframes(frames, "test-key", cache=cache)

    assert second.calls == 0
    assert again == notes
    assert [note["description"] for note in notes] == ["Slide 1", "Slide 3"]


def test_progress_counts_finished_frames(tmp_path, monkeypatch):
    _use(monkeypatch, FakeVisionClient(lambda n: f"Slide {n}"))
    reports = []
    vision.describe_keyframes(
        _frames(tmp_path, 4), "test-key", progress_callback=reports.append
    )

    # Reported before each request: nothing is finished when frame 1 starts.
    assert [report["progress"] for report in reports] == [0.0, 0.25, 0.5, 0.75]
    assert reports[0]["message"] == "Reading screen 1 of 4…"


def test_an_empty_answer_is_a_failed_frame_not_a_crash(tmp_path, monkeypatch):
    _use(monkeypatch, FakeVisionClient(lambda n: None if n == 1 else f"Slide {n}"))
    failed = []
    notes = vision.describe_keyframes(_frames(tmp_path, 3), "test-key", failed=failed)

    assert [note["description"] for note in notes] == ["Slide 2", "Slide 3"]
    assert failed == [0.0]


def test_no_access_to_the_vision_model_is_not_an_account_error(tmp_path, monkeypatch):
    # A restricted key may transcribe but not use the vision model; that must
    # not cancel a transcription that would succeed.
    body = {"message": "Missing scopes: model.request"}
    _use(
        monkeypatch,
        FakeVisionClient(
            lambda n: _status_error(openai.PermissionDeniedError, 403, body)
        ),
    )
    with pytest.raises(VisualContextError):
        vision.describe_keyframes(_frames(tmp_path, 5), "test-key")


def test_a_resumed_step_is_not_stopped_by_old_bad_frames(tmp_path, monkeypatch):
    # Frames 3, 6 and 9 are always rejected; the others are described once and
    # cached. On the second pass the cached frames must break the "in a row"
    # count, or the resume stops with nothing.
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 9)
    bad = {20.0, 50.0, 80.0}

    def run():
        answers = []

        def answer(n):
            return answers.pop(0)

        client = FakeVisionClient(answer)
        _use(monkeypatch, client)
        return client, answers

    _, answers = run()
    answers.extend(
        _status_error(openai.BadRequestError, 400, {"message": "Invalid image"})
        if f["time"] in bad
        else f"Slide at {f['time']}"
        for f in frames
    )
    notes = vision.describe_keyframes(frames, "test-key", cache=cache)
    assert len(notes) == 6

    second, answers = run()
    answers.extend(
        _status_error(openai.BadRequestError, 400, {"message": "Invalid image"})
        for _ in bad
    )
    failed = []
    again = vision.describe_keyframes(frames, "test-key", cache=cache, failed=failed)

    assert second.calls == 3
    assert again == notes
    assert failed == sorted(bad)


def test_notes_made_before_an_account_error_are_handed_back(tmp_path, monkeypatch):
    body = {"message": "no", "code": "insufficient_quota", "type": "insufficient_quota"}

    def answer(n):
        if n == 3:
            return _status_error(openai.RateLimitError, 429, body)
        return f"Slide {n}"

    _use(monkeypatch, FakeVisionClient(answer))
    collected = []
    with pytest.raises(OpenAIAccountError):
        vision.describe_keyframes(_frames(tmp_path, 5), "test-key", collected=collected)
    assert [note["description"] for note in collected] == ["Slide 1", "Slide 2"]
