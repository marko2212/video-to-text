"""Tests for the vision step: cost estimates and describing frames (no network)."""

import base64

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


def test_estimate_frame_tokens_uses_the_measured_full_hd_figure():
    # Measured on gpt-5.4: a Full HD frame is 2,519 input tokens, low or high.
    assert vision.estimate_frame_tokens(5, "low") == 5 * 2519
    assert vision.estimate_frame_tokens(5, "high") == vision.estimate_frame_tokens(
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
        tokens = {"prompt_tokens": 630, "completion_tokens": 40}
        return type("Response", (), {"choices": [choice], "usage": tokens})()

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


def test_every_description_used_is_accounted_for_including_cached_ones(
    tmp_path, monkeypatch
):
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 3)
    _use(monkeypatch, FakeVisionClient(lambda n: f"Slide {n}"))
    first = []
    vision.describe_keyframes(frames, "test-key", cache=cache, spent=first)

    _use(monkeypatch, FakeVisionClient(lambda n: "never asked"))
    second = []
    vision.describe_keyframes(frames, "test-key", cache=cache, spent=second)

    assert len(first) == 3 and all(r["input_tokens"] == 630 for r in first)
    # Reused descriptions were paid for by the first attempt; they still count.
    assert second == first


def test_descriptions_counted_by_a_saved_run_are_not_counted_again(
    tmp_path, monkeypatch
):
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 3)

    def answer(n):
        if n == 2:
            return _status_error(openai.BadRequestError, 400, {"message": "bad"})
        return f"Slide {n}"

    _use(monkeypatch, FakeVisionClient(answer))
    first = []
    vision.describe_keyframes(frames, "test-key", cache=cache, spent=first)
    # The app does this after saving the history row.
    vision.mark_billed(cache, [vision.frame_name(f["time"]) for f in frames])

    retry = FakeVisionClient(lambda n: "Slide now")
    _use(monkeypatch, retry)
    second = []
    vision.describe_keyframes(frames, "test-key", cache=cache, spent=second)

    assert len(first) == 2
    # Only the screenshot that was missing is paid for, and counted, again.
    assert retry.calls == 1 and len(second) == 1


def test_a_cached_description_from_before_cost_recording_counts_as_unknown(
    tmp_path, monkeypatch
):
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    checkpoints.save(cache, "frame_000000000", {"description": "Old slide"})
    _use(monkeypatch, FakeVisionClient(lambda n: "never asked"))
    spent = []
    vision.describe_keyframes(
        _frames(tmp_path, 1), "test-key", cache=cache, spent=spent
    )

    assert len(spent) == 1
    assert spent[0]["cost_usd"] is None and spent[0]["estimated"] is True


def test_frame_tokens_follow_the_measured_sizes():
    # Measured: 1920x1080 -> 2,519 tokens, 1280x720 -> 1,175.
    assert abs(vision.frame_tokens(1920, 1080) - 2519) <= 5
    assert abs(vision.frame_tokens(1280, 720) - 1175) <= 5


def test_each_frame_is_sent_as_its_own_picture_and_cached_ones_are_not_fetched(
    tmp_path, monkeypatch
):
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 3)
    checkpoints.save(cache, "frame_000000000", {"description": "Seen before"})
    fetched = []

    def picture(frame):
        fetched.append(frame["time"])
        own = tmp_path / f"own_{frame['time']:.0f}.jpg"
        own.write_bytes(f"picture at {frame['time']:.0f} s".encode())
        return own

    sent = []
    client = FakeVisionClient(lambda n: f"Slide {n}")
    original = client.create

    def create(**kwargs):
        url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
        sent.append(base64.b64decode(url.split(",", 1)[1]))
        return original(**kwargs)

    client.chat.completions.create = create
    _use(monkeypatch, client)
    vision.describe_keyframes(frames, "test-key", cache=cache, picture=picture)

    assert fetched == [10.0, 20.0]  # the cached frame is not extracted again
    assert sent == [b"picture at 10 s", b"picture at 20 s"]


def test_only_the_frames_a_saved_row_used_are_marked_as_counted(tmp_path, monkeypatch):
    # An unsaved attempt at one interval paid for frames at 0, 10 and 20 s; the
    # saved row used another interval and only the frame at 0 s.
    cache = checkpoints.run_dir("video123", "frames", "gpt-5.4-nano", "low")
    frames = _frames(tmp_path, 3)
    _use(monkeypatch, FakeVisionClient(lambda n: f"Slide {n}"))
    vision.describe_keyframes(frames, "test-key", cache=cache, spent=[])

    vision.mark_billed(cache, [vision.frame_name(0.0)])
    leftover = vision.unbilled_usage(cache, "gpt-5.4-nano", [vision.frame_name(0.0)])

    # The two frames no row has counted are still owed to a total.
    assert len(leftover) == 2
    assert all(record["input_tokens"] == 630 for record in leftover)
    again = []
    _use(monkeypatch, FakeVisionClient(lambda n: "never asked"))
    vision.describe_keyframes(frames[1:], "test-key", cache=cache, spent=again)
    assert len(again) == 2  # counted by the first saved row that uses them


def test_an_answer_without_choices_is_still_counted_as_paid(tmp_path, monkeypatch):
    _use(monkeypatch, FakeVisionClient(lambda n: None if n == 1 else f"Slide {n}"))
    spent = []
    failed = []
    vision.describe_keyframes(
        _frames(tmp_path, 2), "test-key", failed=failed, spent=spent
    )

    assert failed == [0.0]
    assert len(spent) == 2  # the empty answer's tokens were billed too
