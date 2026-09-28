"""What paid requests used and cost (pure functions, prices from config)."""

from types import SimpleNamespace

import pytest

import usage


def test_a_token_billed_transcription_is_priced_from_reported_tokens():
    result = SimpleNamespace(
        usage={"type": "tokens", "input_tokens": 1_000, "output_tokens": 200}
    )
    record = usage.transcription_record(result, "gpt-4o-transcribe", 60.0)

    # $2.50 per 1M input tokens, $10 per 1M output tokens.
    assert record["cost_usd"] == pytest.approx(0.0025 + 0.002)
    assert (record["input_tokens"], record["output_tokens"]) == (1_000, 200)
    assert record["estimated"] is False


def test_whisper_is_billed_per_minute_exactly():
    result = SimpleNamespace(text="hello")  # no usage at all
    record = usage.transcription_record(result, "whisper-1", 90.0)

    assert record["cost_usd"] == pytest.approx(0.009)
    assert record["estimated"] is False


def test_a_duration_usage_overrides_the_chunk_length():
    result = SimpleNamespace(usage={"type": "duration", "seconds": 30})
    record = usage.transcription_record(result, "whisper-1", 90.0)
    assert record["seconds"] == 30.0
    assert record["cost_usd"] == pytest.approx(0.003)


def test_a_token_model_without_usage_falls_back_to_an_estimate():
    record = usage.transcription_record(
        SimpleNamespace(text="x"), "gpt-4o-transcribe", 120.0
    )
    assert record["cost_usd"] == pytest.approx(0.012)
    assert record["estimated"] is True


def test_an_unknown_model_is_an_estimate():
    record = usage.transcription_record(SimpleNamespace(), "some-future-model", 60.0)
    assert record["estimated"] is True
    assert record["cost_usd"] == pytest.approx(0.006)


def test_a_vision_answer_is_priced_from_its_token_usage():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=630, completion_tokens=40)
    )
    record = usage.vision_record(response, "gpt-5.4-nano")

    # $0.20 per 1M input tokens, $1.25 per 1M output tokens.
    assert record["cost_usd"] == pytest.approx((630 * 0.20 + 40 * 1.25) / 1e6)
    assert record["estimated"] is False


def test_a_vision_answer_without_usage_has_no_cost():
    record = usage.vision_record(SimpleNamespace(), "gpt-5.4-nano")
    assert record["cost_usd"] is None
    assert record["estimated"] is True


def test_summaries_add_up_and_flag_estimates():
    records = [
        {"seconds": 60.0, "input_tokens": 100, "output_tokens": 10, "cost_usd": 0.01},
        {
            "seconds": 30.0,
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": 0.003,
            "estimated": True,
        },
    ]
    summary = usage.summarize(records)

    assert summary == {
        "requests": 2,
        "seconds": 90.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": pytest.approx(0.013),
        "estimated": True,
    }


def test_costs_read_well_at_every_scale():
    assert usage.format_usd(0.0042) == "$0.0042"
    assert usage.format_usd(0.07) == "$0.07"
    assert usage.format_usd(1.234, estimated=True) == "≈ $1.23"


def test_zero_and_tiny_costs_do_not_read_as_a_rounding_error():
    assert usage.format_usd(0) == "$0"
    assert usage.format_usd(0.00004) == "<$0.0001"


def test_a_title_is_priced_like_a_chat_answer_and_marked_as_a_title():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=17_000, completion_tokens=10)
    )
    record = usage.title_record(response, "gpt-5.4-mini")

    # $0.75 per 1M input tokens, $4.50 per 1M output tokens.
    assert record["cost_usd"] == pytest.approx((17_000 * 0.75 + 10 * 4.50) / 1e6)
    assert record["kind"] == "title"
