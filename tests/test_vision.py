"""Tests for the cost estimates used by the on-screen context UI (no network)."""

import config
import vision


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
