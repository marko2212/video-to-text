"""Headless checks of the on-screen context controls (no browser, no network)."""

from streamlit.testing.v1 import AppTest

import config

_SCRIPT = """
import streamlit as st
import app

st.session_state.result = app.render_visual_options(st.session_state.source_type)
"""


def _run(source_type: str, **session: object) -> AppTest:
    test = AppTest.from_string(_SCRIPT)
    test.session_state["source_type"] = source_type
    for key, value in session.items():
        test.session_state[key] = value
    return test.run()


def test_audio_uploads_are_never_offered_on_screen_context():
    test = _run("audio")
    assert test.session_state.result is None
    assert not test.checkbox


def test_video_uploads_are_offered_the_checkbox_but_default_to_off():
    test = _run("video")
    assert len(test.checkbox) == 1
    assert test.checkbox[0].value is False
    # Nothing is configured until the box is ticked.
    assert test.session_state.result is None


def test_ticking_the_checkbox_reveals_the_controls():
    test = _run("video")
    test.checkbox[0].set_value(True).run()

    assert len(test.selectbox) == 1
    assert len(test.radio) == 1
    assert len(test.slider) == 1
    result = test.session_state.result
    assert result["model"] in config.VISION_MODELS
    assert result["detail"] in config.FRAME_DETAIL_LEVELS
    assert result["interval"] == config.FRAME_MAX_INTERVAL_SECONDS
    # The user is told the ceiling before spending anything.
    assert any("At most" in caption.value for caption in test.caption)


def test_the_screenshot_interval_is_adjustable():
    test = _run("video")
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(90).run()

    assert test.session_state.result["interval"] == 90.0


def test_the_interval_slider_stays_within_the_configured_range():
    test = _run("video")
    test.checkbox[0].set_value(True).run()
    slider = test.slider[0]

    assert slider.min == config.FRAME_INTERVAL_MIN_SECONDS
    assert slider.max == config.FRAME_INTERVAL_MAX_SECONDS
    assert slider.step == config.FRAME_INTERVAL_STEP_SECONDS


def test_no_api_key_warns_instead_of_offering_the_feature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config.get_settings.cache_clear()
    try:
        test = _run("video")
        test.checkbox[0].set_value(True).run()
        assert test.session_state.result is None
        assert len(test.warning) == 1
        assert not test.selectbox
    finally:
        config.get_settings.cache_clear()
