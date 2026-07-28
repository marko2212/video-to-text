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


def test_the_caption_shows_the_ceiling_when_the_video_length_is_unknown():
    test = _run("video")
    test.checkbox[0].set_value(True).run()

    caption = test.caption[0].value
    assert f"At most {config.DEFAULT_FRAME_MAX_COUNT}" in caption


def test_the_caption_estimates_the_screenshot_count_from_the_video_length(
    monkeypatch, tmp_path
):
    import app

    # A real file so the size lookup works; only the probe itself is stubbed.
    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    # 10 minutes, sampled every 30 s: one at the start plus 20 more.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 600.0)

    test = _run("video", video_path=video)
    test.checkbox[0].set_value(True).run()

    caption = test.caption[0].value
    assert "About **21** screenshots" in caption
    assert "10:00 video" in caption


def test_the_estimate_follows_the_interval_slider(monkeypatch, tmp_path):
    import app

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    # 83:12 — the length that exposed the estimate being pinned to the cap.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 4992.0)

    test = _run("video", video_path=video)
    test.checkbox[0].set_value(True).run()
    at_default = test.caption[0].value

    test.slider[0].set_value(120).run()
    at_120 = test.caption[0].value

    assert "About **167** screenshots" in at_default
    assert "About **42** screenshots" in at_120
    assert "1:23:12 video" in at_default


def test_a_binding_cap_is_stated_rather_than_silently_applied(monkeypatch, tmp_path):
    import app

    video = tmp_path / "long.mkv"
    video.write_bytes(b"not really a video")
    # 10 hours every 5 s is far past the cap, so the interval cannot be honoured.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 36000.0)

    test = _run("video", video_path=video)
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(5).run()

    caption = test.caption[0].value
    assert f"{config.DEFAULT_FRAME_MAX_COUNT}-screenshot limit" in caption
    assert "one about every 180 s" in caption
    # The substitution must be stated, not just the fact that a cap exists.
    assert "spread across the whole video" in caption


def test_a_raised_cap_from_the_environment_reaches_the_caption(monkeypatch, tmp_path):
    import app

    monkeypatch.setenv("FRAME_MAX_COUNT", "300")
    config.get_settings.cache_clear()

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(app, "_video_length", lambda path, size: 4992.0)

    test = _run("video", video_path=video)
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(5).run()

    caption = test.caption[0].value
    # 4992 s over 300 screenshots is one every ~17 s, not the default's ~25 s.
    assert "300-screenshot limit" in caption
    assert "**300** screenshots" in caption
    assert "one about every 17 s" in caption


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


_INIT_SCRIPT = """
import streamlit as st
import app

app.init_session_state()
st.session_state.seen = sorted(
    key for key in app._RUN_STATE_KEYS + ("original_filename",)
    if key in st.session_state
)
"""


def test_every_run_state_key_is_initialised():
    # These three lists used to be maintained by hand and drifted apart, leaving
    # newly added keys missing from init and read before they existed.
    test = AppTest.from_string(_INIT_SCRIPT).run()
    import app

    assert test.session_state.seen == sorted(
        (*app._RUN_STATE_KEYS, "original_filename")
    )
    assert "elapsed_seconds" in app._RUN_STATE_KEYS
    assert "video_path" in app._RUN_STATE_KEYS


_RESULTS_SCRIPT = """
import streamlit as st
import app

app.render_results()
"""


def _run_results(tmp_path, elapsed):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("hello", encoding="utf-8")
    test = AppTest.from_string(_RESULTS_SCRIPT)
    test.session_state["transcript_path"] = transcript
    test.session_state["srt_path"] = None
    test.session_state["elapsed_seconds"] = elapsed
    return test.run()


def test_the_result_reports_how_long_the_run_took(tmp_path):
    test = _run_results(tmp_path, 154.5)
    assert any("Finished in 2:34" in caption.value for caption in test.caption)


def test_a_long_run_is_reported_in_hours(tmp_path):
    test = _run_results(tmp_path, 3725.0)
    assert any("Finished in 1:02:05" in caption.value for caption in test.caption)


def test_no_timing_is_shown_when_it_was_not_recorded(tmp_path):
    test = _run_results(tmp_path, None)
    assert not any("Finished in" in caption.value for caption in test.caption)


_FAILED_RUN_SCRIPT = """
import streamlit as st
import app

app.run_transcription("OpenAI API", "whisper-1", False, "audio", visual=None)
st.session_state.result_elapsed = st.session_state.elapsed_seconds
app.render_results()
"""


def test_a_failed_run_does_not_keep_the_previous_run_s_time(tmp_path, monkeypatch):
    # Without a key the run returns early. The old figure must not survive, or
    # the panel claims "Finished in ..." for a run that never finished.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config.get_settings.cache_clear()
    try:
        transcript = tmp_path / "transcript_talk.txt"
        transcript.write_text("previous run", encoding="utf-8")

        test = AppTest.from_string(_FAILED_RUN_SCRIPT)
        test.session_state["original_filename"] = "talk.mp4"
        test.session_state["audio_path"] = tmp_path / "talk.wav"
        test.session_state["video_path"] = None
        test.session_state["transcript_path"] = transcript
        test.session_state["srt_path"] = None
        test.session_state["elapsed_seconds"] = 42.0
        test.run()

        assert test.session_state.result_elapsed is None
        assert len(test.error) == 1
        assert not any("Finished in" in caption.value for caption in test.caption)
    finally:
        config.get_settings.cache_clear()


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
