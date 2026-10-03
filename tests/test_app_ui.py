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
    cap = config.DEFAULT_FRAME_MAX_COUNT
    assert f"At most **{cap}** screenshots per video" in caption


_GUESS_SCRIPT = """
import streamlit as st
import app

st.caption(app._screenshot_estimate(st.session_state.interval, "gpt-5.4-nano", "low"))
"""


def _guess(video, interval: float) -> str:
    """The caption before a run: a ceiling, since nothing is scanned before Start."""
    test = AppTest.from_string(_GUESS_SCRIPT)
    test.session_state["video_path"] = video
    test.session_state["interval"] = interval
    return test.run().caption[0].value


def test_the_caption_gives_the_most_screenshots_the_video_can_yield(
    monkeypatch, tmp_path
):
    import app

    # A real file so the size lookup works; only the probe itself is stubbed.
    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    # 10 minutes, a look every 30 s: one at the start plus 20 more, if every
    # look shows a new picture.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 600.0)

    caption = _guess(video, 30.0)
    assert "At most **21** screenshots" in caption
    assert "one every 30 s of this 10:00 video" in caption
    # Faster changes are taken too — one per 2 s at most, within the cap.
    assert f"up to {config.DEFAULT_FRAME_MAX_COUNT} (" in caption
    # Repeats are left out, so the real number is usually much lower.
    assert "far fewer" in caption
    # Two prices in one caption: unescaped, Markdown took the text between the
    # dollar signs for a formula and dropped the signs.
    assert caption.count("$") == caption.count(chr(92) + "$") == 2


def test_the_ceiling_follows_the_interval_slider(monkeypatch, tmp_path):
    import app

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    # 83:12 — the length that exposed the estimate being pinned to the cap.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 4992.0)

    at_default = _guess(video, 30.0)
    at_120 = _guess(video, 120.0)

    assert "At most **167** screenshots" in at_default
    assert "At most **42** screenshots" in at_120
    assert "1:23:12 video" in at_default


def test_a_binding_cap_is_stated_rather_than_silently_applied(monkeypatch, tmp_path):
    import app

    video = tmp_path / "long.mkv"
    video.write_bytes(b"not really a video")
    # 10 hours every 5 s is far past the cap, so the interval cannot be honoured.
    monkeypatch.setattr(app, "_video_length", lambda path, size: 36000.0)

    caption = _guess(video, 5.0)
    assert f"At most **{config.DEFAULT_FRAME_MAX_COUNT}** screenshots" in caption
    assert "the limit per video" in caption
    # The substitution must be stated, not just the fact that a cap exists.
    assert "one every 5 s would give 7201" in caption
    assert "spread evenly" in caption
    assert "can add more" not in caption  # nothing can exceed the cap


def test_a_raised_cap_from_the_environment_reaches_the_caption(monkeypatch, tmp_path):
    import app

    monkeypatch.setenv("FRAME_MAX_COUNT", "300")
    config.get_settings.cache_clear()

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(app, "_video_length", lambda path, size: 4992.0)

    caption = _guess(video, 5.0)
    assert "At most **300** screenshots" in caption
    assert "would give 999" in caption


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
    key for key in app._SESSION_KEYS if key in st.session_state
)
"""


def test_every_run_state_key_is_initialised():
    # These three lists used to be maintained by hand and drifted apart, leaving
    # newly added keys missing from init and read before they existed.
    test = AppTest.from_string(_INIT_SCRIPT).run()
    import app

    assert test.session_state.seen == sorted(app._SESSION_KEYS)
    assert "elapsed_seconds" in app._RUN_STATE_KEYS
    assert "video_path" in app._RUN_STATE_KEYS
    # Reset with every new upload, so one file's result never shows for another.
    assert "partial" in app._RUN_STATE_KEYS
    assert "run_notices" in app._RUN_STATE_KEYS
    # These detect the new upload and carry the run request, so they are not.
    assert "upload_id" not in app._RUN_STATE_KEYS
    assert "job" not in app._RUN_STATE_KEYS


_RESULTS_SCRIPT = """
import streamlit as st
import app

app.init_session_state()
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

app.init_session_state()
app.run_transcription("OpenAI API", "whisper-1", False, "audio", visual=None)
st.session_state.result_elapsed = st.session_state.elapsed_seconds
app.render_run_notices()
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


def test_ticking_the_box_shows_the_settings_at_once_without_a_scan(
    monkeypatch, tmp_path
):
    # The scan decodes the whole video (about 1.5 min per hour); the slider
    # used to wait for it. The run scans instead.
    import app
    import frames

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not decoded")
    scans = []
    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: scans.append(1))
    monkeypatch.setattr(app, "_video_length", lambda path, size: 600.0)

    test = _run("video", video_path=video)
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(60).run()

    assert scans == []
    assert test.session_state.result["interval"] == 60.0
    assert "At most **11** screenshots" in test.caption[0].value


_JOB_SCRIPT = """
import streamlit as st
import app

st.session_state.result = app.render_visual_options(
    "video", disabled=st.session_state.get("disabled", False)
)
"""


def test_a_running_job_never_starts_a_scan_while_its_page_is_drawn(
    monkeypatch, tmp_path
):
    # A Stop pressed during such a scan left every control locked.
    import frames

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not decoded: the scan is faked")
    scans = []
    fake = {"duration": 60.0, "width": 0, "height": 0, "frames": []}
    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: scans.append(1) or fake)
    test = AppTest.from_string(_JOB_SCRIPT)
    test.session_state["video_path"] = video
    test.run()
    test.checkbox[0].set_value(True).run()

    test.session_state["disabled"] = True
    test.run()

    assert scans == []
    assert test.session_state.result is not None  # the job still gets its settings


def test_while_a_job_runs_a_stored_scan_gives_the_exact_count(monkeypatch, tmp_path):
    # An earlier run of the same video left its scan; reading it costs nothing.
    import frames

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not decoded: the scan is faked")
    frame = {"path": video, "scene": False, "hash": 1}
    stored = {
        "duration": 60.0,
        "width": 1920,
        "height": 1080,
        "frames": [{**frame, "time": 0.0}, {**frame, "time": 30.0, "hash": 99}],
    }
    monkeypatch.setattr(frames, "load_scan", lambda scan_dir: stored)
    test = AppTest.from_string(_JOB_SCRIPT)
    test.session_state["video_path"] = video
    test.run()
    test.checkbox[0].set_value(True).run()
    assert "At most" in test.caption[0].value  # before Start: the ceiling

    test.session_state["disabled"] = True
    test.run()

    assert "**2** screenshots will be described" in test.caption[0].value


def test_a_stopped_local_run_does_not_point_to_an_openai_rerun():
    import app

    text = app._describe_cost(
        {
            "provider": config.PROVIDER_LOCAL,
            "transcription": None,
            "vision": {
                "requests": 3,
                "seconds": 0.0,
                "input_tokens": 7_500,
                "output_tokens": 120,
                "cost_usd": 0.0017,
                "estimated": False,
            },
            "cost_usd": 0.0017,
            "estimated": False,
            "partial": True,
        }
    )

    assert "screenshot descriptions are kept" in text
    assert "OpenAI" not in text


def test_a_local_row_with_an_unknown_screenshot_cost_is_not_called_free():
    import json

    import app

    row = {
        "cost_usd": 0.0,
        "usage_json": json.dumps(
            {"provider": config.PROVIDER_LOCAL, "estimated": True}
        ),
    }
    assert app._history_cost(row) == "≈ $0"
    exact = {**row, "usage_json": json.dumps({"provider": config.PROVIDER_LOCAL})}
    assert app._history_cost(exact) == "free (local)"


def test_the_ceiling_is_priced_for_the_video_s_own_frame_size(monkeypatch, tmp_path):
    # Found by review: frames are described at full size, so a 4K screen costs
    # several times the Full HD figure the caption assumed.
    import app
    import vision

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(app, "_video_length", lambda path, size: 600.0)
    monkeypatch.setattr(app, "_video_frame_size", lambda path, size: (3840, 2160))

    caption = _guess(video, 30.0)

    tokens = vision.frame_tokens(3840, 2160)
    cost = vision.estimate_frame_cost(21, "gpt-5.4-nano", "low", tokens)
    assert f"At most **21** screenshots (up to {chr(92)}${cost:.2f})" in caption


def test_a_prepared_video_that_does_not_say_its_length_is_told_so(
    monkeypatch, tmp_path
):
    import app

    video = tmp_path / "browser recording.webm"
    video.write_bytes(b"no duration in the file")
    monkeypatch.setattr(app, "_video_length", lambda path, size: 0.0)

    caption = _guess(video, 30.0)

    assert "does not say how long it is" in caption
    assert "appears once it has been prepared" not in caption
