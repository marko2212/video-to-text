"""The whole page, headless: uploads, runs, failures and history (no network).

The engines are replaced with fakes that write a transcript, so these tests
exercise the UI flow — the two-step Start, run notices, result state and the
history fragment — not transcription itself.
"""

import threading
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
import config
import db
import transcribe
from exceptions import (
    IncompleteTranscriptionError,
    OpenAIAccountError,
    TranscriptionError,
)

_APP = """
import app
app.main()
"""
_TIMEOUT = 30


def _start(test: AppTest) -> AppTest:
    button = next(b for b in test.button if b.label == "Start Transcription")
    return button.click().run()


def _upload(test: AppTest, name: str, content: bytes) -> AppTest:
    return test.file_uploader[0].set_value((name, content, "audio/wav")).run()


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the OpenAI pipeline; each call writes a transcript of the input."""
    calls = []

    def fake(input_file, output_file, api_key, **kwargs):
        calls.append(Path(input_file).read_bytes())
        callback = kwargs["progress_callback"]
        callback({"status": "progress", "message": "part 1 of 1", "progress": 0.0})
        Path(output_file).write_text(
            f"Transcript of {Path(input_file).read_bytes().decode()}", encoding="utf-8"
        )
        callback({"status": "complete", "message": "Transcription completed"})

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    return calls


def test_a_same_name_upload_with_new_content_replaces_the_old_file(fake_openai):
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "call.wav", b"first recording")
    _start(test)
    assert "first recording" in test.session_state.transcript_path.read_text(
        encoding="utf-8"
    )

    _upload(test, "call.wav", b"second recording")

    assert test.session_state.audio_path.read_bytes() == b"second recording"
    # The previous file's transcript must not be shown as this one's.
    assert test.session_state.transcript_path is None
    _start(test)
    assert fake_openai == [b"first recording", b"second recording"]


def test_two_runs_in_one_session_both_finish_and_are_saved(fake_openai):
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")

    _start(test)
    _start(test)

    assert not test.exception
    assert len(db.list_transcriptions()) == 2
    assert test.session_state.job is None
    start = next(b for b in test.button if b.label == "Start Transcription")
    assert not start.disabled
    assert "progress_container" not in test.session_state
    assert any("Transcript of talk audio" in area.value for area in test.text_area)


_PROGRESS_SCRIPT = """
import streamlit as st
import app

if st.checkbox("More fields", key="more"):
    st.text_input("Extra field")
report = app.make_progress_callback(st.empty())
report({"status": "complete", "message": "Run finished"})
st.button("Below the progress box")
"""


def test_progress_is_drawn_in_this_run_s_placeholder():
    # A placeholder cached in session_state drew into a stale position from the
    # second run on: the message vanished, or replaced a widget after a layout
    # change. A per-run placeholder survives both.
    test = AppTest.from_string(_PROGRESS_SCRIPT).run()
    assert [s.value for s in test.success] == ["Run finished"]

    test.checkbox(key="more").check().run()

    assert [s.value for s in test.success] == ["Run finished"]
    assert len(test.text_input) == 1
    assert [b.label for b in test.button] == ["Below the progress box"]


_PARAMS = {
    "provider": config.PROVIDER_OPENAI,
    "model": config.DEFAULT_MODEL,
    "with_timestamps": False,
    "source_type": "audio",
    "visual": None,
}


def _start_button(test: AppTest):
    return next(b for b in test.button if b.label == "Start Transcription")


def test_an_interrupted_run_is_reported_and_the_controls_return(fake_openai):
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    # What a stopped script leaves behind: a job whose worker thread has ended
    # without clearing it.
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    test.session_state["job"] = {"params": _PARAMS, "worker": worker}
    test.run()

    assert test.session_state.job is None
    assert any("stopped before it finished" in w.value for w in test.warning)
    assert not _start_button(test).disabled
    assert fake_openai == []  # not silently restarted


def test_a_stopped_run_still_in_its_request_keeps_the_page_locked(fake_openai):
    # Stop only takes effect at the thread's next Streamlit call; until then it
    # may still be waiting for a paid request. Unlocking early let Start send
    # the same chunk again, in parallel.
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    earlier = test.session_state.audio_path.parent / "transcript_earlier.txt"
    earlier.write_text("An earlier result.", encoding="utf-8")
    test.session_state["transcript_path"] = earlier
    release = threading.Event()
    worker = threading.Thread(target=release.wait)
    worker.start()
    try:
        test.session_state["job"] = {"params": _PARAMS, "worker": worker}
        test.run()

        assert test.session_state.job is not None
        assert _start_button(test).disabled
        assert any("Stopping" in i.value for i in test.info)
        assert fake_openai == []  # the job is not started a second time
        # Editing the result is a widget change, which would stop a job too.
        [preview] = [a for a in test.text_area if a.label == "Transcript preview:"]
        assert preview.disabled
    finally:
        release.set()
        worker.join()

    test.run()
    assert test.session_state.job is None
    assert not _start_button(test).disabled
    assert any("stopped before it finished" in w.value for w in test.warning)


def test_no_credit_is_one_clear_error_and_nothing_is_saved(monkeypatch):
    def refused(*args, **kwargs):
        raise OpenAIAccountError("OpenAI credit exhausted — add credit.")

    monkeypatch.setattr(transcribe, "transcribe_openai", refused)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)

    [error] = test.error
    assert error.value.startswith("OpenAI credit exhausted — add credit.")
    if app.LOCAL_AVAILABLE:
        assert "Local engine needs no key or credit" in error.value
    assert db.list_transcriptions() == []
    assert test.session_state.transcript_path is None


def test_a_partial_run_shows_what_it_has_but_does_not_save_it(monkeypatch):
    def stops_early(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("Part one.\n\n⚠️ Not transcribed", encoding="utf-8")
        raise IncompleteTranscriptionError("Cannot reach OpenAI.", completed=2, total=3)

    monkeypatch.setattr(transcribe, "transcribe_openai", stops_early)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)

    assert any("Partial transcript — 2 of 3" in w.value for w in test.warning)
    assert any("Stopped after part 2 of 3" in e.value for e in test.error)
    assert any("Part one." in area.value for area in test.text_area)
    assert db.list_transcriptions() == []
    assert test.session_state.elapsed_seconds is None


def test_history_entries_stay_closed_and_unloaded():
    db.init_db()
    for index in range(3):
        db.add_transcription(
            filename=f"meeting{index}.mkv",
            source_type="video",
            model="gpt-4o-transcribe",
            with_timestamps=False,
            transcript="long transcript " * 100,
        )
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()

    assert len(test.expander) == 3
    # Nothing inside a closed entry is built: no text boxes, no downloads.
    assert not test.text_area


_VISUAL_RUN_SCRIPT = """
import streamlit as st
import app
import db

db.init_db()
app.init_session_state()
st.session_state.original_filename = "talk.mp4"
st.session_state.audio_path = st.session_state.audio
st.session_state.video_path = st.session_state.audio
app.run_transcription(
    st.session_state.provider, st.session_state.model, False, "video",
    visual={"model": "gpt-5.4-nano", "detail": "low", "interval": 30.0},
)
app.render_run_notices()
"""


def _fake_scan(image: Path, count: int, spacing: float = 10.0) -> dict:
    """A scan of `count` distinct screens (all scene changes), for the UI flow."""
    return {
        "duration": count * spacing,
        "width": 1920,
        "height": 1080,
        "frames": [
            {"time": i * spacing, "path": image, "scene": True, "hash": 0xFF << (8 * i)}
            for i in range(count)
        ],
    }


def _visual_run(tmp_path, monkeypatch, provider, model):
    """Run with on-screen context where OpenAI refuses the account."""
    import frames
    import vision

    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"audio")
    ran = []

    def refused(*args, **kwargs):
        raise OpenAIAccountError("OpenAI credit exhausted — add credit.")

    def fake_local(input_file, output_file, whisper_model, **kwargs):
        ran.append("local")
        Path(output_file).write_text("Local transcript", encoding="utf-8")

    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: _fake_scan(audio, 1))
    monkeypatch.setattr(vision, "describe_keyframes", refused)
    monkeypatch.setattr(
        transcribe, "transcribe_openai", lambda *a, **k: ran.append("x")
    )
    monkeypatch.setattr(transcribe, "transcribe_local", fake_local)
    monkeypatch.setattr(app, "load_whisper_model", lambda *a: object())
    monkeypatch.setattr(app, "_model_is_cached", lambda name: True)

    test = AppTest.from_string(_VISUAL_RUN_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["audio"] = audio
    test.session_state["provider"] = provider
    test.session_state["model"] = model
    return test.run(), ran


def test_no_credit_for_screenshots_stops_an_openai_run(tmp_path, monkeypatch):
    test, ran = _visual_run(tmp_path, monkeypatch, config.PROVIDER_OPENAI, "whisper-1")

    assert ran == []  # the transcription would be refused the same way
    [error] = test.error
    assert error.value.startswith("OpenAI credit exhausted — add credit.")


def test_no_credit_for_screenshots_only_warns_a_local_run(tmp_path, monkeypatch):
    test, ran = _visual_run(tmp_path, monkeypatch, config.PROVIDER_LOCAL, "base")

    assert ran == ["local"]
    assert any("On-screen context skipped" in w.value for w in test.warning)
    assert not test.error


def test_the_server_listens_on_localhost_only():
    # Transcripts are confidential and the app has no login: it must not be
    # reachable from the network unless someone deliberately says so.
    settings = tomllib.loads(
        (config.BASE_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )
    assert settings["server"]["address"] in {"localhost", "127.0.0.1"}
    assert settings["browser"]["gatherUsageStats"] is False


_TWO_RUNS_SCRIPT = """
import streamlit as st
import app
import db

db.init_db()
app.init_session_state()
st.session_state.original_filename = "talk.wav"
st.session_state.audio_path = st.session_state.audio
if st.checkbox("More fields", key="more"):
    st.text_input("Extra field")
app.run_transcription("OpenAI API", "whisper-1", False, "audio")
st.button("Below the run")
"""


def test_each_run_draws_its_progress_in_a_fresh_placeholder(tmp_path, fake_openai):
    # The placeholder used to be kept in session_state; from the second run on
    # it pointed at a stale position, so the message vanished or replaced a
    # widget once the layout above it changed.
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"talk audio")
    test = AppTest.from_string(_TWO_RUNS_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["audio"] = audio
    test.run()
    assert [s.value for s in test.success] == ["Transcription completed"]

    test.checkbox(key="more").check().run()

    assert [s.value for s in test.success] == ["Transcription completed"]
    assert len(test.text_input) == 1
    assert [b.label for b in test.button] == ["Below the run"]
    assert len(db.list_transcriptions()) == 2


def test_cleaning_up_empties_the_uploader_instead_of_preparing_it_again():
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    uploads = config.get_settings().upload_dir
    assert (uploads / "talk.wav").exists()

    clean = next(b for b in test.button if "Clean temporary files" in b.label)
    clean.click().run()

    assert not (uploads / "talk.wav").exists()
    assert test.session_state.audio_path is None
    # A new uploader key is a new, empty uploader, so the next click cannot save
    # and extract the same file again. (AppTest cannot run past a key change
    # made by st.rerun, so the empty uploader itself is checked live.)
    assert test.session_state.uploader_generation == 1


def test_clean_up_is_refused_while_a_run_is_active_anywhere():
    import checkpoints

    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    uploads = config.get_settings().upload_dir
    with checkpoints.active_run():  # e.g. a run in another tab
        test.run()
        clean = next(b for b in test.button if "Clean temporary files" in b.label)
        assert clean.disabled
    assert (uploads / "talk.wav").exists()


_CLEAN_SCRIPT = """
import streamlit as st
import app
import checkpoints

app.init_session_state()
if st.session_state.get("hold"):
    with checkpoints.active_run():
        app.clean_temp_files()
else:
    app.clean_temp_files()
st.session_state.result = st.session_state.clean_message
"""


def test_clean_up_refuses_at_click_time_too(tmp_path):
    uploads = config.get_settings().upload_dir
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "call.wav").write_bytes(b"x")
    test = AppTest.from_string(_CLEAN_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["hold"] = True
    test.run()

    assert test.session_state.result[0] == "warning"
    assert (uploads / "call.wav").exists()


def test_clean_up_never_deletes_history_kept_inside_the_temp_folder(
    tmp_path, monkeypatch
):
    # DATA_DIR may point inside TEMP_DIR; the history must survive a clean-up.
    temp_dir = tmp_path / "work"
    monkeypatch.setenv("TEMP_DIR", str(temp_dir))
    monkeypatch.setenv("DATA_DIR", str(temp_dir / "history"))
    config.get_settings.cache_clear()
    db.init_db()
    (temp_dir / "segments-x").mkdir()
    (temp_dir / "transcript_call.txt").write_text("t", encoding="utf-8")

    test = AppTest.from_string(_CLEAN_SCRIPT, default_timeout=_TIMEOUT).run()

    assert test.session_state.result == ("success", "Temporary files cleaned.")
    assert (temp_dir / "history" / "transcriptions.db").exists()
    assert not (temp_dir / "segments-x").exists()
    assert not (temp_dir / "transcript_call.txt").exists()


def test_a_retry_that_fails_early_keeps_the_partial_banner(monkeypatch):
    calls = []

    def first_partial_then_refused(input_file, output_file, api_key, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            Path(output_file).write_text("Part one.", encoding="utf-8")
            raise IncompleteTranscriptionError("No connection.", completed=1, total=3)
        raise OpenAIAccountError("OpenAI credit exhausted — add credit.")

    monkeypatch.setattr(transcribe, "transcribe_openai", first_partial_then_refused)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)
    _start(test)

    # The partial text is still on screen, so it must still be marked partial.
    assert any("Partial transcript — 1 of 3" in w.value for w in test.warning)
    assert any("Part one." in area.value for area in test.text_area)


def test_notes_described_before_a_refusal_still_reach_a_local_transcript(
    tmp_path, monkeypatch
):
    import frames
    import vision

    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"audio")
    received = []

    def refused_after_two(*args, collected=None, **kwargs):
        collected.extend(
            [
                {"time": 5.0, "description": "Title slide"},
                {"time": 40.0, "description": "Budget slide"},
            ]
        )
        raise OpenAIAccountError("OpenAI credit exhausted — add credit.")

    def fake_local(input_file, output_file, whisper_model, **kwargs):
        received.append(kwargs["visual_notes"])
        Path(output_file).write_text("Local transcript", encoding="utf-8")

    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: _fake_scan(audio, 4))
    monkeypatch.setattr(vision, "describe_keyframes", refused_after_two)
    monkeypatch.setattr(transcribe, "transcribe_local", fake_local)
    monkeypatch.setattr(app, "load_whisper_model", lambda *a: object())
    monkeypatch.setattr(app, "_model_is_cached", lambda name: True)

    test = AppTest.from_string(_VISUAL_RUN_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["audio"] = audio
    test.session_state["provider"] = config.PROVIDER_LOCAL
    test.session_state["model"] = "base"
    test.run()

    assert [n["description"] for n in received[0]] == ["Title slide", "Budget slide"]
    assert any("incomplete" in w.value and "2 notes" in w.value for w in test.warning)


def test_the_cost_of_a_run_is_shown_and_saved(monkeypatch):
    import json

    def paid(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("Transcript.", encoding="utf-8")
        return {
            "requests": 2,
            "seconds": 600.0,
            "input_tokens": 12_000,
            "output_tokens": 3_000,
            "cost_usd": 0.06,
            "estimated": False,
        }

    monkeypatch.setattr(transcribe, "transcribe_openai", paid)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)

    assert any(
        "💵 $0.06" in c.value and "12,000 → 3,000 tokens" in c.value
        for c in test.caption
    )
    [row] = db.list_transcriptions()
    assert row["cost_usd"] == 0.06
    saved = json.loads(db.get_transcription(row["id"])["usage_json"])
    assert saved["transcription"]["requests"] == 2


def test_a_partial_run_shows_what_it_has_spent_so_far(monkeypatch):
    def stops_early(input_file, output_file, api_key, **kwargs):
        Path(output_file).write_text("Part one.", encoding="utf-8")
        spent = {
            "requests": 2,
            "seconds": 600.0,
            "input_tokens": 9_000,
            "output_tokens": 2_000,
            "cost_usd": 0.0425,
            "estimated": False,
        }
        raise IncompleteTranscriptionError(
            "No connection.", completed=2, total=3, spent=spent
        )

    monkeypatch.setattr(transcribe, "transcribe_openai", stops_early)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)

    assert any(
        "💵 $0.04" in c.value and "spent so far, not saved to history" in c.value
        for c in test.caption
    )
    assert db.list_transcriptions() == []


def test_history_keeps_the_estimate_mark_and_says_local_runs_are_free():
    import json

    db.init_db()
    db.add_transcription(
        "a.wav",
        "audio",
        "gpt-4o-transcribe",
        False,
        "t",
        cost_usd=0.03,
        usage_json=json.dumps({"provider": config.PROVIDER_OPENAI, "estimated": True}),
    )
    db.add_transcription(
        "b.wav",
        "audio",
        "base",
        False,
        "t",
        cost_usd=0.0,
        usage_json=json.dumps({"provider": config.PROVIDER_LOCAL, "estimated": False}),
    )
    texts = {r["filename"]: app._history_cost(r) for r in db.list_transcriptions()}

    assert texts == {"a.wav": "≈ $0.03", "b.wav": "free (local)"}


_ESTIMATE_SCRIPT = """
import streamlit as st
import app

app.init_session_state()
st.session_state.video_path = st.session_state.video
visual = app.render_visual_options("video")
if visual and st.session_state.get("collect"):
    app.collect_visual_notes(visual, lambda info: None, [], stop_on_account_error=True)
"""


def test_the_count_shown_before_the_run_is_the_count_described(tmp_path, monkeypatch):
    import frames
    import vision

    video = tmp_path / "meeting.mkv"
    video.write_bytes(b"not decoded: the scan is faked")
    # Ten minutes: a distinct picture every 5 s for the first two, then a static
    # screen, plus two cuts later on.
    entries = [
        {
            "time": float(t),
            "path": video,
            "scene": t in (0, 300, 450),
            "hash": (t + 1) * 7919 if t < 120 or t in (300, 450) else 1,
        }
        for t in range(0, 600, 5)
    ]
    scan = {"duration": 600.0, "width": 1920, "height": 1080, "frames": entries}
    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: scan)
    described = []
    monkeypatch.setattr(
        vision,
        "describe_keyframes",
        lambda keyframes, *a, **k: described.append(len(keyframes)) or [],
    )

    test = AppTest.from_string(_ESTIMATE_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["video"] = video
    test.run()
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(60).run()
    caption = next(c.value for c in test.caption if "will be described" in c.value)
    shown = int(caption.split("**")[1])

    test.session_state["collect"] = True
    test.run()

    assert shown == len(frames.select_from_scan(scan, 60.0))
    assert described == [shown]


def test_a_run_describes_a_screenshot_that_shares_a_picture_with_its_own(
    tmp_path, monkeypatch
):
    from PIL import Image, ImageDraw

    import frames
    import vision

    video = tmp_path / "deck.mkv"
    video.write_bytes(b"not decoded: the scan is faked")
    slide_one, slide_two = tmp_path / "frame_00001.jpg", tmp_path / "slide_two.jpg"
    for path, bars in ((slide_one, 2), (slide_two, 5)):
        image = Image.new("L", (160, 90), 0)
        for bar in range(bars):
            ImageDraw.Draw(image).rectangle((bar * 30, 0, bar * 30 + 12, 89), 255)
        image.save(path)
    scan = {
        "duration": 125.0,
        "width": 1920,
        "height": 1080,
        "frames": [
            {
                "time": 0.0,
                "path": slide_one,
                "scene": True,
                "hash": frames.dhash(slide_one),
                "shared": False,
            },
            # A different slide whose JPEG the scan shared with slide one.
            {
                "time": 120.0,
                "path": slide_one,
                "scene": True,
                "hash": frames.dhash(slide_two),
                "shared": True,
            },
        ],
    }
    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: scan)
    extracted = []

    def extract(video_path, time, output):
        extracted.append(time)
        output.write_bytes(slide_two.read_bytes())
        return output

    monkeypatch.setattr(frames, "extract_frame", extract)
    sent = []

    def describe(keyframes, *args, picture=None, **kwargs):
        sent.extend(picture(frame).read_bytes() for frame in keyframes)
        return []

    monkeypatch.setattr(vision, "describe_keyframes", describe)

    test = AppTest.from_string(_ESTIMATE_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["video"] = video
    test.session_state["collect"] = True
    test.run()
    test.checkbox[0].set_value(True).run()

    assert not test.exception
    assert extracted == [120.0]
    assert sent == [slide_one.read_bytes(), slide_two.read_bytes()]


def test_when_the_cap_binds_the_caption_and_the_run_use_the_whole_limit(
    tmp_path, monkeypatch
):
    import frames
    import vision

    video = tmp_path / "busy.mkv"
    video.write_bytes(b"not decoded: the scan is faked")
    # 17 minutes of a different picture every 5 s.
    scan = {
        "duration": 1020.0,
        "width": 1920,
        "height": 1080,
        "frames": [
            {"time": float(t), "path": video, "scene": t == 0, "hash": (t + 1) * 7919}
            for t in range(0, 1020, 5)
        ],
    }
    monkeypatch.setattr(frames, "scan_video", lambda *a, **k: scan)
    described = []
    monkeypatch.setattr(
        vision,
        "describe_keyframes",
        lambda keyframes, *a, **k: described.append(len(keyframes)) or [],
    )

    test = AppTest.from_string(_ESTIMATE_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["video"] = video
    test.run()
    test.checkbox[0].set_value(True).run()
    test.slider[0].set_value(5).run()
    caption = next(c.value for c in test.caption if "will be described" in c.value)

    test.session_state["collect"] = True
    test.run()

    assert "would give 204 screenshots, over the 200-screenshot limit" in caption
    assert "**200** will be described" in caption
    assert described == [200]


def test_start_uses_the_settings_on_the_page_not_those_of_the_run_before(monkeypatch):
    # Start clicked in the same moment as a change (or while the page was still
    # busy scanning) used to run with the previous run's settings.
    models = []

    def fake(input_file, output_file, api_key, **kwargs):
        models.append(kwargs["model"])
        Path(output_file).write_text("Said.", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")

    [model] = [s for s in test.selectbox if s.label == "Transcription model"]
    model.set_value("whisper-1")
    _start_button(test).click().run()

    assert models == ["whisper-1"]


def test_a_run_stopped_while_the_page_is_drawn_does_not_restart_later(
    fake_openai, monkeypatch
):
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    original = app.render_history_tab

    def stopped(*args, **kwargs):
        raise RuntimeError("stands in for Stop, before the job started")

    # The Start click's run ends before it reaches the job.
    monkeypatch.setattr(app, "render_history_tab", stopped)
    _start_button(test).click().run()
    monkeypatch.setattr(app, "render_history_tab", original)
    test.run()

    assert fake_openai == []  # the stopped job is not run by the next rerun
    assert test.session_state.job is None
    assert any("stopped before it finished" in w.value for w in test.warning)
    assert not _start_button(test).disabled


def test_a_first_chunk_that_failed_after_paying_says_what_it_spent(monkeypatch):
    def fails(*args, **kwargs):
        error = TranscriptionError("Cannot reach OpenAI.")
        error.spent = {
            "requests": 2,
            "seconds": 450.0,
            "input_tokens": 2_000,
            "output_tokens": 2_148,
            "cost_usd": 0.026,
            "estimated": False,
        }
        raise error

    monkeypatch.setattr(transcribe, "transcribe_openai", fails)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    _upload(test, "talk.wav", b"talk audio")
    _start(test)

    assert any("$0.03" in i.value and "spent so far" in i.value for i in test.info)
