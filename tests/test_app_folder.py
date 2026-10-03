"""Choosing a recording from a folder on this computer, in the page (no network).

The file is read where it lies: nothing goes through the browser or into
``uploads/`` (a 372.6 MB upload failed with MemoryError, 2026-09-29).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
import audio
import config
import db
import recordings
import transcribe

_APP = """
import app
app.main()
"""
_TIMEOUT = 30


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the OpenAI pipeline; list the files it was given."""
    sent = []

    def fake(input_file, output_file, api_key, **kwargs):
        sent.append(Path(input_file))
        Path(output_file).write_text("Transcript of the call.", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)
    return sent


@pytest.fixture
def folder(tmp_path):
    """A recordings folder with one call, one video and a file that is neither."""
    recordings = tmp_path / "Recordings"
    recordings.mkdir()
    (recordings / "call.wav").write_bytes(b"call audio")
    (recordings / "meeting.mkv").write_bytes(b"video")
    (recordings / "notes.txt").write_text("not a recording", encoding="utf-8")
    return recordings


def _in_folder_mode(folder: Path) -> AppTest:
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.radio(key="source").set_value(config.SOURCE_FOLDER).run()
    return test.text_input(key="recordings_folder").input(str(folder)).run()


def _pick(test: AppTest, path: Path) -> AppTest:
    return test.selectbox(key="folder_file").set_value(str(path)).run()


def _start(test: AppTest) -> AppTest:
    button = next(b for b in test.button if b.label == "Start Transcription")
    return button.click().run()


def test_a_recording_from_a_folder_is_transcribed_where_it_lies(folder, fake_openai):
    test = _in_folder_mode(folder)
    listed = test.selectbox(key="folder_file")
    # The options as shown: name · creation time.
    assert sorted(option.split(" · ")[0] for option in listed.options) == [
        "call.wav",
        "meeting.mkv",
    ]
    # Nothing is chosen, so nothing is prepared, until the owner picks a file.
    assert listed.value is None
    assert test.session_state.audio_path is None

    _pick(test, folder / "call.wav")
    _start(test)

    assert not test.exception
    assert fake_openai == [folder / "call.wav"]
    assert test.session_state.audio_path == folder / "call.wav"
    assert list(config.get_settings().upload_dir.iterdir()) == []  # not copied
    [row] = db.list_transcriptions()
    assert row["filename"] == "call.wav"
    assert any("read from disk, not uploaded" in c.value for c in test.caption)


def test_a_video_from_a_folder_has_its_audio_extracted_from_the_original(
    folder, fake_openai, monkeypatch
):
    extracted = []

    def fake_to_wav(source, output):
        extracted.append(Path(source))
        Path(output).write_bytes(b"wav")
        return Path(output)

    monkeypatch.setattr(audio, "to_wav", fake_to_wav)
    monkeypatch.setattr(app, "_make_preview", lambda source, stem: None)

    test = _pick(_in_folder_mode(folder), folder / "meeting.mkv")

    assert extracted == [folder / "meeting.mkv"]
    # On-screen context reads frames from the original video.
    assert test.session_state.video_path == folder / "meeting.mkv"
    assert (folder / "meeting.mkv").read_bytes() == b"video"


def test_the_source_and_the_folder_are_kept_for_the_next_session(folder):
    _in_folder_mode(folder)

    assert db.get_preferences()["source"] == config.SOURCE_FOLDER
    assert db.get_preferences()["recordings_folder"] == str(folder)
    later = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    assert later.radio(key="source").value == config.SOURCE_FOLDER
    assert later.text_input(key="recordings_folder").value == str(folder)


def test_a_pasted_recording_path_chooses_that_recording(folder):
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.radio(key="source").set_value(config.SOURCE_FOLDER).run()

    test.text_input(key="recordings_folder").input(f'"{folder / "call.wav"}"').run()

    assert test.text_input(key="recordings_folder").value == str(folder)
    assert test.selectbox(key="folder_file").value == str(folder / "call.wav")
    assert test.session_state.original_filename == "call.wav"


def test_a_folder_that_does_not_exist_is_reported(tmp_path):
    test = _in_folder_mode(tmp_path / "missing")

    assert any("no such folder" in w.value for w in test.warning)
    assert "folder_file" not in [s.key for s in test.selectbox]


def test_cleaning_up_forgets_the_pick_but_keeps_the_recording(folder, fake_openai):
    test = _pick(_in_folder_mode(folder), folder / "call.wav")

    clean = next(b for b in test.button if b.label == "Clean temporary files")
    clean.click().run()

    assert test.selectbox(key="folder_file").value is None
    assert test.session_state.audio_path is None
    assert (folder / "call.wav").read_bytes() == b"call audio"


def test_without_local_files_only_the_uploader_is_offered(folder, monkeypatch):
    # For a copy others can open: it must not show its server's disk.
    monkeypatch.setenv("ALLOW_LOCAL_FILES", "false")
    config.get_settings.cache_clear()
    db.init_db()
    db.set_preference("source", config.SOURCE_FOLDER)

    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()

    assert "source" not in [r.key for r in test.radio]
    assert not test.text_input
    assert len(test.file_uploader) == 1


def test_a_recording_that_changes_on_disk_keeps_the_result_and_says_so(
    folder, fake_openai
):
    # Found by review: its identity included size and time, so a recording
    # still being written wiped the finished transcript on the next click.
    test = _start(_pick(_in_folder_mode(folder), folder / "call.wav"))
    transcript = test.session_state.transcript_path
    assert transcript is not None

    with (folder / "call.wav").open("ab") as recording:
        recording.write(b" and more")
    test.run()

    assert test.session_state.transcript_path == transcript
    assert any("changed on disk" in w.value for w in test.warning)
    reload = next(b for b in test.button if b.label == "↻ Load it again")

    reload.click().run()

    assert test.session_state.transcript_path is None
    assert test.session_state.audio_path == folder / "call.wav"
    assert not any("changed on disk" in w.value for w in test.warning)


_CHOSEN_EARLIER_SCRIPT = """
import streamlit as st
import app

app.init_session_state()
st.session_state.recordings_folder = st.session_state.folder
st.session_state.folder_file = st.session_state.chosen
app._choose_from_folder(False)
"""


def test_a_chosen_recording_that_is_gone_is_reported(folder):
    # Found by review: a pick that vanished (moved, renamed) was forgotten
    # silently, and a Start in that moment did nothing. (Set up directly:
    # AppTest cannot re-send a list value whose file is gone.)
    (folder / "call.wav").unlink()
    test = AppTest.from_string(_CHOSEN_EARLIER_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["folder"] = str(folder)
    test.session_state["chosen"] = str(folder / "call.wav")

    test.run()

    assert test.selectbox(key="folder_file").value is None
    assert any("call.wav is no longer in this folder" in w.value for w in test.warning)


def test_a_pick_typed_in_another_letter_case_is_kept(folder):
    test = AppTest.from_string(_CHOSEN_EARLIER_SCRIPT, default_timeout=_TIMEOUT)
    test.session_state["folder"] = str(folder)
    test.session_state["chosen"] = str(folder / "CALL.WAV")

    test.run()

    if Path("A") == Path("a"):  # Windows: names ignore case
        assert test.selectbox(key="folder_file").value == str(folder / "call.wav")
        assert not test.warning


def test_the_app_s_own_working_folder_is_refused(folder):
    # "Clean temporary files" would delete a recording picked there.
    uploads = config.get_settings().upload_dir
    (uploads / "old upload.wav").write_bytes(b"x")

    test = _in_folder_mode(uploads)

    assert any("app's own working folders" in w.value for w in test.warning)
    assert "folder_file" not in [s.key for s in test.selectbox]


def test_a_folder_that_cannot_be_read_is_not_called_empty(folder, monkeypatch):
    import recordings

    def refused(folder):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(recordings, "list_recordings", refused)

    test = _in_folder_mode(folder)

    assert any("cannot be read: Access is denied" in w.value for w in test.warning)
    assert not any("No video or audio" in i.value for i in test.info)


def test_a_relative_path_is_not_taken_as_a_folder(folder, monkeypatch):
    monkeypatch.chdir(folder.parent)

    test = _in_folder_mode(Path(folder.name))

    assert any("full path" in w.value for w in test.warning)


def test_the_folder_source_is_off_when_others_can_reach_the_page(folder, monkeypatch):
    # Started with --server.address=0.0.0.0 (Docker does), anyone who opens the
    # page could browse this disk and transcribe it with the owner's key.
    import streamlit as st

    real = st.get_option
    monkeypatch.setattr(
        st,
        "get_option",
        lambda key: "192.168.1.10" if key == "server.address" else real(key),
    )
    db.init_db()
    db.set_preference("source", config.SOURCE_FOLDER)

    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()

    assert "source" not in [r.key for r in test.radio]
    assert len(test.file_uploader) == 1


def test_a_recording_that_grows_during_the_run_leaves_no_saved_parts(
    folder, monkeypatch
):
    # The saved parts are named after the audio's content when the run starts;
    # hashing it again at the end (after it grew) missed them.
    import checkpoints

    saved = []

    def fake(input_file, output_file, api_key, **kwargs):
        digest = checkpoints.file_digest(Path(input_file))
        parts = transcribe.checkpoint_dir(digest, kwargs["model"])
        parts.mkdir(parents=True, exist_ok=True)
        (parts / "chunk_000.json").write_text("{}", encoding="utf-8")
        saved.append(parts)
        with Path(input_file).open("ab") as recording:
            recording.write(b" still recording")
        Path(output_file).write_text("Transcript.", encoding="utf-8")

    monkeypatch.setattr(transcribe, "transcribe_openai", fake)

    test = _start(_pick(_in_folder_mode(folder), folder / "call.wav"))

    assert not test.exception
    [parts] = saved
    assert not parts.exists()
    assert len(db.list_transcriptions()) == 1


# --- 📂 Browse… --------------------------------------------------------------


def _choose_with_window(test: AppTest) -> AppTest:
    button = next(b for b in test.button if b.key == "pick_file")
    return button.click().run()


def test_a_file_chosen_in_the_window_is_picked_and_its_folder_kept(folder, monkeypatch):
    # The owner keeps recordings in different folders: the window goes to any
    # of them, and the list then shows that folder.
    import recordings

    opened_in = []

    def window(start):
        opened_in.append(start)
        return folder / "meeting.mkv"

    monkeypatch.setattr(recordings, "ask_for_recording", window)

    def fake_to_wav(source, output):
        Path(output).write_bytes(b"wav")
        return Path(output)

    monkeypatch.setattr(audio, "to_wav", fake_to_wav)
    monkeypatch.setattr(app, "_make_preview", lambda source, stem: None)
    test = AppTest.from_string(_APP, default_timeout=_TIMEOUT).run()
    test.radio(key="source").set_value(config.SOURCE_FOLDER).run()

    _choose_with_window(test)

    assert not test.exception
    assert opened_in == [None]  # no folder yet: the window's own default
    assert test.text_input(key="recordings_folder").value == str(folder)
    assert test.selectbox(key="folder_file").value == str(folder / "meeting.mkv")
    assert test.session_state.video_path == folder / "meeting.mkv"
    assert db.get_preferences()["recordings_folder"] == str(folder)

    _choose_with_window(test)
    assert opened_in[-1] == folder  # next time it opens where the last one was


def test_a_cancelled_window_changes_nothing(folder, monkeypatch):
    import recordings

    monkeypatch.setattr(recordings, "ask_for_recording", lambda start: None)
    test = _pick(_in_folder_mode(folder), folder / "call.wav")

    _choose_with_window(test)

    assert test.text_input(key="recordings_folder").value == str(folder)
    assert test.selectbox(key="folder_file").value == str(folder / "call.wav")
    assert not test.warning


def test_a_window_that_cannot_open_says_why(folder, monkeypatch):
    import recordings
    from exceptions import FilePickerError

    def busy(start):
        raise FilePickerError("A file window is already open — it may be behind.")

    monkeypatch.setattr(recordings, "ask_for_recording", busy)
    test = _in_folder_mode(folder)

    _choose_with_window(test)

    assert any("already open" in w.value for w in test.warning)
    assert test.text_input(key="recordings_folder").value == str(folder)


def test_a_file_that_is_not_a_recording_is_not_picked(folder, monkeypatch):
    import recordings

    monkeypatch.setattr(
        recordings, "ask_for_recording", lambda start: folder / "notes.txt"
    )
    test = _in_folder_mode(folder)

    _choose_with_window(test)

    assert any("not a video or audio file" in w.value for w in test.warning)
    assert test.selectbox(key="folder_file").value is None


def test_a_browse_click_that_lands_with_start_opens_no_window(folder, monkeypatch):
    # Found by review: the click is handled when the folder field is drawn,
    # which in the job's run would block the job behind the file window.
    import threading

    import recordings

    opened = []
    monkeypatch.setattr(
        recordings, "ask_for_recording", lambda start: opened.append(start)
    )
    test = _in_folder_mode(folder)
    release = threading.Event()
    worker = threading.Thread(target=release.wait)
    worker.start()
    try:
        test.session_state["job"] = {"params": None, "worker": worker}
        test.session_state["pick_pending"] = True
        test.run()
    finally:
        release.set()
        worker.join()

    assert opened == []
    assert "pick_pending" not in test.session_state


def test_a_browse_click_overtaken_by_upload_mode_does_not_come_back(
    folder, monkeypatch
):
    import recordings

    opened = []
    monkeypatch.setattr(
        recordings, "ask_for_recording", lambda start: opened.append(start)
    )
    test = _in_folder_mode(folder)
    test.radio(key="source").set_value(config.SOURCE_UPLOAD)
    test.session_state["pick_pending"] = True
    test.run()

    test.radio(key="source").set_value(config.SOURCE_FOLDER).run()

    assert opened == []


@pytest.fixture
def lengths(monkeypatch):
    """Recording lengths as ffprobe would read them; none known at first."""
    known: dict[Path, float] = {}
    monkeypatch.setattr(
        recordings, "durations", lambda paths: {path: known.get(path) for path in paths}
    )
    return known


def test_the_list_shows_each_recording_s_length_first(folder, lengths):
    lengths[folder / "meeting.mkv"] = 3_735.0

    listed = _in_folder_mode(folder).selectbox(key="folder_file").options

    by_name = {option.split(" · ")[-2]: option for option in listed}
    # First, because long recorder names are cut off at the end of the list.
    assert by_name["meeting.mkv"].startswith("1:02:15 · meeting.mkv · ")
    # Unknown (still being recorded, or unreadable): just name and date.
    assert by_name["call.wav"].startswith("call.wav · ")
    assert len(listed) == 2


def _send_held(test: AppTest, held: str) -> AppTest:
    """Run with the text the browser holds as the list's value.

    AppTest re-sends the list's value in its current wording; a browser sends
    the text it was last given, which is the old one after a change of the
    entry unless the page sends the value again. Uses AppTest internals of
    Streamlit 1.56.
    """
    states = test._tree.get_widget_states()
    box = test.selectbox(key="folder_file")
    for state in states.widgets:
        if state.id == box.id:
            state.string_value = held
    return test._run(states)


def test_the_choice_stays_when_its_length_appears(folder, lengths):
    # Picked while still being recorded (no length yet); finishing it adds the
    # length to its entry. Live, the next click but one forgot the choice.
    test = _pick(_in_folder_mode(folder), folder / "call.wav")
    box = test.selectbox(key="folder_file")
    held = box.options[box.index]
    lengths[folder / "call.wav"] = 2_832.0

    for _ in range(3):  # three clicks elsewhere on the page
        test = _send_held(test, held)
        box = test.selectbox(key="folder_file")
        if box.proto.set_value:
            held = box.proto.raw_value

    assert box.value == str(folder / "call.wav")
    assert held.startswith("47:12 · call.wav · ")
    assert not test.warning


def test_cleaning_up_is_offered_in_the_sidebar_with_the_space_it_frees(folder):
    test = _pick(_in_folder_mode(folder), folder / "call.wav")
    uploads = config.get_settings().upload_dir
    (uploads / "old call.amr").write_bytes(b"a" * 300_000)

    test.run()

    assert [b.label for b in test.sidebar.button] == ["Clean temporary files"]
    assert not [b for b in test.main.button if "Clean" in b.label]
    assert "**0.3 MB** in use." in [c.value for c in test.sidebar.caption]

    test.sidebar.button[0].click().run()

    assert not (uploads / "old call.amr").exists()
    assert "Nothing to clean." in [c.value for c in test.sidebar.caption]
    assert any("cleaned" in s.value for s in test.sidebar.success)


def test_a_recording_picked_from_a_list_drawn_before_its_length_stays_chosen(
    folder, lengths
):
    # Found by review: the list was drawn while the call was still recording;
    # it finished, and the owner picked it from that list. The page had only
    # remembered the chosen entry, so the next click (Start) lost the choice.
    test = _in_folder_mode(folder)
    box = test.selectbox(key="folder_file")
    held = next(option for option in box.options if " call.wav · " in f" {option}")
    lengths[folder / "call.wav"] = 2_832.0

    for _ in range(3):  # the pick, then two clicks elsewhere
        test = _send_held(test, held)
        box = test.selectbox(key="folder_file")
        if box.proto.set_value:
            held = box.proto.raw_value

    assert box.value == str(folder / "call.wav")
    assert not test.warning


def test_only_the_newest_recordings_are_given_a_length(folder, monkeypatch):
    # A phone's call folder can hold thousands of files; reading each one's
    # length would hold up the page.
    asked = []
    monkeypatch.setattr(
        recordings, "durations", lambda paths: asked.append(paths) or {}
    )
    monkeypatch.setattr(app, "RECORDING_PROBE_LIMIT", 1)

    test = _in_folder_mode(folder)

    assert asked and all(len(paths) == 1 for paths in asked)
    assert len(test.selectbox(key="folder_file").options) == 2


def test_the_space_shown_includes_what_this_click_prepared(folder, monkeypatch):
    def fake_to_wav(source, output):
        Path(output).write_bytes(b"w" * 400_000)
        return Path(output)

    monkeypatch.setattr(audio, "to_wav", fake_to_wav)
    monkeypatch.setattr(app, "_make_preview", lambda source, stem: None)

    test = _pick(_in_folder_mode(folder), folder / "meeting.mkv")

    assert "**0.4 MB** in use." in [c.value for c in test.sidebar.caption]
