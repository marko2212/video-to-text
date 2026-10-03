"""Tests for the pure helpers in key-frame extraction (no ffmpeg, no network)."""

import itertools
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

import config
import frames
from exceptions import VisualContextError


def _write_image(path, colour, box=None):
    image = Image.new("RGB", (160, 120), colour)
    if box:
        Image.Image.paste(image, Image.new("RGB", (40, 30), box), (10, 10))
    image.save(path)
    return path


def test_parse_frame_times_reads_showinfo_lines():
    stderr = (
        "[Parsed_showinfo_1 @ 0000022f] n:   0 pts:  20480 pts_time:2       "
        "duration:   1024 fmt:yuv420p\n"
        "[Parsed_showinfo_1 @ 0000022f] n:   1 pts:  61440 pts_time:6.3     "
        "duration:   1024 fmt:yuv420p\n"
    )
    assert frames._parse_frame_times(stderr) == [2.0, 6.3]


def test_parse_frame_times_ignores_lines_from_other_filters():
    # ffmpeg logs this at verbose level; matching it would shift every frame.
    stderr = (
        "[graph -1 input from stream 0:0 @ 000001] video frame properties "
        "congruent with link at pts_time: 0\n"
        "[Parsed_showinfo_1 @ 0000022f] n:   0 pts:  0 pts_time:4 fmt:yuv420p\n"
        "config in time_base: 1/10240, frame_rate: 10/1\n"
    )
    assert frames._parse_frame_times(stderr) == [4.0]


def test_parse_frame_times_without_matches():
    assert frames._parse_frame_times("nothing useful here") == []


def test_effective_interval_tightens_for_short_videos():
    # A 25 s clip would otherwise get a single sample from a 30 s interval.
    assert frames.effective_interval(25.0, 30.0) == 12.5


def test_effective_interval_honours_the_chosen_cadence():
    assert frames.effective_interval(3600.0, 30.0) == 30.0
    assert frames.effective_interval(3600.0, 300.0) == 300.0


def test_effective_interval_widens_when_the_cap_would_bind():
    # 30 minutes every 10 s is 180 frames; a 60-frame cap means one every 30 s.
    assert frames.effective_interval(1800.0, 10.0, max_frames=60) == 30.0


def test_effective_interval_never_goes_below_the_floor():
    assert frames.effective_interval(4.0, 30.0) == 5.0


def test_effective_interval_falls_back_when_duration_is_unknown():
    assert frames.effective_interval(0.0, 30.0) == 30.0


def test_estimate_frame_count_divides_duration_by_the_interval():
    # A frame at t=0 and then one per interval: 600 / 30 = 20, plus the first.
    assert frames.estimate_frame_count(600.0, 30.0) == 21


def test_estimate_frame_count_respects_the_hard_cap():
    cap = frames.max_frames_setting()
    assert frames.estimate_frame_count(36000.0, 10.0) == cap


def test_the_cap_is_read_from_the_environment(monkeypatch):
    import config

    monkeypatch.setenv("FRAME_MAX_COUNT", "300")
    config.get_settings.cache_clear()

    assert frames.max_frames_setting() == 300
    # A cap read at call time is useless if the callers baked in the default.
    assert frames.estimate_frame_count(36000.0, 10.0) == 300
    assert frames.effective_interval(3000.0, 5.0) == 10.0


def test_estimate_frame_count_tracks_the_slider_on_a_long_meeting():
    # The regression this guards: with too tight a cap every interval collapsed
    # to the same number, so moving the slider changed nothing on screen.
    counts = [frames.estimate_frame_count(4992.0, i) for i in (30.0, 40.0, 120.0)]
    assert counts == [167, 125, 42]
    assert len(set(counts)) == len(counts)


def test_estimate_frame_count_uses_the_tightened_short_video_interval():
    # 25 s at a 30 s setting really samples every 12.5 s, so 3 — not 1.
    assert frames.estimate_frame_count(25.0, 30.0) == 3


def test_estimate_frame_count_is_zero_when_duration_is_unknown():
    assert frames.estimate_frame_count(0.0, 30.0) == 0


def test_estimate_frame_count_falls_as_the_interval_grows():
    counts = [frames.estimate_frame_count(1800.0, i) for i in (30.0, 120.0, 300.0)]
    assert counts == sorted(counts, reverse=True)


def test_apply_min_interval_drops_closely_spaced_frames():
    candidates = [
        {"time": 0.0},
        {"time": 0.5},
        {"time": 1.0},
        {"time": 4.0},
    ]
    kept = frames.apply_min_interval(candidates, 2.0)
    # One frame per burst — the last, settled one.
    assert [frame["time"] for frame in kept] == [1.0, 4.0]


def test_apply_min_interval_measures_a_burst_from_its_first_frame():
    # 2.5 is 2.5 s after the burst began at 0.0, so it starts a new burst:
    # continuous motion still yields a frame every couple of seconds.
    candidates = [{"time": 0.0}, {"time": 1.5}, {"time": 2.5}]
    kept = frames.apply_min_interval(candidates, 2.0)
    assert [frame["time"] for frame in kept] == [1.5, 2.5]


def test_a_slide_change_right_after_an_interval_sample_is_kept():
    # Measured on a real extraction: an interval sample at 39.67 s still showed
    # slide 3, the cut to slide 4 came at 41.53 s. Keeping the first of the two
    # stamped slide 4 a whole interval late.
    candidates = [
        {"time": 29.67, "slide": 3},
        {"time": 39.67, "slide": 3},
        {"time": 41.53, "slide": 4},
    ]
    kept = frames.apply_min_interval(candidates, 2.0)
    assert [(f["time"], f["slide"]) for f in kept] == [(29.67, 3), (41.53, 4)]


def test_cap_frame_count_keeps_everything_under_the_cap():
    candidates = [{"time": float(index)} for index in range(3)]
    assert frames.cap_frame_count(candidates, 10) == candidates


def test_cap_frame_count_spreads_samples_across_the_timeline():
    candidates = [{"time": float(index)} for index in range(10)]
    kept = frames.cap_frame_count(candidates, 4)
    assert [frame["time"] for frame in kept] == [0.0, 3.0, 6.0, 9.0]


def test_cap_frame_count_handles_degenerate_limits():
    candidates = [{"time": 0.0}, {"time": 1.0}]
    assert frames.cap_frame_count(candidates, 0) == []
    assert frames.cap_frame_count(candidates, 1) == [candidates[0]]


def test_hamming_distance():
    assert frames.hamming_distance(0b1011, 0b1011) == 0
    assert frames.hamming_distance(0b1011, 0b1000) == 2


def test_dhash_is_stable_for_identical_images(tmp_path):
    first = _write_image(tmp_path / "a.png", (30, 60, 120), box=(240, 240, 0))
    second = _write_image(tmp_path / "b.png", (30, 60, 120), box=(240, 240, 0))
    assert frames.dhash(first) == frames.dhash(second)


def test_dhash_separates_different_pictures(tmp_path):
    plain = _write_image(tmp_path / "plain.png", (30, 60, 120))
    marked = _write_image(tmp_path / "marked.png", (30, 60, 120), box=(255, 255, 255))
    distance = frames.hamming_distance(frames.dhash(plain), frames.dhash(marked))
    assert distance > frames.FRAME_DUPLICATE_DISTANCE


def test_drop_near_duplicates_keeps_one_of_each_picture(tmp_path):
    candidates = [
        {"time": 0.0, "path": _write_image(tmp_path / "1.png", (10, 10, 10))},
        {"time": 5.0, "path": _write_image(tmp_path / "2.png", (10, 10, 10))},
        {
            "time": 9.0,
            "path": _write_image(tmp_path / "3.png", (10, 10, 10), box=(250, 250, 250)),
        },
    ]
    kept = frames.drop_near_duplicates(candidates)
    assert [frame["time"] for frame in kept] == [0.0, 9.0]


# --- the video scan and the exact selection -----------------------------------------


def _scan(entries, duration=None):
    """entries: (time, scene, hash) tuples."""
    frames_ = [
        {"time": t, "path": Path(f"f{i}.jpg"), "scene": scene, "hash": h}
        for i, (t, scene, h) in enumerate(entries)
    ]
    return {
        "duration": duration or (entries[-1][0] + 5),
        "width": 1920,
        "height": 1080,
        "frames": frames_,
    }


def _distinct(value):
    """A 64-bit hash far (in bits) from the hash of any other value."""
    return (value * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF


def _grid(seconds, step=5.0, hash_of=lambda t: 0):
    return [(float(t), t == 0, hash_of(t)) for t in range(0, seconds, int(step))]


def test_scene_scores_are_read_in_order():
    stderr = (
        "[Parsed_metadata_1 @ 0x1] frame:0 pts:0 pts_time:0\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.scene_score=0.000000\n"
        "[Parsed_metadata_1 @ 0x1] frame:1 pts:9 pts_time:14.866\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.scene_score=1.000000\n"
    )
    assert frames._parse_scene_scores(stderr) == [0.0, 1.0]


def test_a_static_screen_yields_one_screenshot_at_any_interval():
    scan = _scan(_grid(3600))  # an hour of the same picture
    for interval in (5, 30, 245):
        assert len(frames.select_from_scan(scan, interval, 300)) == 1


def test_every_distinct_screen_change_is_kept_whatever_the_interval():
    # Distinct grid frames every 5 s, plus two real cuts.
    entries = [(float(t), False, _distinct(t)) for t in range(0, 600, 5)]
    entries[0] = (0.0, True, _distinct(0))
    entries += [(122.0, True, _distinct(10_000)), (377.0, True, _distinct(20_000))]
    entries.sort()
    scan = _scan(entries)

    sparse = frames.select_from_scan(scan, 300, 300)
    times = [f["time"] for f in sparse]
    assert 122.0 in times and 377.0 in times
    dense = frames.select_from_scan(scan, 30, 300)
    assert len(dense) > len(sparse)


def test_the_interval_takes_the_first_grid_frame_after_it():
    scan = _scan([(float(t), t == 0, _distinct(t)) for t in range(0, 200, 5)])
    chosen = frames.select_from_scan(scan, 60, 300)
    assert [f["time"] for f in chosen] == [0.0, 60.0, 120.0, 180.0]


def test_the_cap_still_limits_a_scanned_selection():
    scan = _scan([(float(t), True, _distinct(t)) for t in range(0, 3000, 5)])
    assert len(frames.select_from_scan(scan, 5, 50)) == 50


def _striped(stripes, dot=False):
    """Flat pictures all hash alike; stripes of a given width do not."""
    image = Image.new("L", (64, 36), 0)
    for x in range(64):
        if stripes and (x // stripes) % 2:
            for y in range(36):
                image.putpixel((x, y), 255)
    if dot:  # a small change the 8x8 hash hardly sees
        for x in range(28, 34):
            for y in range(14, 20):
                image.putpixel((x, y), 128)
    return image


def _fake_ffmpeg(monkeypatch, pictures, calls, indexed=True, duration=None):
    """Replace the scan pass: write one JPEG per (time, score, picture).

    A picture is a stripe width or an image. ``indexed`` is what ffprobe says
    about the container; ``duration`` the length it reports.
    """

    def run_scan(video_path, output_dir, threshold, grid):
        calls.append(1)
        lines = []
        for index, (time, score, picture) in enumerate(pictures, start=1):
            image = picture if isinstance(picture, Image.Image) else _striped(picture)
            image.save(output_dir / f"frame_{index:05d}.jpg")
            lines.append(f"[Parsed_metadata_1 @ 0x1] frame:{index} pts_time:{time}")
            lines.append(f"[Parsed_metadata_1 @ 0x1] lavfi.scene_score={score:.6f}")
            lines.append(f"[Parsed_showinfo_2 @ 0x2] n:{index} pts_time:{time} fmt:yuv")
        return "\n".join(lines)

    monkeypatch.setattr(frames, "_run_scan", run_scan)
    length = pictures[-1][0] + 5 if duration is None else duration
    monkeypatch.setattr(
        frames,
        "_probe",
        lambda path: {"has_video": True, "duration": length, "indexed": indexed},
    )


def test_a_scan_is_stored_once_and_repeated_pictures_share_one_file(
    tmp_path, monkeypatch
):
    calls = []
    flat, striped = 0, 8
    _fake_ffmpeg(
        monkeypatch,
        [(0.0, 0.0, flat), (5.0, 0.0, flat), (9.5, 0.9, striped), (14.5, 0.0, striped)],
        calls,
    )
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    scan_dir = config.get_settings().temp_dir / "scan-abc"

    scan = frames.scan_video(video, scan_dir)

    assert [f["scene"] for f in scan["frames"]] == [False, False, True, False]
    # Same picture, same file: only two images are kept on disk.
    assert len(list(scan_dir.glob("frame_*.jpg"))) == 2
    assert scan["frames"][1]["path"] == scan["frames"][0]["path"]
    assert [f["shared"] for f in scan["frames"]] == [False, True, False, True]
    assert (scan["width"], scan["height"]) == (64, 36)

    again = frames.scan_video(video, scan_dir)
    assert calls == [1]  # reused, not rescanned
    assert len(again["frames"]) == 4


def test_an_incomplete_scan_is_redone(tmp_path, monkeypatch):
    calls = []
    _fake_ffmpeg(monkeypatch, [(0.0, 0.0, 0)], calls)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    scan_dir = config.get_settings().temp_dir / "scan-abc"
    scan_dir.mkdir(parents=True)
    (scan_dir / "frame_00001.jpg").write_bytes(b"left over, no index")

    frames.scan_video(video, scan_dir)
    assert calls == [1]
    index = json.loads((scan_dir / "index.json").read_text(encoding="utf-8"))
    assert len(index["frames"]) == 1


def test_the_cap_spreads_frames_at_the_chosen_interval_instead_of_skipping_grid_steps():
    # 17 minutes at slider 5 s with a cap of 200: widening the interval to 5.1 s
    # snapped it to the next 5 s grid frame, 10 s later — 102 screenshots.
    scan = _scan([(float(t), t == 0, _distinct(t)) for t in range(0, 1020, 5)])
    chosen = frames.select_from_scan(scan, 5, 200)
    assert len(chosen) == 200
    gaps = [b["time"] - a["time"] for a, b in itertools.pairwise(chosen)]
    assert sum(gap == 5.0 for gap in gaps) >= 190


def test_a_scan_made_with_other_settings_is_redone(tmp_path, monkeypatch):
    calls = []
    _fake_ffmpeg(monkeypatch, [(0.0, 0.0, 0), (5.0, 0.0, 8)], calls)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    scan_dir = config.get_settings().temp_dir / "scan-abc"

    frames.scan_video(video, scan_dir, threshold=0.1)
    frames.scan_video(video, scan_dir, threshold=0.1)
    frames.scan_video(video, scan_dir, threshold=0.3)
    assert calls == [1, 1]


def test_a_frame_that_shares_a_picture_is_described_with_its_own(tmp_path, monkeypatch):
    # Three slides of one template: S1 and S2 differ, but their hashes do not.
    s0, s1, s2 = _striped(0), _striped(8), _striped(8, dot=True)
    hashes = [
        frames.dhash(_saved(tmp_path, name, s)) for name, s in (("1", s1), ("2", s2))
    ]
    assert frames.hamming_distance(*hashes) <= config.FRAME_DUPLICATE_DISTANCE
    timeline = [(float(t), 0.0, s0) for t in range(0, 65, 5)]
    timeline += [(float(t), 0.0, s1) for t in range(65, 95, 5)]
    timeline += [(float(t), 0.0, s2) for t in range(95, 130, 5)]
    _fake_ffmpeg(monkeypatch, timeline, [])
    extracted = []

    def extract(video_path, time, output):
        extracted.append(time)
        s2.save(output)  # what the video shows at that time
        return output

    monkeypatch.setattr(frames, "extract_frame", extract)
    video = tmp_path / "deck.mp4"
    video.write_bytes(b"video")
    scan = frames.scan_video(video, config.get_settings().temp_dir / "scan-deck")

    chosen = frames.select_from_scan(scan, 60, 300)
    assert [f["time"] for f in chosen] == [0.0, 120.0]
    late = chosen[1]
    assert late["shared"]  # the scan kept S1's JPEG for it

    with frames.pictures(video) as picture:
        own = picture(late)
        assert own.read_bytes() != Path(late["path"]).read_bytes()  # not S1
        assert picture(chosen[0]) == chosen[0]["path"]  # not shared: its own file
        folder = own.parent
    assert extracted == [120.0]
    assert not folder.exists()  # extracted pictures are removed after the run


def _saved(tmp_path, name, image):
    path = tmp_path / f"{name}.jpg"
    image.save(path)
    return path


def test_a_container_without_an_index_keeps_every_frame_s_own_picture(
    tmp_path, monkeypatch
):
    # Seeking in MPEG-TS/PS landed up to a keyframe interval late, so a shared
    # frame would have been re-extracted as a later screen.
    _fake_ffmpeg(monkeypatch, [(0.0, 0.0, 0), (5.0, 0.0, 0)], [], indexed=False)
    video = tmp_path / "call.ts"
    video.write_bytes(b"video")
    scan = frames.scan_video(video, config.get_settings().temp_dir / "scan-ts")

    assert [f["shared"] for f in scan["frames"]] == [False, False]
    assert scan["frames"][0]["path"] != scan["frames"][1]["path"]


def test_a_file_without_a_picture_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(
        frames,
        "_probe",
        lambda path: {"has_video": False, "duration": 30.0, "indexed": True},
    )
    video = tmp_path / "voice.3gp"
    video.write_bytes(b"audio only")
    with pytest.raises(VisualContextError, match="no picture"):
        frames.scan_video(video, config.get_settings().temp_dir / "scan-voice")


def test_a_video_without_a_length_uses_its_last_frame(tmp_path, monkeypatch):
    _fake_ffmpeg(
        monkeypatch, [(0.0, 0.0, 0), (5.0, 0.0, 8), (10.0, 0.0, 4)], [], duration=0.0
    )
    video = tmp_path / "browser.webm"
    video.write_bytes(b"video")
    scan = frames.scan_video(video, config.get_settings().temp_dir / "scan-webm")
    assert scan["duration"] == 10.0


def test_two_scans_of_one_video_at_once_decode_it_once(tmp_path, monkeypatch):
    calls = []
    _fake_ffmpeg(monkeypatch, [(0.0, 0.0, 0), (5.0, 0.0, 8)], calls)
    slow_scan = frames._run_scan

    def slow(*args):
        threading.Event().wait(0.3)
        return slow_scan(*args)

    monkeypatch.setattr(frames, "_run_scan", slow)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"video")
    scan_dir = config.get_settings().temp_dir / "scan-once"
    results = []
    workers = [
        threading.Thread(
            target=lambda: results.append(frames.scan_video(video, scan_dir))
        )
        for _ in range(3)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert calls == [1]
    assert len(results) == 3


def test_a_re_extracted_frame_that_does_not_match_falls_back_to_the_shared_one(
    tmp_path, monkeypatch
):
    shared = tmp_path / "shared.jpg"
    _striped(8).save(shared)
    frame = {
        "time": 12.0,
        "path": shared,
        "shared": True,
        "hash": frames.dhash(shared),
    }

    def extract(video_path, time, output):
        _striped(3).save(output)  # a different screen: the seek missed
        return output

    monkeypatch.setattr(frames, "extract_frame", extract)
    with frames.pictures(tmp_path / "talk.mp4") as picture:
        assert picture(frame) == shared


def test_ffmpeg_errors_reach_the_user_as_their_last_lines():
    log = "ffmpeg version 7\n  configuration: --enable-everything\n\n" + (
        "Stream #0:0: Audio: aac\nStream map '0:v' matches no streams.\n"
        "Error opening output files: Invalid argument\n"
    )
    short = frames._last_lines(log)
    assert short.startswith("Stream map '0:v' matches no streams.")
    assert short.endswith("Error opening output files: Invalid argument")


def test_the_frame_size_is_read_from_the_video_stream(monkeypatch, tmp_path):
    asked = []

    def probe(path, **kwargs):
        asked.append(kwargs)
        return {"streams": [{"codec_type": "video", "width": 2560, "height": 1440}]}

    monkeypatch.setattr(frames.ffmpeg, "probe", probe)

    assert frames.video_frame_size(tmp_path / "call.mkv") == (2560, 1440)
    assert asked == [{"select_streams": "v:0"}]


def test_an_unreadable_frame_size_is_zero(monkeypatch, tmp_path):
    def missing(path, **kwargs):
        raise FileNotFoundError(2, "ffprobe not found")

    monkeypatch.setattr(frames.ffmpeg, "probe", missing)
    assert frames.video_frame_size(tmp_path / "call.mkv") == (0, 0)

    monkeypatch.setattr(frames.ffmpeg, "probe", lambda path, **kwargs: {"streams": []})
    assert frames.video_frame_size(tmp_path / "call.mkv") == (0, 0)
