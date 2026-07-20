"""Tests for the pure helpers in key-frame extraction (no ffmpeg, no network)."""

from PIL import Image

import frames


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


def test_sample_interval_tightens_for_short_videos():
    # A 25 s clip would otherwise get a single sample from a 30 s interval.
    assert frames.sample_interval(25.0, 30.0) == 12.5


def test_sample_interval_honours_the_chosen_cadence_on_long_videos():
    assert frames.sample_interval(3600.0, 30.0) == 30.0
    assert frames.sample_interval(3600.0, 300.0) == 300.0


def test_sample_interval_never_goes_below_the_floor():
    assert frames.sample_interval(4.0, 30.0) == 5.0


def test_sample_interval_falls_back_when_duration_is_unknown():
    assert frames.sample_interval(0.0, 30.0) == 30.0


def test_apply_min_interval_drops_closely_spaced_frames():
    candidates = [
        {"time": 0.0},
        {"time": 0.5},
        {"time": 1.0},
        {"time": 4.0},
    ]
    kept = frames.apply_min_interval(candidates, 2.0)
    assert [frame["time"] for frame in kept] == [0.0, 4.0]


def test_apply_min_interval_measures_from_the_last_kept_frame():
    # 1.5 is dropped, so 2.5 is still only 2.5 s after the last KEPT frame.
    candidates = [{"time": 0.0}, {"time": 1.5}, {"time": 2.5}]
    kept = frames.apply_min_interval(candidates, 2.0)
    assert [frame["time"] for frame in kept] == [0.0, 2.5]


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
