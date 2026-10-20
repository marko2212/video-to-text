"""The picture of the sound: loudness per stretch and its drawing (no ffmpeg)."""

from io import BytesIO

import pytest
from PIL import Image

import waveform
from exceptions import AudioProcessingError


def test_loudness_measures_each_stretch_and_the_length(wav_file):
    path = wav_file("talk.wav", (3, 0.8), (3, 0.0))

    levels, duration = waveform.loudness(path, buckets=6)

    assert duration == pytest.approx(6.0)
    assert len(levels) == 6
    assert all(level > 0.9 for level in levels[:3])
    assert all(level < 0.05 for level in levels[3:])


def test_one_loud_moment_does_not_flatten_the_rest(wav_file):
    # Ninety-nine seconds of ordinary speech and one clap at full scale: the
    # bars are scaled to the speech, and the clap is simply as tall as can be.
    path = wav_file("clap.wav", *([(1, 0.2)] * 99), (1, 1.0))

    levels, _ = waveform.loudness(path, buckets=100)

    assert all(level > 0.9 for level in levels[:99])
    assert levels[99] == 1.0


def test_other_formats_go_through_ffmpeg_to_a_mono_wav(wav_file, monkeypatch):
    # A stereo 48 kHz WAV is not read as it is: it is converted (the real
    # conversion needs ffmpeg) and the converted copy is removed afterwards.
    stereo = wav_file("stereo.wav", (2, 0.5), channels=2, rate=48000)
    plain = wav_file("plain.wav", (2, 0.5))
    made = []

    def fake_to_wav(source, target, start=None, duration=None):
        made.append(target)
        target.write_bytes(plain.read_bytes())
        return target

    monkeypatch.setattr(waveform, "to_wav", fake_to_wav)

    levels, duration = waveform.loudness(stereo, buckets=4)

    assert duration == pytest.approx(2.0)
    assert len(levels) == 4
    assert len(made) == 1 and not made[0].exists()


def test_a_file_that_cannot_be_decoded_is_an_audio_error(tmp_path, monkeypatch):
    junk = tmp_path / "junk.m4a"
    junk.write_bytes(b"not audio")
    made = []

    def failing(source, target, **kwargs):
        made.append(target)
        raise AudioProcessingError("ffmpeg: Invalid data found")

    monkeypatch.setattr(waveform, "to_wav", failing)

    with pytest.raises(AudioProcessingError):
        waveform.loudness(junk)
    assert len(made) == 1 and not made[0].exists()


def test_a_missing_ffmpeg_is_an_audio_error_too(tmp_path, monkeypatch):
    # CI has no ffmpeg: the page must show no slider rather than fail.
    other = tmp_path / "call.amr"
    other.write_bytes(b"not audio")

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'ffmpeg'")

    monkeypatch.setattr(waveform, "to_wav", missing)

    with pytest.raises(AudioProcessingError, match=r"call\.amr"):
        waveform.loudness(other)


def test_a_recording_without_sound_has_no_levels(wav_file):
    assert waveform.loudness(wav_file("empty.wav")) == ([], 0.0)


def test_the_picture_colours_the_kept_stretch_and_greys_the_rest():
    levels = [1.0] * 10
    middle = 10

    image = Image.open(BytesIO(waveform.picture(levels, (0.2, 0.6), 10, 20)))

    assert image.size == (10, 20)
    kept = image.getpixel((3, middle))
    left_out = image.getpixel((8, middle))
    assert kept == (255, 75, 75, 255)
    assert left_out[0] == left_out[1] == left_out[2]  # grey
    assert left_out != kept
    # The background stays transparent, so both themes can show the picture.
    assert image.getpixel((3, 0))[3] == 0

    whole = Image.open(BytesIO(waveform.picture(levels, None, 10, 20)))
    assert all(whole.getpixel((x, middle)) == (255, 75, 75, 255) for x in range(10))


def test_a_silent_stretch_still_shows_a_hairline():
    image = Image.open(BytesIO(waveform.picture([0.0, 0.0], None, 2, 10)))

    assert image.getpixel((0, 5))[3] > 0
    assert image.getpixel((0, 1))[3] == 0
