"""Shared pytest fixtures.

Isolates each test from the real environment: a dummy API key satisfies the
required setting, and the working directories point at a per-test temp path so
the real ``data/`` database is never touched.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point settings at a throwaway temp dir with a dummy API key."""
    import config

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    # Pin the overridable settings, so a developer's own .env cannot change
    # what the tests assert (and CI, which has no .env, agrees with them).
    # load_dotenv(override=True) has already copied .env into the environment,
    # and pydantic would read the file again, so both routes are closed.
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    monkeypatch.setenv("FRAME_MAX_COUNT", str(config.DEFAULT_FRAME_MAX_COUNT))
    monkeypatch.delenv("SEGMENT_DURATION_MINUTES", raising=False)
    monkeypatch.setenv("SERBIAN_LATIN", "true")

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def wav_file(tmp_path):
    """Write a 16-bit PCM WAV made of square-wave stretches, without ffmpeg.

    ``wav_file("talk.wav", (3, 0.8), (2, 0.0))`` is three seconds at 80 % of
    full scale followed by two of silence; ``channels`` and ``rate`` default
    to the mono 16 kHz the app works with.
    """
    import struct
    import wave

    def make(name: str, *stretches, channels: int = 1, rate: int = 16000):
        path = tmp_path / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            for seconds, level in stretches:
                peak = int(level * 32767)
                up = struct.pack("<h", peak) * channels
                down = struct.pack("<h", -peak) * channels
                cycle = up * 20 + down * 20
                cycles, rest = divmod(int(seconds * rate), 40)
                handle.writeframes(cycle * cycles + cycle[: rest * 2 * channels])
        return path

    return make
