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
    monkeypatch.setenv("FRAME_MAX_COUNT", str(config.DEFAULT_FRAME_MAX_COUNT))

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
