"""The application log: a rotating file beside the history, with keys blanked."""

import logging
from logging.handlers import RotatingFileHandler

import pytest

import config
import logger


@pytest.fixture
def file_log(tmp_path):
    """Enable the file log in a scratch folder and remove it afterwards."""
    path = logger.enable_file_log(tmp_path / "logs" / "app.log")
    yield path
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, RotatingFileHandler):
            root.removeHandler(handler)
            handler.close()


def _file_handlers() -> list[RotatingFileHandler]:
    return [
        h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)
    ]


def test_log_lines_land_in_the_file(file_log):
    logger.get_logger("transcribe").info("Part 003 answered in 4.2 s")

    assert "| INFO     | transcribe | Part 003 answered in 4.2 s" in file_log.read_text(
        encoding="utf-8"
    )


def test_the_file_is_rotated_not_grown_without_bound(file_log):
    [handler] = _file_handlers()
    assert handler.maxBytes == config.LOG_FILE_MAX_BYTES
    assert handler.backupCount == config.LOG_FILE_BACKUPS


def test_anything_shaped_like_an_api_key_is_blanked(file_log):
    logger.get_logger("app").warning("client made with sk-proj-abcdefghijklmnop0123")

    text = file_log.read_text(encoding="utf-8")
    assert "sk-proj-abcdefghijklmnop0123" not in text
    assert "client made with sk-***" in text


def test_enabling_twice_keeps_one_handler_and_another_path_replaces_it(
    file_log, tmp_path
):
    assert logger.enable_file_log(file_log) == file_log
    assert len(_file_handlers()) == 1

    other = logger.enable_file_log(tmp_path / "elsewhere.log")
    logger.get_logger("app").info("moved")

    assert other != file_log
    assert len(_file_handlers()) == 1
    assert "moved" in other.read_text(encoding="utf-8")
    assert "moved" not in file_log.read_text(encoding="utf-8")


def test_the_default_path_sits_beside_the_history(tmp_path):
    settings = config.get_settings()
    assert settings.log_path == settings.data_dir / "logs" / "app.log"
    assert settings.log_path.parent.is_dir()  # created with the other folders
    assert settings.runs_dir == settings.data_dir / "runs"


def test_log_dir_can_be_set_in_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "elsewhere"))
    config.get_settings.cache_clear()
    try:
        assert config.get_settings().log_path == tmp_path / "elsewhere" / "app.log"
    finally:
        config.get_settings.cache_clear()
