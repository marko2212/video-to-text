# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Streamlit app that transcribes speech to text via the OpenAI audio API.
It accepts video files (audio is extracted with ffmpeg) and audio files
(fed straight to transcription). Long inputs are split into segments.

Module layout (UI thin, logic separated):

- `app.py` — Streamlit UI only (render functions; no ffmpeg/IO inline)
- `config.py` — Pydantic Settings + shared constants (single source of truth)
- `audio.py` — ffmpeg helpers: save uploads, convert to mono-16k WAV
- `transcribe.py` — transcription pipeline (`process_audio`), UI-agnostic
- `db.py` — SQLite history (`data/transcriptions.db`)
- `exceptions.py` — domain exceptions (`AppError` and subclasses)
- `logger.py` — `get_logger()` (stdlib logging)
- Dependencies managed with `uv` (see `pyproject.toml` / `uv.lock`)

## Code conventions

- **Functional-first.** Pure functions for logic; classes only where natural
  (Pydantic Settings/models, stateful service clients).
- **Modern type hints** everywhere (PEP 585/604: `list[str]`, `str | None`).
  Target Python **3.12+**.
- **Google-style docstrings** on modules and public functions (`Args`/`Returns`/
  `Raises`). Keep trivial private helpers' docstrings short.
- **No `print()`** — use `logger.get_logger(__name__)`.
- **Domain exceptions** from `exceptions.py`, not bare `Exception`.
- **Config/secrets** via `config.get_settings()` (Pydantic Settings + `.env`),
  never `os.getenv` scattered around.
- Code must pass `ruff check` and `ruff format` (config in `pyproject.toml`).
- Prefer `pathlib` over `os.path`.

## Dev commands

Use the Makefile targets (run `make` for the full list):

- `make run` — start the app (`uv run streamlit run app.py`)
- `make sync` — install dependencies from `uv.lock`
- `make lint` / `make format` — ruff
- `make test` — run the pytest suite (`tests/`)
- `make check` — lint + format check + tests (same gate as CI)
- `make reset` — rebuild the `.venv`

Tests live in `tests/` and run without network/ffmpeg (pure helpers + SQLite).
CI (`.github/workflows/ci.yml`) runs `make check`'s steps on push/PR.

## Commit messages

Use **Conventional Commits**:

```
<type>(<scope>): <short summary>
```

- Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`, `style`, `build`, `ci`
- Scope optional but encouraged (e.g. `feat(upload)`, `fix(transcribe)`)
- Summary in imperative mood, lowercase, no trailing period

Examples:

- `feat(upload): add direct audio file upload support`
- `fix(transcribe): offset subtitle timestamps per segment`
- `docs(readme): document supported audio formats`
