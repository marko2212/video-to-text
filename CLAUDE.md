# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Streamlit app that transcribes speech to text via the OpenAI audio API.
It accepts video files (audio is extracted with ffmpeg) and audio files
(fed straight to transcription). Long inputs are split into segments.

- `app.py` — Streamlit UI
- `transcribe.py` — transcription pipeline (`process_audio`)
- Dependencies managed with `uv` (see `pyproject.toml` / `uv.lock`)

## Dev commands

Use the Makefile targets (run `make` for the full list):

- `make run` — start the app (`uv run streamlit run app.py`)
- `make sync` — install dependencies from `uv.lock`
- `make lint` / `make format` — ruff
- `make reset` — rebuild the `.venv`

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
