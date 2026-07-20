# CLAUDE.md — rules for working in this repo

Living docs: [docs/PROJECT.md](docs/PROJECT.md) (goal, architecture, **decisions with
reasons**, done / next, journal) · public docs: [README.md](README.md) ·
screenshots: [docs/screenshots/](docs/screenshots/).

**Read `docs/PROJECT.md` §3 before changing behaviour** — most "why is it like this?"
questions are already answered there (and several choices are workarounds for real bugs).

## Documentation rule

After any significant work, update **docs/PROJECT.md** in the same pass
(Done / Next / Decisions / Journal — entry `### YYYY-MM-DD — title`, newest on top).
User-facing changes (formats, flags, `.env` keys, commands) must also land in README.

## Invariants

- **Never commit unprompted** — the owner says when to commit.
- **Never add a `Co-authored-by` trailer** (it was scrubbed from the whole history).
- **Never print or echo the API key**, not even in tests — `config` calls
  `load_dotenv(override=True)`, so a "cleared" env var comes back with the real value.
- **Constants live in `config.py` only** — no format lists, model names, or limits
  duplicated in `app.py` / `transcribe.py`.
- **`app.py` is UI only** — no ffmpeg, IO, or API calls inline; that belongs in
  `audio.py` / `transcribe.py` (which must stay Streamlit-free).
- **Never widen `.gitignore` to `*.db`** — only `data/transcriptions.db` (+ `-*`).
- Local backend stays **`device="cpu"`** by default; CUDA is opt-in via `LOCAL_DEVICE`.

## Code conventions

- **Functional-first.** Classes only where natural (Pydantic Settings, service clients).
- **Modern type hints** (PEP 585/604) everywhere; target Python **3.12+**.
- **Google-style docstrings** on modules and public functions.
- **No `print()`** — `logger.get_logger(__name__)`. **No bare `Exception`** — use
  `exceptions.py`. **No scattered `os.getenv`** — `config.get_settings()`.
- Prefer `pathlib`. Must pass `ruff check` + `ruff format`.

## Dev commands

`make run` · `make sync` (add `--extra local` for the offline backend) ·
`make lint` / `make format` · `make test` · `make check` (CI gate) · `make reset`.

Tests run without network or ffmpeg. CI runs `make check` on push/PR.

## Commit messages

**Conventional Commits**: `<type>(<scope>): <short summary>` — imperative, lowercase,
no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`,
`style`, `build`, `ci`. Example: `fix(transcribe): offset subtitle timestamps per segment`
