# Video & Audio Transcription — project documentation

Living documentation: goal, architecture, **decisions with the reasons behind them**,
current state, next steps, and a dated journal.

## 1. Goal

A Streamlit app that turns **speech into text**. It accepts **video** (the audio track
is extracted with ffmpeg) and **audio** files, and transcribes them two ways:

- **OpenAI API** — `gpt-4o-transcribe` (more accurate) or `whisper-1` (returns timestamps)
- **Local (offline)** — a `faster-whisper` model on your own machine: **free, private,
  no internet and no API key**

Output: a readable **transcript with timestamps** (plus optional `.srt` subtitles), with
**persistent history** in SQLite.

**Real-world use:** work meetings, phone calls and interviews — `.mkv` screen recordings,
`.amr` phone recordings, long multi-speaker sessions.

### Document map
- **[PROJECT.md](PROJECT.md)** (this file) — living docs: goal, decisions, done, next, journal
- **[../README.md](../README.md)** — public docs: install, usage, offline mode, Docker, screenshots
- **[../CLAUDE.md](../CLAUDE.md)** — rules for working in this repo (conventions, commands, invariants)
- **[screenshots/](screenshots/)** — UI images used by the README

---

## 2. Technology and architecture

**Stack:** Python **3.12+** · Streamlit 1.56 · OpenAI SDK · `faster-whisper` (optional) ·
pydub + **ffmpeg** · Pydantic Settings · SQLite · **uv** (dependencies) · **ruff**
(lint + format) · pytest · GitHub Actions.

### Modules (thin UI, logic extracted)
| File | Role |
|---|---|
| `app.py` | **UI only** — render functions, tabs, sidebar. No ffmpeg/IO logic inline. |
| `config.py` | **Pydantic Settings + every constant** (single source of truth): formats, models, paths, limits |
| `audio.py` | ffmpeg helpers: `save_uploaded_file`, `to_wav` (mono 16 kHz) — UI-agnostic |
| `transcribe.py` | Pipeline: `transcribe_openai` (chunked) and `transcribe_local`; transcript rendering |
| `db.py` | SQLite history (`data/transcriptions.db`) |
| `exceptions.py` | Domain exceptions (`AppError` → `AudioProcessingError`, `TranscriptionError`) |
| `logger.py` | `get_logger()` — stdlib logging (never `print()`) |
| `scripts/` | Launchers: `run.bat` (Windows), `run.sh` (Linux/macOS), `create-shortcut.ps1` |
| `tests/` | pytest — pure functions + SQLite (no network, no ffmpeg) |

### Data flow
```
upload → (video? ffmpeg to_wav mono-16k : use the file as-is)
       → audio player
       → pick engine + model
       → OpenAI (split into 10-min chunks, 25 MB limit)  OR  local (whole file at once)
       → rendering: (M:SS) + paragraphs → .txt   (+ .srt if requested)
       → write to SQLite history
```

### How the two engines differ
| | OpenAI API | Local (offline) |
|---|---|---|
| API key | required (`.env` or sidebar) | **not needed** |
| Chunking | **yes** (25 MB per-request limit) | **no** (whole file) |
| Timestamps | `whisper-1` only (`verbose_json`) | **always** (native) |
| Cost | billed per minute | free (costs CPU time) |

---

## 3. Key decisions (with reasons)

### Transcription
- **`gpt-4o-transcribe` is the default** — measurably more accurate than `whisper-1` in a
  real side-by-side run on the same file ("Microsoft **Stack**" vs "Microsoft's **back**";
  "based in **Serbia**" vs "Croatia, Serbia"). **But it returns no timestamps** — the API
  does not support `verbose_json` for it.
- **`whisper-1` is selected only when timestamps/SRT are wanted** — it is the only OpenAI
  model that returns segments. The trade-off is lower accuracy.
- **Segments are always requested when the model supports them** (not only when SRT is
  requested) — they drive the readable transcript. The SRT *file* is still written only on
  request.
- **Per-chunk timestamp offset** — OpenAI returns times relative to each chunk, so an
  `offset_seconds` is added to keep one continuous timeline. Verified across 4 chunks
  (boundaries at 1:00 / 2:00 / 3:00 are correct).
- **The transcript is rendered from segments**, not written as flat text: inline `(M:SS)`
  markers plus paragraph breaks (on a pause > 2 s, or once a paragraph exceeds 350
  characters). With no segments available, sentences are grouped instead. *Reason: the
  output used to be one unreadable wall of text.*

### Offline engine
- **`faster-whisper`** (not `openai-whisper`, not `whisper.cpp`) — the best fit for Python
  and CPU in 2026 (CTranslate2, int8), easy to install, native timestamps.
- **Optional extra** (`uv sync --extra local`) — ctranslate2/onnxruntime are heavy
  (hundreds of MB) and should not burden API-only users. The UI hides the Local option
  when the package is absent.
- **`device="cpu"` is the default (NOT `auto`)** — `auto` detected a GPU and then failed at
  inference with `RuntimeError: cublas64_12.dll not found` on an incomplete CUDA install.
  GPU is opt-in via `LOCAL_DEVICE=cuda`.
- **The model downloads on demand** (first use) and is cached in `models/` (gitignored).

### Configuration and API key
- **Pydantic Settings** (`config.py`) — one source of truth. Constants used to be duplicated
  between `app.py` and `transcribe.py` and drifted apart easily.
- **Hybrid API key:** the key is **optional** — read from `.env` if present, otherwise typed
  into a sidebar password field (**session only, never written to disk**). *Reason: anyone
  cloning the repo must be able to run it immediately, and the offline engine needs no key
  at all.*

### UI / flow
- **Auto-prepare + ONE click:** a video's audio is extracted automatically on upload (the
  old "Extract Audio" button is gone), but transcription stays an **explicit click**.
  *Reason: that is where the model and timestamp options are chosen, and it prevents every
  dropped file from immediately spending API credits.*
- **Download buttons are rendered OUTSIDE the click handler** — otherwise clicking one
  download makes the other disappear (a Streamlit rerun trap).
- **A spinner instead of a download progress bar** — a polling progress bar did not track
  Hugging Face's **Xet** transfer backend (it sat at ~0). A message with the model size is
  honest and reliable.
- **CSS hack for the uploader label** — Streamlit prints the full list of 27 extensions; it
  is hidden and replaced with short text. The element to target is a **`<span>`** (not
  `<small>` — established by inspecting the DOM).

### Data
- **SQLite lives in `data/`, not `temp/`** — history **survives** "Clean temporary files"
  (which wipes `temp/` and `uploads/`). That is the whole point of having it.
- **PRAGMA `journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON`** — better defaults
  for Streamlit's rerun model and multiple open sessions.
- **Migration without a migration tool:** `init_db()` checks `PRAGMA table_info` and adds
  the `provider` column when missing, so existing databases are not lost.
- **Audio is NOT stored as a BLOB** — only the transcript and SRT text; `audio_path` is a
  best-effort reference that may disappear after a cleanup.

### Code and process
- **Functional-first**, classes only where natural (Pydantic Settings, service clients).
- **Modern type hints (PEP 585/604) + Google-style docstrings** everywhere; ruff `ANN` + `D`.
- **Domain exceptions** instead of `raise Exception(...)`; **logging** instead of `print()`.
- **A single ffmpeg conversion** (`audio.to_wav` → mono 16 kHz). Video used to be converted
  to a full stereo WAV first and then again to mono-16k — a huge intermediate file and
  double the work.
- **Conventional Commits** (`feat(scope): ...`).
- **No `Co-authored-by` trailer** in commit messages (owner's decision; also removed from
  the entire history).
- **History was rewritten and cleaned** before the repo went public (22 messy commits → a
  clean root plus conventional commits). Verified: `.env` was never committed and no key
  material appears anywhere in the history.

### Distribution / running
- **Launchers live in `scripts/`, not the repo root** — the root of a public Python repo
  stays clean and platform-neutral.
- **No `.vbs`** — it carries a historical malware reputation and some AV products flag it.
  Instead of hiding the console, the shortcut starts `run.bat` **minimised**
  (`WindowStyle=7`).
- **Docker is optional and additive** — it bundles ffmpeg (the host needs nothing), but
  `make run` / `uv run` keep working exactly as before.

---

## 4. Done

**Features**
- Upload **video** (16 formats) and **audio** (11 formats, incl. AMR); video audio is extracted automatically
- **Audio player** before and after transcription
- **Two engines**: OpenAI API (`gpt-4o-transcribe` / `whisper-1`) and **Local offline** (faster-whisper, selectable model size)
- **Timestamps + `.srt`** export (whisper-1 and every local model)
- **Readable transcript**: inline `(M:SS)` + paragraphs *(2026-07-20)*
- **Persistent history** (SQLite): browse, re-download TXT/SRT, delete
- **Hybrid API key** (`.env` or sidebar); offline works with no key
- **Wide layout + tabs** (Transcribe / History), two-column arrangement

**Quality / infrastructure**
- Modular refactor (config/audio/transcribe/db/exceptions/logger), type hints + docstrings
- **ruff: 0 errors** (down from 74), `ruff format` clean; modern ruff config (`[tool.ruff.lint]`, `target-version=py312`, plus D/RUF/PTH/T20/S)
- **pytest: 14 tests** (DB CRUD + PRAGMAs, SRT and formatting helpers) — no network, no ffmpeg
- **CI** (GitHub Actions): ruff + format check + pytest on push/PR
- **Makefile**: `run` / `sync` / `lint` / `format` / `test` / `check` / `clean` / `reset`
- **Docker**: `Dockerfile` + `docker-compose.yml` + `.dockerignore` (ffmpeg bundled, offline backend opt-in via `INSTALL_LOCAL=true`)
- **Launchers**: `scripts/run.bat`, `scripts/run.sh`, `scripts/create-shortcut.ps1` + icon
- **README**: badges, full-width screenshots, offline mode, Docker, quick launch
- Repo is **public**, history cleaned, no `Co-authored-by` trailers

---

## 5. Next steps

- [ ] **Visual context from video** *(designed, not implemented)* — capture frames on scene
      changes (`ffmpeg select='gt(scene,0.3)'` plus the **mandatory `-fps_mode vfr`**),
      dedupe with pHash, apply min/max interval and a hard cap, then run a vision model or
      local OCR; merge with the transcript by timestamp. *Roughly 90% fewer tokens than
      sampling a frame every 10 s.*
- [ ] **Hosting** *(researched)* — Hugging Face Spaces (Docker, 16 GB RAM, easy) or an
      Oracle Always Free VM (4 ARM cores / 24 GB, more powerful but more DevOps).
      **Condition:** users must supply **their own** OpenAI key, otherwise the owner pays
      for everyone else's usage.
- [ ] **Concurrency limit + queue** if the offline engine is ever exposed publicly
      (4 cores ≈ 1 fast or ~4 slow transcriptions at once).
- [ ] `.gitattributes` (`*.sh text eol=lf`) so CRLF does not break `run.sh` on Linux.
- [ ] Later ideas: summary / action items (a GPT pass), speaker diarisation
      (WhisperX or Deepgram), a `prompt=` seeded with domain terms for better jargon.

---

## 6. Journal

### 2026-07-20 — Readable transcript + documentation

**Problem:** the transcript was one wall of text (plus `--- Segment 1 ---` noise and
`Transcription started/completed` lines), unlike TurboScribe, which shows `(0:14)` markers
and paragraphs.

**Root cause (found in the code):** `_transcribe_all` wrote the flat `result.text` into the
`.txt` and used segments **only** for the `.srt`. On top of that, `gpt-4o-transcribe`
returns no segments at all.

**Changes:**
- `build_transcript()` — renders `(M:SS)` markers and paragraphs from segments
- `split_into_paragraphs()` — fallback sentence grouping (used for `gpt-4o-transcribe`)
- `_write_outputs()` — one shared writer for both engines
- `whisper-1` now **always** requests `verbose_json` (segments are free); the SRT file stays optional
- the local engine **always** collects segments
- removed the noise from `.txt` (header/footer and chunk markers)
- +5 tests (**14** total), ruff and format clean
- **Verified live** with the local `base` model: `(0:00) Hello, and welcome... (0:04) This tool...`

**Also:** created `docs/PROJECT.md` (this file) so the decisions and their reasons outlive
the session.

### Earlier in the same cycle (condensed)

- **Offline engine** (faster-whisper) as an optional extra plus model-size selection in the
  UI; `device=cpu` became the default after the CUDA `cublas64_12.dll` failure.
- **Hybrid API key** (`.env` or sidebar) — the app no longer crashes without a key.
- **SQLite history** and a History tab; it survives "Clean temporary files".
- **Large refactor** (modularisation, Pydantic Settings, type hints, docstrings, logging,
  domain exceptions, pathlib) — ruff went from 74 errors to 0.
- **pytest + CI + Makefile**; **Docker**; **launchers** moved into `scripts/`.
- **Git history cleanup** before the repo went public; `Co-authored-by` removed from every
  commit.
- **Fixes along the way:** broken `.venv` trampolines after renaming the project folder
  (`rm -rf .venv && uv sync`); Streamlit 1.40 → 1.56; the uploader-label CSS hack
  (`<span>`, not `<small>`); download buttons moved out of the click handler; per-chunk
  timestamp offsets.
