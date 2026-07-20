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
| `frames.py` | ffmpeg key-frame selection + perceptual dedup for on-screen context |
| `vision.py` | Describes those frames with a vision model; cost estimates for the UI |
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
       → pick engine + model  [+ optional on-screen context, video only]
       → (on-screen context: ffmpeg selects key frames → dedup → vision model)
       → OpenAI (split into 10-min chunks, 25 MB limit)  OR  local (whole file at once)
       → rendering: (M:SS) + paragraphs, on-screen notes interleaved → .txt
                                                          (+ .srt if requested)
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

### On-screen context (video)
- **Coverage is guaranteed by time sampling, not by scene detection.** ffmpeg's
  `scene` score is tuned for natural footage: a measured full-screen slide change
  from navy to dark red scored only **0.077**, far below the 0.3 usually quoted.
  So scene detection runs at a permissive 0.1 as a *candidate generator*, and a
  frame is taken at least every 30 s regardless. Relying on the threshold alone
  silently drops slides.
- **One ffmpeg pass does both**, via `select='eq(n,0)+gt(scene,T)+gte(t-prev_selected_t,I)'`.
  `eq(n,0)` is required — the first frame's scene score is undefined, so it is
  otherwise never emitted — and `+` acts as OR because any non-zero value is true.
  One pass keeps every timestamp on the same timeline and avoids one ffmpeg
  invocation per sampled second.
- **`-fps_mode vfr` is mandatory.** Without it the image2 muxer pads back to a
  constant frame rate: one measurement turned 4 selected frames into 79 files,
  which misaligns images against timestamps *silently*. The flag is probed for at
  runtime because it replaced `-vsync` only in ffmpeg 5.1, and git builds report
  no parseable version number.
- **Timestamps come from `showinfo`, matched positionally to the written files.**
  The parser is anchored on `Parsed_showinfo` because ffmpeg logs other lines
  containing `pts_time`, and matching one shifts every frame. A count mismatch
  raises rather than mislabels the video.
- **The sampling interval is a UI slider** (5–300 s, default 30). It is an *upper*
  bound — scene changes are captured on top of it — so the label reads "at least
  every".
- **Short videos get tightened, but only just.** A clip shorter than the chosen
  interval would be represented by a single frame, so the interval drops to
  `duration / 2`. An earlier version aimed for ~8 samples, which silently
  overrode the slider on anything short; the guarantee is now the minimum that
  fixes the bug and otherwise leaves the chosen cadence alone.
- **Deduplication uses a 64-bit dHash, threshold 2.** Bigger hashes measured
  *worse*: slides are mostly flat, so extra bits sample areas where adjacent
  pixels tie and become coin-flips. The threshold is deliberately tight because
  the costs are asymmetric — a false merge loses a slide forever, a false keep
  wastes a fraction of a cent. Each frame is compared to the **last kept** frame,
  not its predecessor, so a slow fade cannot ratchet past the threshold.
  **Known limit:** a slide differing only in a word or number is ~1 bit away —
  inside the noise floor — so it is treated as a duplicate and dropped.
- **A hard cap of 40 frames** keeps the cost bounded; over the cap, frames are
  sampled evenly across the timeline rather than truncated.
- **Chat Completions, not the Responses API** — the repo pins `openai==1.75.0`,
  whose `Responses` type does not accept the newer reasoning/detail values. In
  Chat Completions, `detail` goes *inside* `image_url`.
- **One request per frame.** Batching saves only the shared prompt (image tokens
  dominate and are billed per image regardless) while risking the model
  conflating frames and losing everything on a single failure.
- **Uninformative frames are dropped by the model itself** — it replies `NONE` for
  a face or a blank desktop, so a talking-head recording adds nothing.
- **Failure is never fatal**: if extraction or description fails, the run warns
  and continues, because a transcript without on-screen notes is still worth having.

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
- **On-screen context** for video: key frames described by a vision model and
  interleaved into the transcript by timestamp, with a selectable screenshot
  interval and a pre-run cost ceiling *(2026-07-20)*
- **Persistent history** (SQLite): browse, re-download TXT/SRT, delete
- **Hybrid API key** (`.env` or sidebar); offline works with no key
- **Wide layout + tabs** (Transcribe / History), two-column arrangement

**Quality / infrastructure**
- Modular refactor (config/audio/transcribe/db/exceptions/logger), type hints + docstrings
- **ruff: 0 errors** (down from 74), `ruff format` clean; modern ruff config (`[tool.ruff.lint]`, `target-version=py312`, plus D/RUF/PTH/T20/S)
- **pytest: 46 tests** (DB CRUD + PRAGMAs, SRT/formatting helpers, frame selection and dedup, cost estimates, headless Streamlit UI checks) — no network, no ffmpeg
- **CI** (GitHub Actions): ruff + format check + pytest on push/PR
- **Makefile**: `run` / `sync` / `lint` / `format` / `test` / `check` / `clean` / `reset`
- **Docker**: `Dockerfile` + `docker-compose.yml` + `.dockerignore` (ffmpeg bundled, offline backend opt-in via `INSTALL_LOCAL=true`)
- **Launchers**: `scripts/run.bat`, `scripts/run.sh`, `scripts/create-shortcut.ps1` + icon
- **README**: badges, full-width screenshots, offline mode, Docker, quick launch
- Repo is **public**, history cleaned, no `Co-authored-by` trailers

---

## 5. Next steps

- [ ] **Local OCR as a cheaper visual layer** — Tesseract could read slide text for zero
      tokens, leaving the vision model only for "describe what is happening". Deferred
      because it adds a second system binary alongside ffmpeg. It would also fix the
      known dedup limit (slides differing only in text), since OCR compares words.
- [ ] **Downscale frames before upload** — at `detail: "low"` the server resizes to
      512×512 anyway, so sending 1080p JPEGs wastes upload bandwidth for no benefit.
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

### 2026-07-20 (part 2) — On-screen context from video

**Goal:** a meeting recording's slides and shared screens should reach the transcript,
not just the speech — without paying to look at 360 near-identical frames.

**Built:** `frames.py` (ffmpeg key-frame selection + dHash dedup) and `vision.py`
(captioning + cost estimates), interleaved into the transcript by `build_transcript`.
UI: a video-only checkbox with a model/detail choice and a pre-run cost ceiling.

**Four assumptions that measurement overturned** — each would have shipped a silent bug:

1. **"Scene threshold 0.3 catches slide changes."** It does not. A measured
   full-screen navy→dark-red change scored **0.077**. Fix: treat scene detection as a
   candidate generator at 0.1 and guarantee coverage with time sampling instead.
2. **"A 30 s sampling floor is enough."** On a 25 s test clip it never fired, so a
   whole slide went missing. Fix: the interval tightens to spread ~8 samples over
   short videos.
3. **"A bigger 256-bit hash separates slides better."** My own measurement said yes —
   but it compared two *identically rendered PNGs*, with no JPEG noise. On a realistic
   corpus the bigger hash is **worse**: slides are mostly flat, so extra bits sample
   ties and become coin-flips. Fix: 64-bit dHash, threshold 2 (was 16-bit/10).
4. **"Newer models need the Responses API."** They accept Chat Completions, which is
   what the pinned `openai==1.75.0` actually supports — the newer API rejects its
   reasoning/detail values on that version.

**Also fixed before it bit:** the `pts_time` parser was matching an unrelated ffmpeg
log line, which would have shifted every frame's timestamp. Now anchored on
`Parsed_showinfo`, with a regression test.

**Verified end-to-end**, not just by unit test: a generated four-slide video produced
5 candidates → dedup → exactly the 3 distinct slides, and a real vision call read them
back correctly (`The slide displays "Q3 Revenue" and "4.2M (+18% YoY)"`). Confirmed
`gpt-5.4-nano`/`gpt-5.4-mini` are reachable before defaulting to them.

**Tests: 48** (was 18), including headless Streamlit `AppTest` checks of the new
controls — the browser cannot drive a native file picker, so the upload-gated UI is
covered there instead. Estimated ceiling for the default settings: 40 frames ≈ 25K
image tokens ≈ half a cent.

**Follow-ups the same day:** removed the "install offline Whisper" hint from the UI
(it belongs in the README, not in front of users), and exposed the screenshot
interval as a slider — which surfaced that the short-video tightening was
overriding any value the user picked.

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
