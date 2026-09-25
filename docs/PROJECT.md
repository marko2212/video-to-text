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
| `app.py` | **UI only** — render functions, tabs, sidebar, the two-step run (request → locked page → job). No ffmpeg/IO logic inline. |
| `config.py` | **Pydantic Settings + every constant** (single source of truth): formats, models, paths, limits |
| `audio.py` | ffmpeg helpers: `save_uploaded_file`, `to_wav` (mono 16 kHz), `to_preview` (small MP3 for the player) — UI-agnostic |
| `frames.py` | ffmpeg key-frame selection + perceptual dedup for on-screen context |
| `vision.py` | Describes those frames with a vision model (cached per frame); cost estimates for the UI |
| `transcribe.py` | Pipeline: `transcribe_openai` (chunked, checkpointed) and `transcribe_local`; transcript rendering |
| `openai_api.py` | The OpenAI client (retry/timeout policy) and SDK errors → short app errors |
| `checkpoints.py` | On-disk results of paid requests, keyed by content hash, so failed runs resume |
| `serbian.py` | Serbian Cyrillic → Latin transliteration (pure functions) |
| `db.py` | SQLite history (`data/transcriptions.db`) |
| `exceptions.py` | Domain exceptions (`AppError` → `AudioProcessingError`, `TranscriptionError` → `IncompleteTranscriptionError`, `VisualContextError`, `OpenAIAccountError`) |
| `logger.py` | `get_logger()` — stdlib logging (never `print()`) |
| `scripts/` | Launchers: `run.bat` (Windows), `run.sh` (Linux/macOS), `create-shortcut.ps1`; one-off `serbian_latin_backfill.py` |
| `tests/` | pytest — pure functions, SQLite, the pipeline with a fake client, headless UI (AppTest); no network, no ffmpeg |

### Data flow
```
upload (identified by file_id) → (video? ffmpeg to_wav mono-16k : use the file as-is)
       → audio player (MP3 preview for video, AMR/WMA/AIFF and files > 25 MB)
       → pick engine + model  [+ optional on-screen context, video only]
       → Start: request stored, page redrawn with every control locked, job runs
       → (on-screen context: ffmpeg selects key frames → dedup → vision model,
          each description cached under temp/checkpoints/)
       → OpenAI (5-min chunks for gpt-4o, 10 for whisper-1, cut in pauses; each
          chunk checkpointed; an answer at the output cap is split and redone)
          OR  local (whole file at once)
       → rendering: (M:SS) or (~M:SS) + paragraphs, on-screen notes placed by time,
          Serbian Cyrillic → Latin → .txt (+ .srt if requested)
       → write to SQLite history (a partial run is shown, marked, and not saved)
```

### How the two engines differ
| | OpenAI API | Local (offline) |
|---|---|---|
| API key | required (`.env` or sidebar) | **not needed** |
| Chunking | **yes** (5 min for gpt-4o — output cap; 10 for whisper-1) | **no** (whole file) |
| Timestamps | exact with `whisper-1` (`verbose_json`); approximate `(~M:SS)` with gpt-4o | **always** (native) |
| Resume after a failure | **yes** (chunk checkpoints) | no (one pass) |
| Cost | billed per minute | free (costs CPU time) |

---

## 3. Key decisions (with reasons)

### Transcription
- **⚠ Every OpenAI model this app uses is retired on 2027-02-26** (notice of 2026-08-26,
  developers.openai.com/api/docs/deprecations): `whisper-1`, `gpt-4o-transcribe`,
  `gpt-4o-mini-transcribe` and `gpt-4o-transcribe-diarize`. The replacement,
  `gpt-transcribe` ($0.0045/min; accepts `languages`, `keywords`, `prompt`), returns **no
  timestamps** — `verbose_json`/`srt`/`timestamp_granularities` are whisper-1 only. After
  that date the API path has no segments, SRT or speaker labels, and the **local engine
  is the only source of timestamps**. The two model decisions below are therefore
  time-limited; the migration is in §5.
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
  characters). *Reason: the output used to be one unreadable wall of text.*
- **Without segments, paragraphs get approximate times `(~M:SS)`** *(2026-09-25)*. Where
  each chunk sits in the recording is known, so a paragraph is stamped by its share of
  its chunk's characters. It assumes an even speaking rate — hence the `~` — but it is
  what lets on-screen notes land next to the speech with the default model: before
  this, all 1,174 notes of 23 real runs sat in a block after the transcript. Exact
  placement stays with whisper-1 and the local engine; the stamps are not written to SRT.
- **Chunk length depends on the model: 5 minutes for gpt-4o, 10 for whisper-1**
  *(2026-09-25)*. The gpt-4o family stops writing at about 2,000 output tokens per
  request; dense Serbian speech reaches 2,450–2,920 tokens per 10 minutes, and one
  measured chunk lost its last 30–45 s with no warning. Five minutes peaks near 1,460.
  Billing is per audio minute, so it costs nothing extra. `SEGMENT_DURATION_MINUTES`
  overrides it for every model (1–15; 15 keeps a chunk under 25 MB). Unknown models get
  5 until their cap is known.
- **An answer at the cap is split and redone — once** — `usage.output_tokens` ≥ 1,900
  means the end of the chunk is missing, so its two halves are transcribed instead (each
  half checkpointed on its own). A half that fills the cap again is a loop or noise, not
  speech (dense Serbian is ~730 tokens per 2.5 min), so it is kept with a visible
  "⚠️ … may be missing" line rather than split further — unlimited splitting cost up to
  4× on a looping chunk. With 5-minute chunks the guard should rarely fire. `usage` is read from the response (SDK 1.75 keeps unknown fields);
  whether gpt-4o-transcribe actually returns it could not be confirmed live yet (no
  credit) — without it the guard stays silent and the shorter chunks are the protection.
- **Chunks are cut in pauses, and tiny tails are merged** — each boundary moves to the
  quietest 250 ms window within ±10 s of the nominal cut, so shorter chunks do not
  mean more words cut in half; a final chunk under 10 s joins the previous one. The
  old size-based "too small" checks, which failed whole runs after paid chunks for
  tails under 0.4 s, are gone.
- **No retry loop of our own; the SDK retries, except where it cannot help**
  *(2026-09-25)*. The app used to retry every error 3× on top of the SDK's 2 retries —
  up to 9 requests per chunk for a bad key or an empty balance, then the raw JSON
  twice. Now: `max_retries=3`, 300 s read / 10 s connect timeout, and an httpx response
  hook marks a 429 whose code is `insufficient_quota` / `credit_balance_exhausted` with
  `x-should-retry: false`, which the SDK obeys (it retries every other 429). Measured
  against the live API with an empty balance: **one request**, one sentence. Errors are
  translated in one place (`openai_api.translate_error`); `OpenAIAccountError` (bad
  key, no credit, billing inactive) is deliberately not a subclass of the per-chunk or
  per-frame errors, so nothing that tolerates one failed frame can swallow it. A 403 is
  **not** an account error: it concerns one model (a restricted key may transcribe but
  not use the vision model) and must not cancel a transcription that would succeed.
  The hook never raises — an exception there reaches the SDK as a "connection error".
- **Paid work is checkpointed** *(2026-09-25)*. Each finished chunk is written to
  `temp/checkpoints/<sha256 of the audio>_transcribe_<model>_<N>min_v1/`, and each frame
  description to `…_frames_<model>_<detail>_v1/`, as soon as it arrives. A rerun of the
  same content (any file name) reuses them — chunk checkpoints only if their boundaries
  still match. The run stops at the first failed chunk (the next would most likely
  fail the same way) and writes a partial transcript ending in a `⚠️` line that says
  where and why; it is shown but **not saved to history**, and Start sends only the
  rest. A chunk that was split keeps each half separately, so a failure in one half
  never costs the other. Checkpoints are deleted when the run succeeds (frame
  descriptions too, after the history row is written), by "Clean temporary files",
  and automatically after 14 days without use — they hold transcript text. Each run
  works in its own `mkdtemp` folder, because two tabs sharing `temp/segments` and
  `temp/frames` could otherwise cache one video's screens under another's hash; a
  folder left by a killed process is deleted after 24 hours (never while a run is
  active, since a screenshot folder can sit unchanged for over an hour).
- **Serbian comes out in Latin script** *(2026-09-25)*. Without a language hint the
  model picks the script per chunk, so a meeting read LLL CCC LLL in chunk-sized blocks
  (7 saved records, 2 calls fully Cyrillic). A deterministic 1:1 transliteration runs
  on the finished document when it is recognisably Serbian — Serbian-only letters
  (ђ ћ џ љ њ ј) present and outnumbering letters Serbian never uses — so Russian or
  Bulgarian is never touched. Chosen over `language="sr"` + a prompt because it is
  guaranteed (the prompt is a hint, and `sr` alone may push towards Cyrillic) and it
  also works for the local engine. `SERBIAN_LATIN=false` turns it off. Old records are
  converted only by an explicit one-off script, after a verified backup.

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
- **The expected screenshot count is shown before the run**, from the video's
  duration over the effective interval, alongside the estimated cost. It counts
  only what the interval guarantees, so it reads high when deduplication kicks in
  and low on a video full of cuts — the caption says so rather than implying a
  promise. Measured against real extractions on a 25 s clip it matched exactly at
  10/30/300 s and over-counted 6 vs 3 at 5 s, where dedup did its job. The
  duration probe is cached on path plus file size, so dragging the slider does
  not spawn an `ffprobe` per rerun.
- **The cost estimate includes output tokens.** Counting only image tokens
  understated it by about half, because a caption's output tokens are priced
  several times higher than input ones.
- **Short videos get tightened, but only just.** A clip shorter than the chosen
  interval would be represented by a single frame, so the interval drops to
  `duration / 2`. An earlier version aimed for ~8 samples, which silently
  overrode the slider on anything short; the guarantee is now the minimum that
  fixes the bug and otherwise leaves the chosen cadence alone.
- **Deduplication runs before anything is sent**, so discarded frames cost
  nothing. Measured on a meeting-shaped video (4 slides held 30 s each): ffmpeg
  produced 30 candidates, dedup kept 4 — the slide boundaries — and 86% of the
  frames never reached the API. This is also why the UI estimate is an upper
  bound: it is computed before dedup, so the real spend is usually lower.
  The model's own `NONE` reply is a second filter, but it costs a request and
  only keeps the transcript clean.
- **Deduplication uses a 64-bit dHash, threshold 2.** Bigger hashes measured
  *worse*: slides are mostly flat, so extra bits sample areas where adjacent
  pixels tie and become coin-flips. The threshold is deliberately tight because
  the costs are asymmetric — a false merge loses a slide forever, a false keep
  wastes a fraction of a cent. Each frame is compared to the **last kept** frame,
  not its predecessor, so a slow fade cannot ratchet past the threshold.
  **Known limit:** a slide differing only in a word or number is ~1 bit away —
  inside the noise floor — so it is treated as a duplicate and dropped.
- **The frame cap is a backstop, not the control.** It was originally 40, which
  was low enough to bind at *every* slider position on an 83-minute meeting —
  the estimate sat at "about 40… never more than 40" and dragging the slider
  changed nothing. Raised to 200: the real constraint is wall-clock (one request
  per frame), not money, since 200 frames is roughly 5 cents.
- **The cap widens the interval up front** rather than extracting and discarding.
  Clamping afterwards would write ~1,000 JPEGs and hash them only to throw most
  away. When it binds, the UI says so and shows the interval that will actually
  be used, instead of silently ignoring the chosen one.
- **Widening beats truncating** when the cap binds. Taking the first N frames
  would cover only the opening stretch of a recording; widening keeps coverage
  from start to finish and degrades resolution instead. What is lost is temporal
  detail, which degrades gracefully — whole missing sections do not.
- **`FRAME_MAX_COUNT` is a Settings field, not a bare constant**, so it can be
  set per machine in `.env`. The module constant is only the default; call sites
  resolve it at call time, because a default argument would freeze the value at
  import and quietly ignore the `.env`. `tests/conftest.py` pins it so a
  developer's own `.env` cannot change what the tests assert.
- **Chat Completions, not the Responses API** — the repo pins `openai==1.75.0`,
  whose `Responses` type does not accept the newer reasoning/detail values. In
  Chat Completions, `detail` goes *inside* `image_url`.
- **One request per frame.** Batching saves only the shared prompt (image tokens
  dominate and are billed per image regardless) while risking the model
  conflating frames and losing everything on a single failure.
- **Uninformative frames are dropped by the model itself** — it replies `NONE` for
  a face or a blank desktop, so a talking-head recording adds nothing.
- **Failure is not fatal — except when the account is refused**: if extraction or
  description fails, the run warns and continues, because a transcript without
  on-screen notes is still worth having. A bad key or an empty balance
  (`OpenAIAccountError`) stops at the first frame instead of trying every one; it then
  stops an OpenAI run (its transcription would be refused the same way) and only
  warns a Local one. Three frames failing in a row also end the step, and frames that
  fail one by one are counted in a warning instead of vanishing. Descriptions made
  before a stop are kept and used — they were paid for — and the description cache is
  discarded only when every screenshot was described, so a rerun pays only for the
  missing ones. A cached frame counts as a success, so old bad frames cannot trip the
  "3 in a row" stop on a resume. *(2026-09-25)*

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
- **Start records a job; the run it triggers draws every control locked and does the
  work** *(2026-09-25)*. Streamlit stops a running script whenever a widget changes,
  so any click during a run — even a History download — silently threw it away. Start
  now only records the job, in its `on_click` callback (an `st.rerun()` from the middle
  of the script instead dropped the state of every widget drawn after it, closing open
  History entries). That run draws every control `disabled` (not part of a widget's
  identity, so values survive) — the result box too, since editing it is a widget
  change — and runs the job at the very end of the script, into a container under
  Start, so the whole page is drawn first. History is an `st.fragment`: its reruns
  wait for the job instead of cancelling it (Streamlit never preempts a running script
  for a fragment rerun); Delete is an `on_click` callback and downloads use
  `on_click="ignore"`. Verified live: opening a History entry 5 s into a 33 s run did
  not stop it.
- **A job remembers its thread, because Stop does not stop a request.** The toolbar's
  Stop only takes effect at the thread's next Streamlit call — after a blocking OpenAI
  request or ffmpeg call returns, possibly a minute later. The first watchdog assumed
  the stopped run was gone and unlocked Start at once, so a second pipeline sent the
  same, already paid chunk in parallel; and a thread stopped during the last chunk
  still wrote the transcript and deleted the checkpoints, so the "resume" re-sent
  everything (found by the adversarial review, reproduced in a browser). Now the job
  stores the thread running it; a started job counts as abandoned only when that
  thread has ended. While a job has a live worker, a hidden fragment with
  `run_every=2s` is drawn: its body runs inline when drawn (worker alive → nothing),
  and afterwards only when the owning run was stopped (fragment reruns wait for a live
  run). Once the thread has ended it reruns; the new run says "stopped before it
  finished" and unlocks. Measured live: locked until the in-flight request returned,
  unlocked ~2 s later, the restart re-sent nothing already finished. A background
  worker thread (the "M" option for R05) was not needed: with checkpoints, a stopped
  or reloaded run costs only the chunk in flight.
- **Transcription checkpoints are deleted only after the history row is written** —
  in the app, not in the pipeline, for the same reason: a run stopped between writing
  the transcript and saving it must be able to finish without paying again.
- **"Clean temporary files" is a callback that also empties the uploader** (a new
  widget key). Otherwise the next click saved and extracted the same upload again, the
  WAV download drawn above the button pointed at a deleted file, and a mid-script
  rerun closed open History entries. **It refuses while any transcription runs in the
  process** — in this tab or another: it deletes checkpoints, and a second tab's
  clean-up used to wipe a live run's saved parts, which it then paid for again. Runs
  register in `checkpoints.active_run()` (a module-level counter: `app.py` itself is
  re-executed every run, so state there does not survive). It never deletes anything
  that is or contains the history or model folder, in case `DATA_DIR` points inside
  `TEMP_DIR`.
- **Notices count the saved parts on disk instead of promising them** — "2 of 4 part(s)
  are saved" comes from the checkpoint folder, so a clean-up or disk error cannot turn
  it into a false "will not be paid for again". The "Partial transcript" banner is
  cleared only by a successful run, so a retry that fails early still marks the partial
  text that is on screen.
- **Messages from a run survive its rerun** — errors, warnings and hints go to
  `session_state.run_notices` and are drawn under Start, because the page reruns as
  soon as the job ends.
- **The progress placeholder is created by the run that uses it.** One kept in
  `session_state` pointed, from the second run on, at whatever element now sat at its
  old position: the completion message vanished, and after a layout change the
  progress box replaced a widget.
- **An upload is identified by `file_id`, not its name** — phones and recorders reuse
  names, and a new `call.wav` used to be ignored in favour of the old one, then
  transcribed, billed and saved under the new upload. Clearing the uploader keeps the
  last result on screen.
- **The player gets a small MP3, not the working file** — videos, AMR/WMA/AIFF (which
  browsers cannot play: all 28 phone calls in the history are AMR) and uploads over
  25 MB get a 32 kbps mono preview (LAME's fastest mode: ~5 s for 90 minutes). The full
  WAV was re-read and hashed on every rerun; the extracted-WAV download is now a
  callable, read only on click. Measured with a 90-minute recording: the audio block
  went from **329 ms to 25 ms per rerun**.
- **History entries are lazy** — `st.expander(key=…, on_change="rerun")` and only the
  open one is read from SQLite. Every rerun used to fetch and render all transcripts
  (1.6 MB of text at 104 rows). Measured with 104 synthetic rows: **281 ms → 14 ms per
  rerun** server-side, and 0 instead of 104 text boxes sent to the browser.
- **CSS hack for the uploader label** — Streamlit prints the full list of 27 extensions; it
  is hidden and replaced with short text. The element to target is a **`<span>`** (not
  `<small>` — established by inspecting the DOM).

### Data
- **SQLite lives in `data/`, not `temp/`** — history **survives** "Clean temporary files"
  (which wipes `temp/` and `uploads/`). That is the whole point of having it.
- **PRAGMA `journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON`** — better defaults
  for Streamlit's rerun model and multiple open sessions.
- **Migration without a migration tool:** `init_db()` checks `PRAGMA table_info` and adds
  any missing column — currently `provider` and `elapsed_seconds` — so existing
  databases are not lost. New columns must be nullable, since existing rows have
  no value for them.
- **Audio is NOT stored as a BLOB** — only the transcript and SRT text; `audio_path` is a
  best-effort reference that may disappear after a cleanup.
- **Run time is measured in `app.py`, around the whole click**, not inside the
  pipeline: on-screen context can dominate the wait, and the number worth
  reporting is how long the user actually sat there. Stored per run in a new
  `elapsed_seconds` column — deliberately not the unrelated (and still unused)
  `duration_minutes`, which means the *audio* length. Measured with
  `time.monotonic()`, not `datetime.now()`: it is a duration, so a clock change
  must not corrupt it.

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
- **The app listens on localhost only** *(2026-09-25)*. Transcripts are confidential and
  there is no login, yet Streamlit binds every interface by default and the firewall let
  the LAN in. `.streamlit/config.toml` sets `server.address = "localhost"` (both
  loopbacks — verified with `netstat`: 127.0.0.1 and ::1 only) and turns off usage
  statistics; docker-compose publishes `127.0.0.1:8501`. The Dockerfile keeps
  `--server.address=0.0.0.0` because inside a container that is required, and the CLI
  flag beats the config file. Exposing it is a deliberate flag, documented in README.
- **`make sync` keeps the offline engine** *(2026-09-25)*. `uv sync` is exact: it removed
  faster-whisper after a dependency change, and the Local engine vanished for two
  months. The Makefile's `EXTRAS` adds `--extra local` when the package is installed or
  models are downloaded (`:=`, so `reset` decides before deleting `.venv`); CI installs
  the extra in a separate job so a broken lock for it is caught.

---

## 4. Done

**Features**
- Upload **video** (16 formats) and **audio** (11 formats, incl. AMR); video audio is extracted automatically
- **Audio player** before and after transcription
- **Two engines**: OpenAI API (`gpt-4o-transcribe` / `whisper-1`) and **Local offline** (faster-whisper, selectable model size)
- **Timestamps + `.srt`** export (whisper-1 and every local model)
- **Readable transcript**: inline `(M:SS)` + paragraphs *(2026-07-20)*
- **On-screen context** for video: key frames described by a vision model and placed
  in the transcript by time — exactly with whisper-1 or the local engine, at
  approximate `(~M:SS)` paragraphs with the default gpt-4o-transcribe *(2026-09-25)* —
  with a selectable screenshot interval and a pre-run estimate of the screenshot count
  and cost *(2026-07-20)*
- **Reliable long runs** *(2026-09-25)*: 5-minute chunks cut in pauses (no silent
  truncation at the output cap), checkpoints so a failed or stopped run resumes without
  paying again, partial transcripts clearly marked, one-sentence errors for a bad key
  or an empty balance, controls locked during a run with a watchdog after Stop
- **Serbian always in Latin script** *(2026-09-25)*; one-off backfill script for old records
- **Playable previews** for AMR/WMA/AIFF, video and large files; fast History at 100+
  records *(2026-09-25)*
- **Run time reported** next to the transcript and in history *(2026-07-28)*
- **Persistent history** (SQLite): browse, re-download TXT/SRT, delete
- **Hybrid API key** (`.env` or sidebar); offline works with no key
- **Wide layout + tabs** (Transcribe / History), two-column arrangement

**Quality / infrastructure**
- Modular refactor (config/audio/transcribe/db/exceptions/logger), type hints + docstrings
- **ruff: 0 errors** (down from 74), `ruff format` clean; modern ruff config (`[tool.ruff.lint]`, `target-version=py312`, plus D/RUF/PTH/T20/S)
- **pytest: 152 tests** (DB CRUD + PRAGMAs + schema migration, SRT/formatting helpers, frame selection and dedup, count/cost estimates, `.env` overrides, the OpenAI pipeline and vision step against fake clients and a fake network, checkpoints, transliteration, the backfill, headless UI flows with AppTest) — no network, no ffmpeg; the regression tests were each checked to fail with their bug put back
- **CI** (GitHub Actions): ruff + format check + pytest on Python 3.12 and 3.13 (`uv sync --locked`), plus a job with the offline engine installed
- **Makefile**: `run` / `sync` (keeps the offline engine) / `sync-local` / `lint` / `format` / `test` / `check` / `clean` / `reset`
- **Localhost-only by default** (`.streamlit/config.toml`, compose `127.0.0.1`), usage statistics off *(2026-09-25)*
- **Docker**: `Dockerfile` + `docker-compose.yml` + `.dockerignore` (ffmpeg bundled, offline backend opt-in via `INSTALL_LOCAL=true`)
- **Launchers**: `scripts/run.bat`, `scripts/run.sh`, `scripts/create-shortcut.ps1` + icon
- **README**: badges, full-width screenshots, offline mode, Docker, quick launch
- Repo is **public**, history cleaned, no `Co-authored-by` trailers
- **Full review** of code, live UI, security, dependencies, API options, docs and real
  usage *(2026-09-25)* — 62 verified findings, prioritised in §5; Streamlit 1.64 assessed

---

## 5. Next steps

The verified backlog from the full review of 2026-09-25 (see the journal entries for
method and measurements). Ids such as R01 refer to that review; every item survived
2–3 independent skeptics, and severities and fixes below are the skeptics' corrected
versions, not the first claims. Effort: S / M / L.

### Done 2026-09-25 — P0 and the owner's P1–P3 table

Approved as "do the whole table"; implemented, reviewed by four adversarial reviewers
(32 findings; all fixed except the low ones recorded under P4), the fixes verified by a
second round (3 reviewers + a skeptic per finding: 7 more, all fixed) and checked live on
an isolated instance.
Reasons and measurements are in §3 and the journal.

- [x] **Offline engine stays installed** (R10) — Makefile `EXTRAS`, `sync-local`, a
      sequential `reset` whose `clean-venv` can delete `.venv` on Windows, CI job with
      the extra (and its tests).
- [x] **Localhost only, no usage statistics** (R08 + R50) — verified with `netstat`.
- [x] **Same-name upload** (R06) — keyed on `file_id`; upload names made path-safe.
- [x] **Progress placeholder per run** — plus progress reported as finished/total and
      elapsed time on every spinner (R21).
- [x] **No futile retries, one clear message** (R04) — one request on an empty balance,
      measured against the live API.
- [x] **Output-token cap** (R01) — 5-minute chunks for gpt-4o, cap guard that splits
      once and marks what may still be cut off.
- [x] **Chunk edges** (R13 + R14) — cuts in pauses, tiny tails merged, size heuristics gone.
- [x] **Serbian script** (R02) — Latin in every new transcript; the 10 affected saved
      records converted with the owner's OK (backup
      `data/transcriptions.db-backup-2026-09-25-145853-serbian-latin`; the other 92 rows
      byte-identical, integrity ok).
- [x] **Paid work kept** (R03 + G01) — chunk, half-chunk and frame checkpoints; partial
      transcript with a marker; resume hint and notices counted from disk; pruned after
      14 days; clean-up refused while any run is live.
- [x] **A click no longer aborts a run** (R05) — controls locked, History in a fragment,
      downloads `on_click="ignore"`, watchdog after Stop. The background-worker option
      was not needed (checkpoints make a stopped run cheap to resume).
- [x] **History fast** (R18) — lazy entries: 281 → 14 ms per rerun at 104 rows.
- [x] **WAV no longer re-read per click** (R19) — MP3 preview + lazy download: 329 → 25 ms.
- [x] **AMR/WMA/AIFF play** (R22) — verified: AMR duration 56.4 s, previously NaN.
- [x] **Notes placed by time with the default model** (R12) — approximate `(~M:SS)`
      paragraphs; help text and README say where it is exact.
- [x] Partly: per-run scratch folders for chunks and frames (R07); CI `--locked` and a
      3.12 + 3.13 matrix (R33); the dead `default_model` setting removed and
      `SEGMENT_DURATION_MINUTES` wired (R46); fake-client tests for the paid pipeline
      (R52); automatic pruning of checkpoints (R48).

### Waiting for the owner
- [ ] **OpenAI credit is not visible to the key.** After the owner's top-ups the API still
      answered `429 credit_balance_exhausted` (2026-09-25 06:18, 07:10, and the owner's own
      run on a 1:55 video in the afternoon). None of the agents' calls that day got through
      (every paid request was rejected, so nothing was billed). The key is a project key
      (`sk-proj-`): check in the dashboard that the credit went to the organization that
      owns that project, that the payment completed, and that the project has no $0 budget.
- [ ] **Live check of the OpenAI path once credit works** — only with the owner's OK for
      that spend (a few cents, synthetic fixtures only): `usage.output_tokens` is present in gpt-4o-transcribe answers (the cap guard
      depends on it), `(~M:SS)` + notes on `meeting.mp4`, and a resumed run after a forced
      failure.

### P1 — transcript correctness
- [ ] **Migrate to `gpt-transcribe` before 2027-02-26** (deprecation + R26, M). WER check
      on the fixture and on one real Serbian meeting judged by the owner, then make it the
      default; `languages=["sr","en"]`, a Latin-script `prompt`, an optional keywords
      field for names and jargon. Chunk length stays 5 min until its output cap is known;
      the `(~M:SS)` placement already works without timestamps. Decide what replaces
      whisper-1 timestamps/SRT (local engine). The diarize model is retired too: dropped.
- [ ] **Repetition loops** (R15, S). A pure `detect_repetition()` calibrated on history
      (same word > 30× in a row; periodic runs) that warns per chunk; 3 saved transcripts
      contain loops. The cap guard now marks an answer that loops to the cap, but not a
      shorter loop.

### P2 — responsiveness and UI
- [ ] **Transcript box** (R20, S). `st.code(text, language=None, wrap_lines=True, height=…)`:
      read-only, scrolls, has a copy button (the editable text_area silently drops edits).
- [ ] **Delete needs a confirmation** (R11, S). `st.popover` with "Delete permanently",
      placed to the right of the download buttons.
- [ ] **Search in history** (R23, S–M). Filename and text, with diacritic and script folding.
- [ ] **Parallel API calls** (R16, M). Thread pools for chunks (3–4) and frames (6–8),
      progress reported on the script thread; start frame extraction while chunks run.
      Twice as many chunks since R01 make this more worthwhile.

### P3 — on-screen context quality
- [ ] **Burst thinning keeps the wrong frame** (R09, S). `apply_min_interval` should keep the
      last, settled frame of a burst, not the first.
- [ ] **The caption prompt describes the call window** (R29, S). Describe only shared
      content; NONE for a participant grid (12–24% of notes are only about the call UI).
      Bump `vision._CACHE_VERSION` with it.
- [ ] Smaller: a pixel-diff second opinion for panel-sized changes dHash misses (R28);
      repeated screens re-captioned (R40); downscale frames to 512 px, ~83% less upload
      (R42); honest time/cost estimate and recorded usage (R39); faster scene scan (R41).
- [ ] **Local OCR as a cheaper visual layer** — reads slide text for zero tokens and fixes
      the "slides differing only in text" dedup limit; adds a second system binary.

### P4 — hygiene
- [ ] Rest of R07: transcript output files (`temp/transcript_<name>.txt`) and uploads are
      still named after the upload, so two tabs with the same file name share them.
- [ ] `make backup` via `VACUUM INTO` / `Connection.backup()` plus an integrity check (R24);
      consider a backup before automatic schema migrations (R57).
- [ ] **Streamlit 1.64.0** — safe to adopt (assessed 2026-09-25, see journal); needs
      `wrap=True` on the checkboxes/buttons directly in the left column. If staying on
      1.56, bump tornado and pillow instead (R31). Re-check the job/fragment/watchdog
      behaviour (§3 UI) on the new version.
- [ ] Config and repo: `load_dotenv(override=True)` → `env_file=BASE_DIR / ".env"` (R34);
      `.env.example` without a fake key (R37); duplicated constants (R46 rest);
      `.gitattributes` with `*.sh text eol=lf` (R54); current action versions and a pinned
      setup-uv (R33 rest); Docker cache, non-root user, `.env` passing and a `temp/` volume
      so checkpoints survive a recreated container (R53); openai SDK bump and jiter for
      Python 3.14 (R27); drop pydub (R17); refresh the README screenshots (R35); privacy
      wording and README fixes (R47, R55); auto-clean old uploads (R48 rest); Host
      allow-list (R30).
- [ ] **A committed live-check harness.** Today's browser checks used a copy of the app
      with a fake, slow OpenAI client (every request logged with its thread) and
      puppeteer scenarios, kept only in the session scratchpad. Committing them (e.g.
      `scripts/dev/`, paths from env vars) would make "check it live" mechanical.
- [ ] Review leftovers (low): a stalled upload can wait 4 × 300 s before failing; the SDK
      may resend a chunk after a read timeout (possible double billing); a genuine Russian
      passage inside a Serbian meeting is transliterated too; Macedonian counts as Serbian;
      the developer menu's "Clear cache" stays usable during a run and clears session
      state; a Stop landing in the tens of milliseconds between Start and the job leaves
      it to run on the next rerun.

### P5 — features and later
- [ ] **Meeting summary and action items** (R38, M). An explicit button, gpt-5.4-mini, stored
      in a new nullable `summary` column; hidden for very short transcripts.
- [ ] **Hosting** *(researched)* — Hugging Face Spaces or an Oracle Always Free VM; users
      must bring their own OpenAI key. Needs a **concurrency limit + queue** if the offline
      engine is ever public.

---

## 6. Journal

### 2026-09-25 (part 2) — The owner's table: P0 plus the chosen P1–P3 items

**Asked:** "do the whole table" — the ten user-visible problems from the review (three
P0, three P1, three P2, one P3), on top of the already approved P0. All of it is
implemented; what each fix is and why is in §3, the statuses in §5.

**How:** code first, then `make check`, then live checks on an isolated instance
(`DATA_DIR`/`TEMP_DIR`/`UPLOAD_DIR` in the scratchpad, port 8599, synthetic fixtures,
puppeteer), then four adversarial reviewers with separate lenses (pipeline, UI flow,
errors/security, build/tests). They reported 32 findings, all reproduced; every one is
fixed except the low leftovers in §5 → P4. Every regression test was then checked by
putting its bug back and watching it fail: 21 of 22 did; the 22nd pointed at an
`except` that could never trigger, which was removed. The real history, uploads and
temp folders were snapshotted before and after all live checks — byte-identical.

**Measured, not assumed:**
- *Bind address:* `netstat` shows 127.0.0.1 and ::1 only.
- *Empty balance:* one request and one sentence, against the live API — before: up to 9
  requests per chunk and 3 per frame, raw JSON twice.
- *Reruns with a 90-minute recording:* audio block 329 → 25 ms (MP3 preview 5.2 s once,
  LAME's fastest mode; the default took 9.1 s and Opus 63 s).
- *History with 104 rows:* 281 → 14 ms per rerun, no transcript sent until opened.
- *AMR:* the player now reports 56.4 s and plays (before: `NaN`, silent).
- *Clicks during a run:* a History entry opened 5 s into a 33 s local run did not stop
  it; Stop unlocks the page once the request in flight has returned.
- *Resume:* a run stopped during its last chunk resumed with zero requests.

**Found by the review that I had got wrong:**
1. The watchdog treated a stopped run as gone. Streamlit's Stop only acts at the
   thread's next Streamlit call, so the page unlocked while the paid request was still
   running — Start then sent the same chunk again in parallel, and a thread stopped in
   its last chunk deleted the checkpoints. Fixed with a thread-aware job (§3).
2. Two tabs shared `temp/segments` and `temp/frames`; with the new caches that meant one
   video's screens stored under another's hash. Each run now has its own folder.
3. A failure in the second half of a split chunk threw away the paid first half, and a
   looping answer could be split down to 4× the cost. Halves are checkpointed; one split.
4. A 403 from the vision model would have cancelled a transcription that works.
5. Smaller: the retry hook could crash on a non-standard 429 body and report "no
   connection"; an empty answer raised `IndexError`; `make clean-venv` could not delete
   the venv it was running from on Windows; `make help` broke under sh; tests read the
   developer's `.env`; Windows upload names like `x/D:evil.mp4` resolved outside
   `uploads/`.

**Also mine, caught before the review:** the first R19 fix still handed large playable
uploads (a 150 MB WAV) to the player; the watchdog's first version cancelled every job
because a fragment's body also runs inline when it is drawn (AppTest showed it at once);
a heredoc turned `[\\/]` into `[\/]` in the upload-name regex.

**Not verified live, because the account still has no credit:** the OpenAI API answered
`429 credit_balance_exhausted` after the $10 purchase (06:18 and again 07:10). So the
gpt-4o path — `usage.output_tokens` in real answers, `(~M:SS)` with notes, resume after a
real failure — is covered by fake-client tests only. §5 → "Waiting for the owner".

**Second review round, before the commit** (ultracode: 3 reviewers on the fixes made
after the first review, a skeptic per finding reproducing it; no real OpenAI calls,
isolated folders). 7 findings confirmed, none refuted: a second tab's clean-up deleted
a live run's checkpoints (and the notice then promised they were saved); paid
screenshot descriptions were dropped when the vision step stopped early; a resumed
vision step could trip over old bad frames; scratch folders of a killed process were
never removed; a failed retry hid the "Partial" banner; Clean closed open History
entries; Clean could delete a history folder placed inside TEMP_DIR. All fixed, each
with a regression test that fails when the bug is put back (9 of 9), and the two-tab
case checked in a browser. 152 tests.

**Later the same day:** the owner approved the Serbian backfill — 10 records converted
after a verified backup, the other 92 byte-identical — and the commit. The owner's own
test on a 1:55 video hit the empty balance as well; a scan of every transcript of this
session (main and all agents) found no successful paid request, so the missing credit
is a billing question, not spending by the tests. From now on agents never use the
owner's key for tests (fake clients only). Still open: where the credit went.

**Conclusions worth keeping:**
- In Streamlit, "Stop" and every widget change stop a script only at its next Streamlit
  call. Anything that pays per request must assume the stopped thread finishes that
  request — keep the job locked to its thread and save every paid result immediately.
- A fragment body runs inline when drawn and later only when no full run is active; that
  second property is what makes a watchdog possible, the first is what breaks a naive one.
- The adversarial review again changed the outcome: one high-severity bug (double
  billing after Stop) existed only because of the new design, and only a reviewer
  reproducing it in a real browser, not the AppTest suite, found it.

### 2026-09-25 — Full review of code, UI and real usage; Streamlit 1.64 assessed

**Backup first:** `data/transcriptions.db-backup-2026-09-25`, made with SQLite's backup API
(not a file copy) and verified — 104 rows, integrity ok, identical content hash.

**Method.** Nine independent reviewers, each with one lens (pipeline, on-screen context,
a live UI walkthrough with Playwright, UI robustness, security, dependencies/CI,
OpenAI API options, docs, real usage from DB aggregates), all read-only against the
repo, with isolated app instances and synthetic fixtures (TTS speech, a slide video,
an AMR call). 95 findings → 58 after merging duplicates → each checked by 2–3 skeptics
told to refute it → a completeness critic added 4. None was refuted outright, but 77 of
112 votes were "partially": most severities came down (a local single-user app mostly
loses time and cents, not data) and many proposals were corrected. §5 holds the
corrected versions. Private data stayed private: the DB was opened read-only with
aggregates only, real recordings were analysed locally and never sent anywhere, and
afterwards the DB matched the backup and git was untouched.

**Situation found, beyond the code:**
- **Nothing can transcribe today.** The OpenAI account is out of credit (429
  `credit_balance_exhausted`; every reviewer call was rejected, so the review cost $0),
  and the offline engine is not installed.
- **OpenAI retires every model this app uses on 2027-02-26**, and the replacement
  `gpt-transcribe` returns no timestamps (§3).

**Measured, not inferred:**
- *Truncated transcripts.* gpt-4o-transcribe stops at ~2,048 output tokens per request.
  Transcribing three 10-minute boundaries of real recordings locally and matching 30-s
  windows against the saved text: a Serbian chunk dropped to 12% overlap in its last
  30–45 s, then recovered to 71–83% in the next chunk; the other Serbian boundary and an
  English one stayed at 55–74% and 88–100%. So the loss is real but density-dependent,
  not "the end of every chunk" — an early estimate of 20–30% per meeting was wrong.
- *Every click is slow with a long recording loaded:* ~0.5 s per rerun just re-reading a
  152 MB WAV for the player and download button, plus 0.8–2.9 s rendering the hidden
  History tab with 104 transcripts.
- *The progress placeholder cached in `session_state` misrenders* from the second run in
  a session (AppTest): the completion message disappears, and after a layout change the
  progress box replaces a widget.
- *Real usage:* ~2,400 audio minutes over 5 months (~$14); 23 runs with on-screen
  context produced 1,174 notes, **all** appended after the transcript because the default
  model has no timestamps; all 28 phone calls are .amr, which the browser player cannot
  play.

**My own earlier mistakes this review surfaced:**
1. The `uv sync` I ran on 2026-07-20 after adding Pillow almost certainly removed the
   optional faster-whisper backend — its output listed faster-whisper's dependencies
   (tokenizers, typer, …) among the removals — and the Makefile's `sync` and `reset` do
   the same. The Local engine has been hidden since; the hint that would have
   said so had just been removed from the UI.
2. History rows 48 and 49 (2026-07-28 14:27 and 14:30, "tiny", 0.006 s, test file names)
   were written into the real DB by a verification agent that called `run_transcription`
   without an isolated `DATA_DIR`. Deleted later the same day with the owner's approval
   (see below). This review's agents were given isolated directories and the DB was
   proven unchanged.
3. Adding the 2026-07-28 journal entry deleted the heading of the 2026-07-20 (part 2)
   entry, so the on-screen work read as 07-28. Restored today, and the cap-fix paragraphs
   moved to their own 07-28 entry to match the commit dates.

**Streamlit 1.64.0** (asked by the owner). Assessed with release notes 1.57–1.64, a live
side-by-side run of 1.56 and 1.64, and a skeptic. Nothing the app calls was removed and
all 68 tests pass. Gains are modest: ~25–30% faster reruns with big media (partial
hashing), security options (`server.allowedHosts`), AppTest support for download
buttons, and tornado — with its open CVEs — is gone (the server became Starlette/Uvicorn
in 1.57, not 1.64). Costs: ~0.5 s slower first page load (reproduced with a hello-world
app, so it is Streamlit core), smaller widget fonts, and checkbox/button labels directly
in `st.columns` now ellipsize at laptop widths unless `wrap=True`. It fixes none of the
app's own bugs (placeholder, click-abort, AMR, default all-interfaces bind, telemetry).
Verdict: upgrade with those fixes, as the first step of the fix series; no urgency.

**Later the same day — P0 approved and started.** The owner approved the P0 group and
the deletion of rows 48–49, and topped up the OpenAI account ($10; bought by the owner,
not by the agent). Done so far:
- Rows 48 and 49 deleted by a script that refused to act unless both rows still matched
  every test-artefact trait (2026-07-28 14:2x/14:3x, Local, `tiny`, < 0.01 s, 29 chars,
  a test file name): 104 → 102 rows, integrity ok. The morning backup still holds them.
- Offline engine reinstalled with `uv sync --extra local` (faster-whisper 1.2.1,
  ctranslate2 4.7.2).

No application code has changed yet; the resume point, with the approach decided for each
P0 item, is §5 → P0.

**Conclusions worth keeping:**
- The app's biggest risks were operational and silent, not code bugs: an uninstalled
  engine, an empty balance, a LAN-exposed history, a truncation nobody could see. Each
  was found only by measuring against real data — keep verifying on the owner's real
  recordings (locally), not only on fixtures.
- Adversarial verification changed the result more than the finding did: 77 of 112 votes
  corrected severity or proposal. Report the corrected versions, never the first claims.
- The OpenAI deprecation reorders everything: build P1 on `gpt-transcribe`, and treat the
  local engine as the long-term source of timestamps rather than an optional extra.

### 2026-07-28 — Report how long a run took

The wait had no feedback beyond a spinner, and with on-screen context a run can
take minutes. Each run now reports its wall-clock time next to the transcript
(`⏱️ Finished in 2:34`) and in the history list.

Measured in `app.py` around the whole click rather than inside the pipeline, so
frame extraction and captioning are included — that is the wait the user
experiences. Persisted in a new `elapsed_seconds` column; the existing
`duration_minutes` column was left alone because it means the audio's length, not
the processing time, and conflating them would have been a silent lie in the data.

While here, replaced the two progress messages that printed a raw
`datetime.now()` — "Transcription completed at 2026-07-28 11:16:08.813456" —
with "Transcription completed in 0:03". The timestamp was noise; the duration is
the useful part. That removed the last use of `datetime` in `transcribe.py`.

**Migration verified on the real database, not just a fixture:** a copy of the
owner's 45-record history was migrated first (all records intact, old rows get
`None`, new rows accept the column), then backed up before the app touched the
original. A regression test now recreates the pre-`provider` schema, migrates it,
and asserts the existing rows survive — the previous migration path had no test
at all.

**Two defects found while reviewing the change:**

1. **A failed run kept the previous run's time.** `run_transcription` assigned
   `elapsed_seconds` only on success, so the no-API-key early return and the
   `except AppError` path left the old figure in place — and the result panel
   then claimed "Finished in 2:34" next to a red error, about a run that never
   finished. Worse, a partial failure rewrites the `.txt` before the `.srt`, so a
   *new* transcript could be shown stamped with an *old* duration. Fixed by
   clearing it before the timer starts. The regression test was checked by
   removing the fix and confirming it fails.
2. **`init_session_state()` had drifted out of sync.** It listed four keys while
   the two reset paths listed six; `video_path` and `elapsed_seconds` were never
   initialised and only existed by accident, because the new-upload reset happens
   to run first. Three hand-maintained copies of one list is the actual bug, so
   they now derive from a single `_RUN_STATE_KEYS` tuple, with a test asserting
   every key is initialised.

Tests: 68 (was 61).

### 2026-07-28 — Screenshot cap fix and a configurable cap

The on-screen context caption gained the expected screenshot count next to the
cost — and that count is what exposed the real bug. On an 83-minute meeting the caption
read "About 40 screenshots… never more than 40" and did not move as the slider was
dragged: at 40 frames the cap bound at every position below ~125 s, so the control
did nothing on exactly the long recordings it was meant for. Two things were wrong
at once — a cap set for a cost that turned out to be trivial, and a message that
reported the clamp as if it were an estimate. The cap is now 200 and is applied by
widening the interval before extraction, and when it does bind the caption states
the interval that will really be used. Also corrected the cost estimate, which
omitted output tokens and so read about half the true figure.

**Then made the cap configurable** (`FRAME_MAX_COUNT` in `.env`, owner runs 300).
Chose an env setting over a second UI control deliberately: interval and frame
count both determine the same outcome, so a second widget would give two knobs
where one always silently wins — the exact confusion just fixed. The env route
keeps one control on screen and puts the budget where it is set once and
forgotten. Making it work needed more than a Settings field: every call site took
the cap as a default argument, which is evaluated at import and would have
ignored `.env` entirely.

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
covered there instead. The cap shipped at 40 frames; see the correction below.

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
