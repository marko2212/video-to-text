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
| `recordings.py` | Recordings picked from a folder on this computer instead of uploaded: reading a pasted folder or file path, listing media newest first, their lengths (ffprobe, cached per version), stable list labels, a file's identity — UI-agnostic |
| `frames.py` | One-pass video scan (scene changes + 5 s grid, dHash), selection from the scan, single-frame extraction |
| `vision.py` | Describes those frames with a vision model (cached per frame); cost estimates for the UI |
| `transcribe.py` | Pipeline: `transcribe_openai` (chunked, checkpointed) and `transcribe_local`; transcript rendering |
| `openai_api.py` | The OpenAI client (retry/timeout policy) and SDK errors → short app errors |
| `checkpoints.py` | On-disk results of paid requests, keyed by content hash, so failed runs resume; also the age-based clean-ups of `temp/` and `uploads/` and the space they take |
| `serbian.py` | Serbian Cyrillic → Latin transliteration (pure functions; opt-in) |
| `titles.py` | AI title of a finished transcript: the chat request (with the transcript's script named, and one retry for a title in a foreign script), cleaning the answer into a safe file name, how a title is shown per mode |
| `usage.py` | Per-request token and cost records (transcription, vision, title), totals, USD formatting; pure functions |
| `db.py` | SQLite history (`data/transcriptions.db`) and the sidebar preferences stored with it |
| `exceptions.py` | Domain exceptions (`AppError` → `AudioProcessingError`, `TranscriptionError` → `IncompleteTranscriptionError`, `VisualContextError`, `TitleError`, `OpenAIAccountError`) |
| `logger.py` | `get_logger()` — stdlib logging (never `print()`) |
| `scripts/` | Launchers: `run.bat` (Windows), `run.sh` (Linux/macOS), `create-shortcut.ps1`; one-off `serbian_latin_backfill.py` |
| `tests/` | pytest — pure functions, SQLite, the pipeline with a fake client, headless UI (AppTest); no network, no ffmpeg |

### Data flow
```
upload (identified by file_id, saved to uploads/)
  OR a recording from a folder on this computer (path + size + mtime; read in place)
       → (video? ffmpeg to_wav mono-16k : use the file as-is)
       → audio player (MP3 preview for video, AMR/WMA/AIFF and files > 25 MB)
       → pick engine + model  [+ optional on-screen context, video only]
       → Start: request stored, page redrawn with every control locked, job runs
       → (on-screen context: one ffmpeg scan per video (temp/scan-<hash>) →
          select_from_scan for the slider → a frame whose JPEG is shared is taken
          from the video again at its own time → vision model, each description
          cached under temp/checkpoints/)
       → OpenAI (5-min chunks for gpt-4o, 10 for whisper-1, cut in pauses; each
          chunk checkpointed; an answer at the output cap is split and redone)
          OR  local (whole file at once)
       → rendering: (M:SS) or (~M:SS) + paragraphs, on-screen notes placed by time,
          Serbian Cyrillic → Latin only with SERBIAN_LATIN=true → .txt (+ .srt if requested)
       → (AI title, if on: a chat model names the finished transcript — after either
          engine; a title in a script the transcript never uses is sent back once;
          a failure only warns. History can title a saved row later.)
       → write to SQLite history (a partial run is shown, marked, and not saved)
```

### How the two engines differ
| | OpenAI API | Local (offline) |
|---|---|---|
| API key | required (`.env` or sidebar) | **not needed** |
| Chunking | **yes** (5 min for gpt-4o — output cap; 10 for whisper-1) | **no** (whole file) |
| Timestamps | exact with `whisper-1` (`verbose_json`); approximate `(~M:SS)` with gpt-4o | **always** (native) |
| Resume after a failure | **yes** (chunk checkpoints) | no (one pass) |
| Cost | per token (gpt-4o-transcribe; per minute only as a `≈` estimate when no usage is reported) or per minute (whisper-1); recorded per run | free (costs CPU time) |

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
- **No LiteLLM; if a second provider comes, a thin adapter per vendor** *(2026-09-25,
  researched with sources and a skeptic pass)*. LiteLLM v1.102.1 does route
  transcription to Gemini, Deepgram, ElevenLabs, Soniox, Groq and others, but it loses
  what this app needs: its Gemini path cannot send custom vocabulary and turns on
  diarization with word timestamps (30-minute cap), its Deepgram path never sends the
  `language` parameter, its Azure path uses the short-audio API, speaker labels are
  dropped from the normalized output, it requires `openai>=2.20` (the app pins 1.75),
  and two releases were compromised on PyPI in March 2026 — a poor fit for an app
  holding API keys and confidential meetings. The app's own logic (25 MB chunks, the
  output cap, checkpoints) is tuned per provider anyway.
- **OpenAI is not the only option for Serbian/Macedonian.** Hosted services that
  officially list both sr and mk and return timestamps (which `gpt-transcribe` does not):
  ElevenLabs Scribe v2 ($0.22/h, 10 h per file, diarization, 1,000 keyterms; training on
  content unless opted out), Google Gemini 3.5 Transcribe (~$0.30/h; public preview; 1 h
  per request, 30 min with timestamps), Deepgram Nova-3 ($0.26/h; `mip_opt_out`), Soniox
  ($0.10/h; no training), Azure fast transcription ($0.36/h; Serbian returned in Cyrillic;
  no retention), Amazon Transcribe ($0.36/h). Ruled out for sr/mk: AssemblyAI (Serbian
  rated >50% WER), Mistral Voxtral and Speechmatics (no sr/mk), Google Chirp 3 (preview,
  no diarization). No neutral sr/mk benchmark exists; vendor tiers only. Whisper
  large-v3 (the local engine's best model) scores FLEURS WER sr 11.6 / mk 14.7; the
  local default `base` is far weaker for South Slavic. See §5 → P1.
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
  The price follows the audio length (per minute for whisper-1, audio tokens for
  gpt-4o-transcribe), so it costs nothing extra. `SEGMENT_DURATION_MINUTES`
  overrides it for every model (1–15; 15 keeps a chunk under 25 MB). Unknown models get
  5 until their cap is known.
- **An answer at the cap is split and redone — once** — `usage.output_tokens` ≥ 1,900
  means the end of the chunk is missing, so its two halves are transcribed instead (each
  half checkpointed on its own). A half that fills the cap again is a loop or noise, not
  speech (dense Serbian is ~730 tokens per 2.5 min), so it is kept with a visible
  "⚠️ … may be missing" line rather than split further — unlimited splitting cost up to
  4× on a looping chunk. With 5-minute chunks the guard should rarely fire. `usage` is read from the response (SDK 1.75 keeps unknown fields);
  confirmed live on 2026-09-25: gpt-4o-transcribe returns it (300 input / 79 output
  tokens for a 30 s clip). If an answer ever lacks it, the guard stays silent and the
  shorter chunks are the protection.
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
- **Transcripts keep the script the model returned; Serbian → Latin is opt-in**
  *(2026-09-25)*. Without a language hint the model picks the script per chunk, so a
  Serbian meeting can read LLL CCC LLL in chunk-sized blocks (7 saved records). A
  deterministic 1:1 transliteration exists (`SERBIAN_LATIN=true`): it runs when the text
  is recognisably Serbian — Serbian-only letters (ђ ћ џ љ њ ј) outnumbering letters
  Serbian never uses. It was on by default for a few hours and was turned off, and the
  10 converted records put back, at the owner's request: Macedonian colleagues use the
  app, Macedonian shares exactly those letters and would be rewritten in *Serbian*
  Latin (ќ→ć, ѓ→đ), and a transcript read by another AI does not need one script. No
  per-run switch in the UI either (the owner's call).

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
- **Before the run the caption gives a ceiling; the run scans the video and describes
  its distinct screens** *(2026-10-03, the owner's call)*. From 2026-09-25 ticking the
  box scanned the video so the caption could give the exact count — and the slider
  waited for it, about a minute and a half per hour of video. The owner did not want
  that wait, and noted that a maximum is always known. The caption now says, at once:
  at most one screenshot per look (length ÷ interval, + 1) — the count for slides,
  since ffmpeg's change detection rarely fires on them (a full-screen slide change
  scored 0.077, under the 0.1 threshold) — plus what faster changes can add, up to one
  per 2 s (`FRAME_MIN_INTERVAL_SECONDS`, the burst window) and never past the cap
  (`FRAME_MAX_COUNT`; 300 on the owner's machine: $0.18 on nano, $0.67 on mini), with
  "usually far fewer". It is priced for the video's own frame size (ffprobe; Full HD
  only when unknown): frames are described at full size, and a 4K screen costs about
  3.4 times the Full HD figure (pre-commit review). A prepared video whose file states
  no length gets the per-video cap and says so. Its prices are written `\$`: two `$`
  in one caption made Markdown show the text between as a formula, signs dropped
  (seen live). A stored
  scan (an earlier run of the video) still gives the exact count while a job runs; it
  is not looked up before Start, since naming it means hashing the whole video. The
  slider's help now says what it sets: how often the screen is *looked at*.
- **The screenshots described are the distinct screens a scan finds** *(2026-09-25)*.
  The count used to be the video's length over the interval, which is not what gets
  described: on four real recordings a static 115-minute meeting estimated at 231
  screenshots described 10, and a busy 16-minute screen share estimated at 4
  described 28 — the number described is the number of *distinct screens*. The run
  scans the video once (`frames.scan_video`): one ffmpeg pass keeps
  every scene change plus a frame every 5 s (the slider's finest step), with each
  frame's scene score and perceptual hash, and a run of near-identical frames keeps
  one JPEG (the static 115-minute meeting: 1,390 frames, 2 MB instead of 253 MB).
  `frames.select_from_scan` then gives the selection for any slider position in
  under a millisecond — every scene change, the first grid frame after each
  interval, one frame per burst, repeats dropped, the cap — and **the run describes
  exactly that selection**. The scan costs about the time the old extraction took
  (170 s for 115 min, 23 s for 16 min); it is stored as `temp/scan-<hash of the
  video>/` until clean-up or 24 h, with its threshold and grid, so a scan made with
  other settings is redone. Interval samples may come up to one grid step (5 s) later
  than the exact interval. A scan that fails leaves the run without on-screen notes,
  with a warning.
- **A frame whose JPEG is shared is taken from the video again before it is
  described** *(2026-09-27, review finding S1)*. The shared JPEG is only a disk
  saving: "near-identical" is the 8×8 hash's opinion, and two slides of one template
  hash within 1 bit. Describing the shared JPEG put an earlier screen under a later
  time (a synthetic deck: the 2:00 note described the previous slide; a typing
  editor: pictures up to 100 s old), and a slide that only ever shared a picture was
  never described. Each frame now records `shared`, and `frames.pictures` seeks to
  the frame's own time (`ffmpeg -ss`) when it is about to be described — only for
  frames not in the description cache. Measured with real ffmpeg: the extracted
  JPEG is byte-identical to what the scan wrote at that time (80 of 80 frames on two
  synthetic videos; the review then confirmed it for MP4, MKV, WebM with and without
  cues, AVI, WMV, VFR, edit lists, a start offset and a one-hour file), 0.2–0.3 s per
  frame, so 200 frames add about a minute at worst to a run that spends several
  seconds per description anyway. Keeping every JPEG instead would bring back the
  253 MB. **Not in containers without an index** (MPEG-TS/PS: .ts, .mts, .m2ts, .mpg,
  .mpeg, .vob): there the seek landed up to a keyframe interval late — the next slide
  under the label — or wrote nothing in the last GOP, so the scan keeps every frame's
  own JPEG for them (`FRAME_UNINDEXED_FORMATS`). As a last check, an extracted frame
  whose hash is more than 2 bits from the scan's falls back to the shared JPEG.
- **The cost estimate includes output tokens.** Counting only image tokens
  understated it by about half, because a caption's output tokens are priced
  several times higher than input ones.
- **Image tokens are measured, not assumed** *(2026-09-25)*. Real requests: a
  1920×1080 frame costs 2,519 input tokens and a 1280×720 one 1,175 on gpt-5.4-nano
  and -mini alike — roughly proportional to the pixel count — and `detail` low or high
  makes no difference. The estimate had assumed 630 (half or less). Downscaling frames
  before upload (R42) would now cut the cost roughly in proportion.
- **A burst keeps its last frame** *(2026-09-25, R09)*. Frames less than 2 s apart
  count as one burst; keeping the *first* threw away the scene-change frame whenever a
  slide changed within 2 s after an interval sample — on the test video slide 4 (cut at
  41.5 s) was stamped 46.5 s or 51.5 s, a whole interval late, and a slide shown
  briefly could be lost. Now every interval from 5 to 300 s catches it at 41.53 s, and
  every kept frame shows the slide that was on screen at its time (checked by
  perceptual hash against the slide images).
- **Short videos get tightened, but only just.** A clip shorter than the chosen
  interval would be represented by a single frame, so the interval drops to
  `duration / 2`. An earlier version aimed for ~8 samples, which silently
  overrode the slider on anything short; the guarantee is now the minimum that
  fixes the bug and otherwise leaves the chosen cadence alone.
- **Deduplication runs before anything is sent**, so discarded frames cost
  nothing. Measured on a meeting-shaped video (4 slides held 30 s each): ffmpeg
  produced 30 candidates, dedup kept 4 — the slide boundaries — and 86% of the
  frames never reached the API. The count shown before a run is taken from the
  scan after this dedup (and the cap), so it is the number described; only the rough
  guess shown while a job runs without a stored scan ignores scene changes and dedup.
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
- **The job reads its settings from the page it draws, and owns its run from the
  first line** *(2026-09-27, review findings C4 and C6)*. Start used to capture the
  settings in its click; clicked while the page was still busy (ticking on-screen
  context started a scan of a minute or more, until 2026-10-03), it carried the
  previous run's settings
  and transcribed without screenshots. Now the click only records that a job was
  requested; the job run takes the settings from the widgets it draws (locked, so
  unchanged). The run is also the job's owner from `main()`'s first line, and while a
  job runs the settings panel only *loads* a stored scan, never starts one: a Stop
  during such a scan used to leave a job with no owner — every control locked, and
  the next rerun ran the stopped job.
- **One scan per video at a time** *(2026-09-27, review findings C1/C5/C7)*. Every
  rerun during a long scan started another full decode of the same video; a lock per
  scan folder makes a second caller wait and load the result. While ticking the box
  scanned (until 2026-10-03), a failed scan was also remembered in the session, as it
  ran again on every slider step; the scan now runs only in the run, where a failure
  is a warning.
  The warning shows the last lines of ffmpeg's log, not its 2,000-character banner, and
  a file with no picture stream says so instead of promising screenshots.
- **Costs of work paid in an earlier, unsaved attempt land in a row** *(2026-09-27,
  review findings C9–C14)*. Only the descriptions a saved row used are marked as
  counted; when a complete run discards the description cache, entries no row has
  counted are added to that row. Split markers and halves saved before cost recording
  count as per-minute estimates, not as free; a partial run reports the length its
  transcript covers; a first chunk that fails after a paid split says what it spent;
  an answer without choices still counts its tokens.
- **The dHash rule stays, although it merges same-template slides** *(2026-09-27,
  review finding S3)*. Two slides of one template are 0–1 bits apart, and so are
  frames a few code lines apart, so the second one is never described. A study
  replaced the dHash with three thumbnail-based rules (motion masks for webcams,
  pointer tolerance), each measured on 7 synthetic videos with exact ground truth and
  then cross-checked on 21 more adversarial ones: the two finalists caught every
  reachable change (58/58 states against 45 at slider 5) with 0–1 extra frames. **On
  the owner's real recordings they made things worse.** Counts on five recordings (at
  30 s): 4 → 15 and 33 → 64 described; with the owner's OK, 13 of the disputed frame
  pairs were looked at, and 12 differed only in the Teams UI — the active-speaker
  border moving between tiles, the participant tiles rearranging, a hover preview, a
  page still loading. The participants were avatars, so the motion masks had nothing
  to learn from, and a thin border reads like a new line of text. One finalist also
  described 13 fewer of 45 frames on a busy screen share, because its mask covered 60%
  of the screen. Shipping either would have cost ~500 lines, 10–15 tuned constants
  and a second ffmpeg stream in the scan, for more noise on the recordings that
  matter. What would change this: a rule that ignores the call UI (speaker border,
  tile layout) — see §5 P3.
- **The frame cap is a backstop, not the control.** It was originally 40, which
  was low enough to bind at *every* slider position on an 83-minute meeting —
  the estimate sat at "about 40… never more than 40" and dragging the slider
  changed nothing. Raised to 200: the real constraint is wall-clock (one request
  per frame), not money — 200 Full HD frames are about 12 cents on the default
  gpt-5.4-nano (about 45 on mini; the old "5 cents" assumed 630 tokens per frame).
- **When the cap binds, the screenshots are spread evenly** rather than truncated,
  and the UI says so with both numbers ("One every 5 s would give 204 screenshots,
  over the 200-screenshot limit, so 200 will be described, spread evenly…") instead
  of silently ignoring the chosen interval. Taking the first N frames would cover
  only the opening stretch; spreading keeps coverage from start to finish. The
  scan-based selection no longer *widens the interval* to fit the cap *(2026-09-27,
  review finding S2)*: grid frames are 5 s apart, so a widened 5.1 s snapped to the
  next grid frame 10 s later and a 17-minute video at slider 5 got 122 screenshots
  where the cap allowed 200, under a caption that still said "every 5 s". Widening
  up front only made sense when frames were extracted per run; with the scan they
  exist already. The rough no-scan estimate still widens.
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

### AI title
- **A title is stored beside the file name, never over it; the mode is applied when
  shown** *(2026-09-29)*. The owner asked for a setting, chosen once, that names each
  transcript from its content and either replaces the file name or is added after it
  (Teams names recordings `Meeting in General-<date>-Meeting Recording.mp4`). The
  title goes into its own nullable `title` column; History labels and download names
  are built from file name + title + the current mode (`titles.display_name`). So
  switching the mode renames earlier titled rows too, Off brings back the old names,
  and a title is never paid for twice. Download names without a title keep the old
  `transcript_<name>` form. The entry's caption keeps the original file name visible
  when the label shows the title.
- **The preferences live in the history database** (`preferences` table), not in
  `.env` and not in the session: the page must be able to change them, and they
  belong with the history they rename (backed up with it, no new file to gitignore).
  Read once per session; a stored model no longer offered falls back to the default.
  The same table keeps the file source and the recordings folder (`source`,
  `recordings_folder`; `app._PREFERENCES`).
- **Titles after Local runs too — the owner's call** (2026-09-29, "neko ko lokalno
  pravi transcript slobodno može isto da generiše naslov"). The Local engine keeps the
  audio on the machine, but with titles on, its transcript is sent to OpenAI; the
  README says so. Same key as transcription (`.env` or sidebar); without a key the run
  is saved untitled with a note.
- **A title never fails a run.** It is requested after the transcript is written and
  before the row is saved; any error (no credit, network, an empty answer) becomes a
  warning and the row is saved without a title. Its tokens go into the run's cost
  (`title` part of `usage_json`), including an empty answer, which is paid too.
- **Models and cost:** `gpt-5.4-nano` by default, `gpt-5.4-mini` offered — the same
  price table as screenshots (`CHAT_PRICE_PER_MTOK`, renamed from
  `VISION_PRICE_PER_MTOK`). About 17k tokens per hour of speech (measured on dense
  Serbian, 2026-09-25): ≈ $0.004 per hour on nano, ≈ $0.014 on mini. Up to 200,000
  characters are sent (about four hours); a longer transcript is sent as its beginning
  and end.
- **The answer is cleaned into a safe file name** — first line only, a `Title:` /
  `Naslov:` label and quotes removed, `:` → ` -`, characters Windows forbids replaced,
  at most 80 characters cut at a word, reserved names (`CON`, `COM1`…) extended.
- **The title must be in a script the transcript uses; one retry, then no title**
  *(2026-09-29)*. nano titled the owner's English Teams meeting
  `部署自动化与TIS订阅环境选型讨论` — right content, wrong language; the transcript had
  30,375 letters, all Latin (counted by script, read-only). The prompt already said "in
  the language and script the speakers use". Now the transcript's letters are counted by
  script (`titles.letter_scripts`); the user message ends, after the transcript, with a
  reminder naming the main script (a small model follows what it read last); and a
  title that does not fit goes back once, as the assistant's answer plus a correction.
  A second misfit → no title with a warning ("wrote it in CJK script twice"), since a
  third try would mostly pay for the same mistake. The retry re-sends the transcript,
  so it doubles that title's cost (≈ $0.002 for the owner's 53-minute meeting); every
  paid request is counted, also when the title fails (`TitleError.usage_records`).
  - *What fits:* the title's **main** script must hold at least **10 %** of the
    transcript's letters, and its other letters must occur in the transcript at all.
    The first version only asked for presence; the review showed one line of the
    Chinese subtitle credits Whisper hallucinates into silence would then let a Chinese
    title through. The share lets either script of a Serbian meeting that alternates by
    chunk pass (while it is a tenth of the text), and presence keeps "Azure" in a
    Cyrillic title.
  - *Counting:* NFKC first and modifier letters (Lm) skipped — the first version, which
    took the first word of each letter's Unicode name, called `º`, `ª`, `µ`, `ＡＩ` and
    `ʼ` scripts of their own (MASCULINE, FEMININE, FULLWIDTH…) and would have sent good
    titles back; hiragana and katakana count as CJK, so Japanese is one script. No
    `regex` dependency for the Script property: NFKC + the name covers the owner's
    languages.
  - *Timeout:* a title request waits at most 60 s (`TITLE_TIMEOUT_SECONDS`), not the
    5 minutes a transcription chunk may take.
  - Rejected: detecting the *language* (English vs Serbian share the Latin script) —
    it needs a language model or a dictionary, and the observed failure was the script.
    Not caught: a Latin title in the wrong Latin language.
- **Titles for saved rows come from a History button, and cost is added to the row**
  *(2026-09-29)*. "🏷️ Make a title" / "🏷️ New title" in an open entry, only while the
  mode is on (Off shows no titles), disabled without a key. The request's usage joins
  the row's `title` part of `usage_json` and `cost_usd` (`usage.add_to_cost`) — also
  when no title came back, since it was paid. Rows saved before costs were recorded
  keep no cost: the title's alone would read as the price of the whole transcript. The
  owner can fix the Chinese title of their row this way (their click, ≈ $0.002). How it
  runs, each point a finding of the review:
  - The click only marks the entry; the request is made while the list is drawn, under
    a spinner in the entry's place. Done in the callback, it blocked the page with no
    sign of life.
  - **Streamlit 1.56 identifies an expander by its label too** (`key_as_main_identity`
    is False for `st.expander`), so a new title made a new, closed entry and its message
    was never seen. The entry is drawn with `expanded=True` for `(id, new label)`
    (`hist_reopen`). The AppTest that opened entries through
    `session_state["hist_<id>"]` passed anyway — that value survives the identity
    change; the tests now send the expander's state the way a browser does
    (`_run_as_browser`, AppTest internals).
  - A second click within 2 s of the last title of that entry is ignored
    (`TITLE_REPEAT_GUARD_SECONDS`): the second half of a double click would pay again.
    A token drawn into the button's arguments would not help: Streamlit 1.56 keeps one
    set of callback arguments per widget, from the latest run that drew it
    (`SessionState._set_widget_metadata`, read in the code). Checked live against a
    fake OpenAI server: a JS double click sent one pair of requests. The outcome
    message stays while the entry is open, so the ignored click's rerun does not wipe it.
  - A new title for the row just transcribed also renames the Result's downloads
    (`run_row_id`).

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
- **A visible ⏹ Stop button, the same mechanism as the toolbar's Stop**
  *(2026-09-30)*. The owner did not notice the toolbar's Stop. The button, under
  Start, is the one control left enabled during a run (only in the run that owns the
  job — not while an earlier run's thread is already stopping), with a caption that it
  stops after the part in progress. Its click is an ordinary rerun request, which
  stops the job's script at its next Streamlit call — **session-state reads and writes
  count too**, not only drawing. How the rerun then runs depends on
  `runner.fastReruns`:
  - *On (Streamlit's default):* the rerun starts at once in a **new** thread while the
    old one finishes its request; the page stays locked and says "Stopping — waiting for
    the request in progress to finish…" within about a second, and the watchdog
    retires the job when the old thread ends — the toolbar Stop's path. This was found
    live, not assumed: the first live test seemed to send one part too many, until the
    fake server's log showed that part had started before the click.
  - *Off (and in AppTest, which has no AppSession):* the rerun runs in the job's
    **own** thread, which would start the job again; the callback sets
    `stop_requested`, and `_settle_job` retires the job when the current thread is
    its worker.
  An instant stop was not built: it needs a background worker, and the part in flight
  is paid either way. A video scan still running (it reports no progress) finishes
  before the stop.
- **Between the history row and discarding the checkpoints there is no Streamlit
  call** *(2026-09-30)*: `run_row_id` is stored after the discard. A Stop landing
  between the two (with session state as a stop point) would have kept the parts of a
  saved transcript, and the next Start would have saved it a second time.
- **Transcription checkpoints are deleted only after the history row is written** —
  in the app, not in the pipeline, for the same reason: a run stopped between writing
  the transcript and saving it must be able to finish without paying again.
- **"Clean temporary files" sits in the sidebar, in a "🧹 Working files" section** that
  says how much space `temp/` and `uploads/` take *(2026-10-03, the owner's call)*.
  With the one-day cleanup it is rarely needed; it is maintenance, not a transcription
  step; and under Start/Stop it was a click away from deleting the saved parts of
  unfinished runs. The size counts what the button deletes (`checkpoints.
  working_files_size`): everything in both folders except the history or model folder,
  each file once. The section is drawn after the page, so the size includes what the
  same click prepared (an extracted WAV); the caption is only the size ("**24.6 MB** in
  use.", shortened at the owner's request), the details are in the button's help.
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
- **Model dropdowns say what each option is** *(2026-09-29, the owner's request)*: the
  OpenAI models show release year, strength and price per minute
  (`TRANSCRIPTION_MODEL_NOTES`), and the local list is labelled "Local Whisper
  model" — its options are sizes of one model, which the owner could not tell. The
  gpt-4o-transcribe figure (~$0.004/min) comes from the recorded cost of the owner's
  first 4 runs ($0.0027–0.0033) and the dense-Serbian estimate (~$0.0045); whisper-1's
  is its list price, kept in step with `TRANSCRIPTION_PRICE_PER_MINUTE` by a test. A
  hint must stay under ~360 px (about 55 characters): on a 1860 px window the
  dropdown cut the old large-v3-turbo hint mid-word.
- **Recordings can come from a folder on this computer, read where they lie**
  *(2026-09-29)*. A 372.6 MB video upload failed: `MemoryError` in Tornado's
  `parse_multipart_form_data`, then `Invalid multipart/form-data`, HTTP 500, the file
  red in the uploader. Streamlit receives an upload as one request body in memory, and
  Tornado copies it while parsing (the slice and the split: roughly 3–4× the file for a
  moment); measured soon after, the machine had **1.1 GB of commit free of 47.5 GB**
  (Claude 11.3 GB, Edge 9.6 GB), and the app held 834 MB, partly the previous 437.6 MB
  upload still in the uploader. Streamlit 1.56 has no chunked or on-disk upload, so the fix is
  not to upload: the app runs on the machine that has the file. A source choice
  ("Upload a file" / "From a folder on this computer") above the uploader; the folder
  is typed or pasted once — a pasted recording path (Explorer's *Copy as path*, with
  quotes) means its folder with that file chosen — and kept with the source in the
  `preferences` table; the list shows the folder's video and audio files (not
  subfolders), newest first, and starts empty so opening the page never starts an
  extraction. The file is used in place: no copy in `uploads/`, ffmpeg reads the
  original, on-screen context scans the original, clean-up never touches it (only
  `temp/` and `uploads/`).
- **A folder recording is identified by its path; a change on disk is reported, not
  acted on** (a review finding). The first version keyed it on path + size + mtime,
  like an upload's `file_id`; a recording still being written (OBS, a OneDrive sync)
  then had a new identity on every click, which wiped a finished transcript from the
  page and extracted the video again. Now the size and mtime at the moment of choosing
  are kept (`source_version`), and a later change shows a warning with "↻ Load it
  again". The checkpoint and screenshot-cache digests are taken when the run starts
  and reused at its end, so a file that grows during the run does not leave its saved
  parts behind.
- **The folder field takes full paths only, and not the app's own folders**
  (review findings): a relative path or a bare `C:` was read against the app's working
  directory, whose `temp/` and `uploads/` hold its working copies — and "Clean temporary
  files" would have deleted a recording picked there. `temp/`, `uploads/`, `data/` and
  `models/` (and folders inside them) are refused with a warning. A folder that cannot be
  read says so ("cannot be read: Access is denied") instead of "no video or audio files";
  a chosen file that disappeared says so instead of being forgotten silently (a Start at
  that moment did nothing); a path pasted in another letter case finds its file
  (`os.path.normcase`).
- **📂 Browse… opens the computer's own file window, in a process of its own**
  *(2026-09-30, the owner: recordings live in different folders, and pasting paths is
  clumsy)*. The app runs on the owner's machine, so the server can show Windows' *Open*
  window (tkinter's `askopenfilename`, which is the native dialog); the chosen file's
  folder becomes the recordings folder and the file is chosen. Decisions:
  - **A child process** (`sys.executable -c <script>`), not a thread: Tk must not be used
    from Streamlit's script threads (a Tk object collected in another thread aborts the
    whole process with `Tcl_AsyncDelete`), and each rerun may be a different thread. The
    child prints the path; `PYTHONUTF8=1` keeps č, ć, š intact in the pipe.
  - **Topmost, not foreground:** measured on the owner's machine, the window appears in
    0.7 s, 960×540, on top of the browser — but Windows does not let a background process
    take the keyboard focus, so the owner clicks into it. The spinner says where to look.
  - One window at a time for the whole app (a process-wide lock: a second click or tab
    gets "already open — it may be behind the browser"), opened in the last folder, and
    given up after 10 minutes (`FILE_PICKER_TIMEOUT_SECONDS`), since the page waits.
  - A native picker, not a file tree drawn in the page (a component) or a list of
    several fixed folders: it goes anywhere, looks like every other program, and needs
    no dependency (tkinter ships with the uv Python). Only where the folder source is
    offered (loopback), since the window opens on the machine that runs the app.
  - Labelled "📂 Browse…": "Choose file…" wrapped onto two lines at a 1440 px window.
- **The folder list's labels: length first, then name and creation time — nothing that
  changes while a file grows.** Streamlit 1.56 keeps a keyed selectbox's value as the
  *formatted label*, so labels hold no size or modification time (they would change on
  every click while OBS records the next meeting into the same folder). Labels are
  `length · name · creation time` (`st_birthtime`, Windows/macOS; Linux falls back to
  mtime), the list is ordered by creation time for the same reason, and the size and
  last save are a caption under the list.
  - **The length** *(2026-10-03, the owner's request)* is read by ffprobe
    (`recordings.duration`): about 0.13 s a file, mostly starting the program, so eight
    are read at once and each file once per version (size, mtime; `lru_cache`). It
    leads because recorder names (`2026-09-30 15-02-17 - daily okej prep for
    deployment.mkv`) are cut off at the list's width (about 55 characters), date and
    all. Format `M:SS` / `H:MM:SS`, as elsewhere on the page.
  - Measured on synthetic files: an open-ended MKV being recorded gives `N/A` (no
    length until the recorder finishes it), an MP4 without its index fails, a
    fragmented MP4 gives the length so far. So a recording in progress shows no length,
    and its entry changes once, when it is finished.
  - **A changed entry of the chosen recording is sent again.** Checked in a real browser
    (puppeteer, a probe app): after the chosen entry's text changes, the browser keeps
    the old text and sends it back. The first click still works (the server reads it
    with the previous run's options before the script runs); the second matches no
    entry, and the choice is lost. AppTest does not show this — it re-sends the value in
    its new wording — so the test sends the held text itself. The page remembers every
    entry it drew (`folder_shown`) and, when the chosen one's text differs from what was
    drawn, sets the value again, which sends the new text. Every entry, not only the
    chosen one (pre-commit review): a recording picked from a list drawn before its
    length appeared was lost on the next click — Start.
  - Only the newest 200 recordings get a length (`RECORDING_PROBE_LIMIT`): a phone's
    call folder can hold thousands, and 1,500 would take about 25 s.
- **The folder source is offered only while the app listens on loopback, and
  `ALLOW_LOCAL_FILES=false` turns it off anyway.** The page can list and read any
  folder of its machine; that is the point locally, and a leak to anyone else who can
  open it — including someone on the LAN after `--server.address=0.0.0.0`, the way
  `.streamlit/config.toml` suggests for reaching it from another device (they would also
  transcribe with the owner's key; a UNC path they type would make the machine connect
  to their SMB host). So `server.address` must be `localhost`, `127.0.0.1` or `::1`;
  Docker (which binds `0.0.0.0` inside the container) gets only the uploader. See
  Hosting in §5.
- **CSS hack for the uploader label** — Streamlit prints the full list of 27 extensions; it
  is hidden and replaced with short text. The element to target is a **`<span>`** (not
  `<small>` — established by inspecting the DOM).

### Data
- **SQLite lives in `data/`, not `temp/`** — history **survives** "Clean temporary files"
  (which wipes `temp/` and `uploads/`). That is the whole point of having it.
- **Working copies of recordings are deleted after a day, each time the page is
  opened; transcripts are not — yet** *(2026-09-30, the owner's choice)*. Nothing but
  the "Clean temporary files" button removed uploads and extracted audio: `uploads/`
  held 11 MKVs back to 2026-09-14 (2.67 GB) and `temp/` 794 MB, mostly WAVs.
  `checkpoints.prune_working_copies` now deletes files directly in `uploads/`, and
  `.wav` / `.mp3` files directly in `temp/`, whose modification time is over 24 h
  (`WORKING_COPY_MAX_AGE_HOURS`), once per page opened — next to the existing 24 h
  scratch rule and the 14-day rule for saved parts, whose folders it does not touch.
  - *Rejected by the owner:* also deleting the previous file's copies whenever a new
    file is chosen. The age rule alone is simpler and bounds the pile to one day.
  - *Transcripts (`.txt`, `.srt` in `temp/`) are kept for now* — the owner's call
    ("to ćemo kasnije"); they are small (18 files, 0.3 MB) and the Result panel reads
    them. See §5.
  - *Safe by construction:* nothing while a run is active in this process; nothing in
    a folder that holds the history or the models (with `DATA_DIR` = `UPLOAD_DIR`
    nothing there is deleted at all); a file another program holds open is skipped and
    logged. The guard is per process — a second app instance does not see this one's
    run — but a run reads its WAV at the start and digests it then, so a copy deleted
    under it costs nothing.
  - *A page that outlived its copy prepares it again* (`prepare_audio` checks that the
    prepared file still exists): an upload is re-saved from the browser's copy in the
    session, a folder recording re-read, a video's audio re-extracted. Re-extraction is
    byte-identical (checked: two extractions of the same video a moment apart gave the
    same checkpoint digest), so the saved parts of an unfinished run are still used.
  - When introduced it frees about 1.9 GB on the owner's machine at the first page
    opened (12 uploads, 1.5 GB; 15 WAV/MP3, 0.4 GB — sizes read, nothing opened).
- **PRAGMA `journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON`** — better defaults
  for Streamlit's rerun model and multiple open sessions.
- **Migration without a migration tool:** `init_db()` checks `PRAGMA table_info` and adds
  any missing column — currently `provider`, `elapsed_seconds`, `cost_usd`,
  `usage_json` and `title` — and creates the `preferences` table, so existing
  databases are not lost. New columns must be nullable, since existing rows have
  no value for them.
- **Audio is NOT stored as a BLOB** — only the transcript and SRT text; `audio_path` is a
  best-effort reference that may disappear after a cleanup.
- **Every run records what it actually used and cost** *(2026-09-25)* — per history row
  `cost_usd` and `usage_json` (requests, audio seconds, input → output tokens for the
  transcription and for the screenshots). Tokens are the ones the API reports in each
  answer's `usage`; the cost is those tokens at the prices in `config.py` (checked on
  OpenAI's pricing pages), or audio minutes for models billed per minute (whisper-1,
  gpt-transcribe). When an answer carries no usage, the per-minute estimate is used and
  the figure is shown with `≈`. Each record travels with its checkpointed result, so a
  resumed run counts what its first attempt paid, and a capped answer that was split
  counts too. The owner's question "how much does a clip cost?" had no answer before:
  the app knew only its own pre-run estimate. The OpenAI dashboard (Usage) remains the
  authority for billing; this is per transcript.
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
- **Or pick a recording from a folder on this computer** — read in place, no upload, no
  copy; for large files that an upload cannot hold in memory *(2026-09-29)*; found with
  **📂 Browse…** in Windows' own file window, in any folder *(2026-09-30)*; each entry
  shows the recording's length *(2026-10-03)*
- **Audio player** before and after transcription
- **Two engines**: OpenAI API (`gpt-4o-transcribe` / `whisper-1`) and **Local offline** (faster-whisper, selectable model size)
- **Timestamps + `.srt`** export (whisper-1 and every local model)
- **Readable transcript**: inline `(M:SS)` + paragraphs *(2026-07-20)*
- **On-screen context** for video: key frames described by a vision model and placed
  in the transcript by time — exactly with whisper-1 or the local engine, at
  approximate `(~M:SS)` paragraphs with the default gpt-4o-transcribe *(2026-09-25)* —
  with a selectable screenshot interval *(2026-07-20)*; the distinct screens found by
  a one-pass scan are described *(2026-09-27)*, and the most they can number and cost
  is shown as soon as the box is ticked, with no wait *(2026-10-03)*
- **Reliable long runs** *(2026-09-25)*: 5-minute chunks cut in pauses (no silent
  truncation at the output cap), checkpoints so a failed or stopped run resumes without
  paying again, partial transcripts clearly marked, one-sentence errors for a bad key
  or an empty balance, controls locked during a run with a watchdog after Stop, and a
  visible ⏹ Stop button that stops after the part in progress *(2026-09-30)*
- **Original script kept by default**; Serbian → Latin is opt-in (`SERBIAN_LATIN=true`)
  *(2026-09-25)*; one-off backfill script for saved records
- **Playable previews** for AMR/WMA/AIFF, video and large files; fast History at 100+
  records *(2026-09-25)*
- **Run time reported** next to the transcript and in history *(2026-07-28)*
- **Actual tokens and cost of every run** next to the transcript and in history *(2026-09-25)*
- **AI title** *(2026-09-29)*: an optional sidebar setting, kept across sessions, that
  names each transcript from its content (nano or mini) and uses the title instead of,
  or after, the file name in History and in downloaded file names; the title keeps to
  the transcript's script (one retry), and saved rows can be (re)titled from History
- **Persistent history** (SQLite): browse, re-download TXT/SRT, delete
- **Working copies clear themselves** — uploads and extracted audio over a day old are
  deleted when the page is opened *(2026-09-30)*; the sidebar shows the space working
  files take, with the button to delete them now *(2026-10-03)*
- **Hybrid API key** (`.env` or sidebar); offline works with no key
- **Wide layout + tabs** (Transcribe / History), two-column arrangement

**Quality / infrastructure**
- Modular refactor (config/audio/transcribe/db/exceptions/logger), type hints + docstrings
- **ruff: 0 errors** (down from 74), `ruff format` clean; modern ruff config (`[tool.ruff.lint]`, `target-version=py312`, plus D/RUF/PTH/T20/S)
- **pytest: 334 tests** (DB CRUD + PRAGMAs + schema migration + preferences, AI titles and their script check, recordings from a folder, SRT/formatting helpers, frame selection and dedup, count/cost estimates, `.env` overrides, the OpenAI pipeline and vision step against fake clients and a fake network, checkpoints, transliteration, the backfill, headless UI flows with AppTest) — no network, no ffmpeg; the regression tests were each checked to fail with their bug put back
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
      byte-identical, integrity ok). **Later reversed at the owner's request:**
      `SERBIAN_LATIN` is off by default and the 10 rows were restored to their original
      script (§3, journal 2026-09-25).
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
- [x] **Burst thinning keeps the settled frame** (R09) — found again while checking the
      screenshot count; see §3.
- [x] Partly: per-run scratch folders for chunks and frames (R07); CI `--locked` and a
      3.12 + 3.13 matrix (R33); the dead `default_model` setting removed and
      `SEGMENT_DURATION_MINUTES` wired (R46); fake-client tests for the paid pipeline
      (R52); automatic pruning of checkpoints (R48).

### Done 2026-09-27 — scan review findings (S1–S4) and the second review

The work after commit 46f3212 — real tokens and cost per record (`usage.py`, DB
columns), the original script kept (`SERBIAN_LATIN` off), and the video scan with the
exact screenshot count — was held back until the scan was reviewed. The first review
(workflow `wf_6030706f-ea1`) stopped mid-run with 4 scan findings; the second covered
scan, UI, cost and docs. Committed and pushed with the owner's approval ("komituj i
pušuj"). Gate: 210 tests pass, ruff clean.

- [x] **S1 (high, regression) — a described screenshot could show an earlier
      screen.** Fixed 2026-09-27: shared frames are marked and re-extracted at their own
      time before being described (§3); byte-identical to the scan's frame.
- [x] **S2 (medium, regression) — the cap-widened interval snapped to the 5 s grid.**
      Fixed 2026-09-27: no widening in the scan selection, the cap spreads evenly and
      the caption gives both numbers (§3).
- [x] **S3 (medium, pre-existing) — the 8×8 dhash calls different screens repeats.**
      Studied 2026-09-27 and **not changed**: the candidate rules add mostly Teams UI
      noise on the owner's real recordings (§3). The picture-sharing part is solved by
      S1. Follow-up in P3.
- [x] **S4 (low) — `extract_keyframes` defaulted to a shared `temp/frames`.** Fixed
      2026-09-27: the function (no callers) is deleted, and a scan stores its threshold
      and grid, so one made with other settings is redone.
- [x] Regression tests for S1/S2/S4, each checked by putting its bug back (9 of 9 caught).
- [x] Second adversarial review (4 lenses: scan, UI, cost, docs; a skeptic per finding):
      26 confirmed, 1 refuted — all fixed (§3 entries of 2026-09-27; journal), each code
      fix with a regression test that fails with its bug put back (17 of 17).
- [x] Committed and pushed (code, then docs), CI on GitHub.

### Waiting for the owner
**Next step: the owner opens the app** — that runs the first real cleanup — **and tries
the live checks below**.
- [x] **Commit and push** — done 2026-10-03 ("komituj i pušuj"): the title's script
      check and the History title button, the folder source with 📂 Browse…, the ⏹ Stop
      button and the one-day cleanup, after a last review of Stop, Browse and the cleanup
      (journal 2026-10-03). One feat commit and one docs commit: `app.py` carries all
      four features, so commits per feature would have left states that do not pass.
- [x] **Commit and push the three changes of 2026-10-03 (parts 2 and 3)** — done the
      same day ("komituj i pušuj"), after a reviewer subagent's pass and its fixes
      (journal), again one feat and one docs commit, as `app.py` carries all three.
- [x] **Move "Clean temporary files" to the sidebar** — done 2026-10-03 (owner: "uradi"),
      in a "🧹 Working files" section with the space the files take (§3); its caption
      cut to "**24.6 MB** in use." at the owner's request, the rest in the button's help.
- [x] **The recording's length in the folder list** — done 2026-10-03 (owner's request),
      checked live on an isolated instance (§3).
- [x] **On-screen context: show its settings at once** — done 2026-10-03 ("uradi
      ovako"): no scan before Start, the caption gives a ceiling (§3).
      The discussion that led there: ticking "Describe what's on screen" shows a spinner for about a minute
      and a half per hour of video before the slider appears — the scan decodes the
      whole video once so the caption can give the exact screenshot count and cost (the
      run then reuses it). The owner does not want the wait, and asked why the rough
      estimate (length ÷ interval) is so far off: it counts samples, while the run
      describes *distinct screens* (repeats dropped, every change added) — 231 vs 10 on
      a static 115-minute meeting, 4 vs 28 on a busy 16-minute screen share (§3).
      Proposed: draw the settings at once; estimate from **keyframes only**
      (`-skip_frame nokey`: the full pictures the file stores every few seconds, hashed
      like the scan's frames) — measured on a synthetic 5-minute 1080p MKV: 0.9 s
      against 8.5 s for the full scan (about 10 s per hour of video instead of 1.5
      min), one picture every 8.3 s (x264's default keyframe interval); the exact scan
      runs at Start, as part of the run. To check first, on the owner's recordings
      (numbers only): their keyframe interval, and the quick count against the exact
      one. **Simpler, now recommended** (after the owner noted a maximum is always
      known): no scan before Start; show at once "at most N screenshots (at most $X) —
      usually fewer, a repeated picture is not described again". N = length ÷ interval
      for slides (ffmpeg's change detection rarely fires on them: a full-screen slide
      change scored 0.077, under the 0.1 threshold), never more than one per 2 s
      (`FRAME_MIN_INTERVAL_SECONDS`, fast changes) nor the cap (`FRAME_MAX_COUNT`, 300
      in the owner's `.env`). Cost of the cap: nano $0.18, mini $0.67. The owner chose
      this one.
- [ ] **First real run of the one-day cleanup.** On 2026-10-03 it had not run yet: the
      owner's app last worked on 2026-09-30 16:32, just before the cleanup was added, and
      was not running. `uploads/` held 16 files (2.67 GB, back to 2026-09-14) and `temp/`
      45 (880 MB: 13 WAV, 14 MP3, 18 TXT). The first page opened should remove everything
      but the TXT files older than a day — check the sizes afterwards (totals only).
- [ ] **Live check of 📂 Browse… with a real choice** — the window was opened and
      cancelled from scripts only; choosing a file in it is the owner's to try.
- [x] **The key and the credit were on different accounts** — resolved 2026-09-25: the
      old key belonged to another OpenAI login (balance $0); the owner put a key from the
      funded marko2212 account into `.env`.
- [x] **Live check of the OpenAI path** (with the owner's OK, synthetic 30 s video):
      gpt-4o-transcribe + 3 screenshots on nano, $0.0023 in total; `usage` is returned
      (300 input / 79 output tokens for 30 s — so the cap guard works on real answers),
      `(~M:SS)` paragraphs with the notes in place, the cost saved with the row.
      Still untested live: a resumed run after a real failure.
- [x] **Live check of the AI title** — done by the owner 2026-09-29 ("radi okej"): a
      7:30 English meeting (MKV) on gpt-4o-transcribe, title on nano in Replace mode:
      "Validation and Release Blockers Coordination", 1,146 → 9 tokens, so nano's answer
      fits the 200-token limit with room to spare. Whole run $0.02, 18 s. Still unseen
      live: a Serbian title, and a Local run with a title.
- [ ] **Live check of the script guard and the History title** (the owner's credit,
      ≈ $0.002): open the 2026-09-29 "Venki join us" entry, whose title nano wrote in
      Chinese, and press 🏷️ New title — it should come back in English, and the entry's
      cost grow by the title. Checked only without OpenAI: on an isolated copy with
      OpenAI unreachable (the warning, the unchanged row) and against a local fake chat
      server (Chinese answer → correction → English title, entry kept open), and in
      AppTest.
- [ ] **Live check of a large recording from a folder** — e.g. the 372.6 MB "Cisco"
      MKV whose upload failed. The isolated check used a 4 MB synthetic MKV and a
      synthetic M4A (extraction, player, a Local run, clean-up leaving the originals).

### P1 — transcript correctness
- [ ] **Blind test of providers on real audio before 2027-02-26** (M). 10–15 min of real
      Serbian and Macedonian meeting audio plus an AMR call, hand-corrected reference;
      candidates: gpt-transcribe, ElevenLabs Scribe v2 (Data-use opt-out on), Gemini 3.5
      Transcribe (paid tier), Deepgram Nova-3 (language sr/mk, `mip_opt_out`), Soniox;
      compare WER and which script each returns. The winner gets a thin adapter at the
      two call sites (`transcribe._request`, `vision._describe_frame`). Costs a few
      dollars across vendors — the owner decides.
- [ ] **Local default model** (S). `base` is weak for South Slavic; `large-v3-turbo`
      (1.6 GB) or `medium` would be the honest default — measure CPU time on a real
      meeting first.
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
- [x] **A visible Stop button next to the progress** — done 2026-09-30 (the owner's
      "uradi sada dugme stop"): "⏹ Stop" under Start while a run works, the same
      mechanism as the toolbar's Stop (§3 UI).
- [ ] **Transcript box** (R20, S). `st.code(text, language=None, wrap_lines=True, height=…)`:
      read-only, scrolls, has a copy button (the editable text_area silently drops edits).
- [ ] **Delete needs a confirmation** (R11, S). `st.popover` with "Delete permanently",
      placed to the right of the download buttons.
- [ ] **Search in history** (R23, S–M). Filename and text, with diacritic and script folding.
- [ ] **Parallel API calls** (R16, M). Thread pools for chunks (3–4) and frames (6–8),
      progress reported on the script thread; start frame extraction while chunks run.
      Twice as many chunks since R01 make this more worthwhile.

### P3 — on-screen context quality
- [ ] **The caption prompt describes the call window** (R29, S). Describe only shared
      content; NONE for a participant grid (12–24% of notes are only about the call UI).
      Bump `vision._CACHE_VERSION` with it.
- [ ] Smaller: a pixel-diff second opinion for panel-sized changes dHash misses (R28);
      repeated screens re-captioned (R40); downscale frames to 512 px, ~83% less upload
      (R42); an honest time estimate — recorded usage is done, see §3 (R39); faster
      scene scan (R41).
- [ ] **Local OCR as a cheaper visual layer** — reads slide text for zero tokens and fixes
      the "slides differing only in text" dedup limit; adds a second system binary.
- [ ] **A duplicate-screen rule that sees same-template slides but not the call UI**
      (L; S3, §3). Thumbnail rules with motion masks were built and measured
      2026-09-27; on real Teams recordings they counted the active-speaker border and
      tile rearrangements as new screens. Needs a way to ignore the participant area
      (avatars do not move) before it can replace the dHash. A cheap side finding:
      comparing with the last three kept screens instead of one stops a screen that
      toggles back and forth from being described each time (a real 29-minute call at
      slider 5: 23 → 4 frames with the dHash alone) — not adopted, since it also drops
      a genuine return to an earlier slide.

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
      wording and README fixes (R47, R55); Host allow-list (R30). (Auto-clean of old
      uploads, R48 rest, was done 2026-09-30 — §3 Data.)
- [ ] **Transcript files in `temp/` (`transcript_*.txt` / `.srt`)** — left out of the
      one-day cleanup by the owner (2026-09-30, "to ćemo kasnije"). They are small and the
      Result panel reads them; decide whether they follow the same rule (the history
      keeps the text anyway) or a longer one.
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
- [ ] Review leftovers of 2026-10-03 (low, from reading): **a Stop that lands in the
      milliseconds after the history row is saved** (between the insert and the next
      session-state access) retires the job with "stopped before it finished … Start
      transcribes it from the beginning" although the row is saved — pressing Start would
      pay again and save a second row. Suggested fix: record the row on the job dict right
      after the insert (a dict write is no stop point) and let `_settle_job` say the run
      finished. Smaller: a Stop during the title request loses that paid title (it is
      bought again on the next Start). Two tabs titling one row at once can lose one cost
      update (2026-09-29 review).

### P5 — features and later
- [ ] **Meeting summary and action items** (R38, M). An explicit button, gpt-5.4-mini, stored
      in a new nullable `summary` column; hidden for very short transcripts. `titles.py`
      is the pattern (one chat request, usage record, failure only warns).
- [x] **AI titles for older rows** — done 2026-09-29: "🏷️ Make a title" in a History
      entry, and "🏷️ New title" for a row that has one (to replace a bad title), while
      the mode is on; its cost is added to the row (§3 AI title).
- [ ] **Hosting** *(researched)* — Hugging Face Spaces or an Oracle Always Free VM; users
      must bring their own OpenAI key. Needs a **concurrency limit + queue** if the offline
      engine is ever public. The folder source switches itself off when the server does
      not listen on loopback only; `ALLOW_LOCAL_FILES=false` makes that explicit.

---

## 6. Journal

### 2026-10-03 (part 3) — No wait for the screenshot slider

The owner disliked the spinner after ticking "Describe what's on screen" (the scan,
about a minute and a half per hour of video, before the slider appeared) and asked why
the rough estimate was so far off. Explained: the run describes distinct screens, not
one per interval (231 estimated vs 10 described on a static 115-minute meeting). Two
proposals followed — a quick count from keyframes only (measured on a synthetic
5-minute 1080p MKV: 0.9 s against 8.5 s for the full scan), and, after the owner
noted that a maximum is always known, simply showing that maximum. The owner chose the
second ("uradi ovako"), and asked what the slider really sets (how often the screen
is looked at; repeats are not sent).

- Ticking the box no longer scans; the run does (its spinner now says how long that
  takes). The caption: at most one per look, plus what faster changes can add up to
  the cap, "usually far fewer", the exact number during the run. A binding cap is
  stated with the number one look per interval would give.
- The tick-time scan's failure memory (`scan_failures`, "untick and tick to try
  again") went with it: a failed scan is now a warning of the run.
- Live, two prices in one caption lost their `$` (Markdown read the text between as a
  formula); now escaped, with a test.

Verified: 327 tests (the scan-on-tick tests became "no scan on tick" and "the run never
describes more than the ceiling"; a stored scan still gives the exact count during a
job); 6 of 6 changes undone in a copy of the repo were caught. Isolated instance with a
synthetic 1:01:40 video in headless Chrome: the slider appeared 0.4 s after the tick
(it used to take the scan), no scan folder was created, and the caption read "At most
**124** screenshots (up to $0.07) … up to 300 (up to $0.18) …".

Then, at the owner's request, the sidebar's caption was cut to the size alone
("**24.6 MB** in use."); what the button deletes and the one-day rule moved into its
help.

Before the commit a reviewer subagent read the whole diff (isolated, in a copy). It
found one real bug and four smaller ones, all fixed, each with a test that fails when
its fix is undone (5 of 5, in a copy of the repo):
- A recording picked from a list drawn before its length appeared lost the choice on
  the next click (Start): only the chosen entry's text was remembered. Now every
  entry drawn is.
- "Up to $X" assumed Full HD frames; frames are described at full size, so a 4K screen
  costs about 3.4 times that. The caption now uses the video's own frame size.
- A folder of over 1,024 recordings defeated the length cache and re-read every file
  on each click (about 25 s for 1,500); only the newest 200 get a length now.
- A prepared video that states no length was told its count "appears once it has
  been prepared"; it now says the file does not say how long it is.
- The sidebar's size was one click behind (drawn before the page prepared a WAV); it
  is drawn after the page now.
The docs said a failed scan was remembered until the box is unticked (that went with
the scan on tick) and quoted the owner's cap of 300 as the README example; both
corrected. 334 tests. Committed and pushed: one feat and one docs commit.

### 2026-10-03 (part 2) — Recording lengths in the folder list; clean-up in the sidebar

The owner asked for each recording's length in the folder list, and for "Clean temporary
files" to move to the sidebar (the proposal of 2026-09-30).

- **Length first in each entry** (`47:12 · name · creation time`), read by ffprobe,
  eight files at once, each once per version. Measured: 0.13 s a file; an MKV still
  being recorded gives `N/A`, so its length appears when the recorder finishes it.
- That makes the chosen entry change once, and §3 said a changed entry loses the choice.
  An AppTest probe kept it, so it was checked in a real browser: the browser keeps the
  old text, the first click still works, the second loses the choice. Fixed by sending
  the value again when the chosen entry's text changes; the test sends the held text
  itself, since AppTest re-sends values in their new wording.
- **🧹 Working files** at the bottom of the sidebar: the space `temp/` and `uploads/`
  take (what the button would delete) and the button, renamed "Clean temporary files"
  without the broom, which the section title now carries. Its message shows there too.

Verified: 326 tests (9 new); 8 of 8 changes undone in a copy of the repo were caught
(choice not sent again, length at the end or missing, button left out, length read on
every click, `N/A` taken as a length, history counted, a folder named twice counted
twice). Isolated instance (port 8597, scratch folders, synthetic recordings) in headless
Chrome: lengths `0:47`, `1:01:40`, `2:00` and none for a recording in progress; that
recording chosen, finished with `q`, then three reruns — entry `0:09 · …`, still chosen,
with the existing "changed on disk" warning; the sidebar said 25.0 MB, then "Empty"
after the button, and the pick was cleared. No page errors.

Also asked: why ticking "Describe what's on screen" spins before the slider appears.
Answered (the one-pass scan, about 1.5 min per hour of video, reused by the run) with
a proposal in §5; nothing changed yet.

### 2026-10-03 — Last review, two fixes, commit

Status first: 314 tests passed and ruff was clean on the uncommitted work; the owner's
app was not running, and its last run (2026-09-30 16:32) predates the cleanup, so
`uploads/` still held 2.67 GB and `temp/` 880 MB.

On "komituj i pušuj", the parts no review had seen yet — Stop, Browse… and the cleanup —
went to one reviewer subagent first (isolated, no real data, no file window). No serious
bug; three low-probability ones, two fixed before the commit:
- A Browse click handled in the same moment as Start opened the file window in the
  job's own run, blocking the job; and one overtaken by a switch to Upload came back
  later. The pending click is now dropped on every run and acted on only while nothing
  runs.
- An upload is saved a little before its WAV, so the one-day cleanup can delete the
  video and keep the WAV; the page did not prepare it again (on-screen context would
  then fail with a warning). `prepare_audio` now also checks the video's copy.
- Left, recorded in §5 P4: a Stop in the milliseconds right after the history row is
  saved gives a misleading "start from the beginning" notice.

317 tests (3 new); the three new tests each failed with their fix undone (in a copy of
the repo). Committed as one feat and one docs commit — see §5.

Done on 2026-09-29/30, in order (details in the entries below and in §3): the AI
title keeps to the transcript's script and can be (re)made from History; recordings
can be picked from a folder on this computer (no upload, no memory copy), with an
adversarial review whose nine findings were fixed; a visible ⏹ Stop button; 📂 Browse…
in Windows' own file window; working copies older than a day deleted when the page is
opened (transcripts kept for now).

### 2026-09-30 (part 3) — Working copies clear themselves after a day

The owner found `uploads/` full of recordings back to 2026-09-14 and asked for automatic
deletion. Two options were proposed — delete the previous file's copies when a new one
is chosen, and delete copies over a day old when the page is opened; the owner chose
the second alone, without transcripts ("uradi tako, samo nemoj i za .txt, to ćemo
kasnije, zapiši to kao odluku"). My first wording said the existing cleanup "already
works this way", which the owner rightly questioned: it only ever covered scratch
folders (24 h) and saved parts (14 days), never `uploads/` or the WAVs. Decision and
safety rules in §3 Data; the transcripts are an open item in §5.

Verified: 314 tests (6 new: what is deleted and kept, a run in progress, the history
sharing the uploads folder, a file held open, once per opened page, a page that
outlived its copy preparing it again and transcribing); 9 of 9 bugs put back were
caught, in a copy of the repo; two extractions of the same video gave the same
checkpoint digest (so resuming after a cleanup pays nothing again). Sizes on the
owner's machine were read only as totals: the first page opened will free about 1.9 GB.
The owner's app (two instances, started 15:22 and 16:24) runs from this checkout and
picks the change up on its next rerun.

### 2026-09-30 (part 2) — 📂 Browse… for recordings in any folder

The owner keeps recordings in different folders and did not want to paste paths
("kreni, slažem se sa predlogom"). The button opens Windows' own *Open* window from a
child process; decisions in §3 UI.

Verified:
- 308 tests: the page (a chosen file is picked and its folder kept, the next window
  opens there, cancel changes nothing, an error or a busy window says why, a non-media
  file is refused) and the helper against a fake `subprocess.run` (arguments, UTF-8,
  a failed or timed-out window, one window at a time).
- 12 of 12 bugs put back were caught, in a copy of the repo (`scratchpad/mutcopy`).
- The real window, without sending a keystroke: opened from a script, found by its
  title — shown after 0.7 s, 960×540, topmost, not foreground — and closed with
  `WM_CLOSE` (what Cancel does), which returned "nothing chosen". End to end: headless
  Chrome clicked Browse… on an isolated instance, the page showed "Choose the recording
  in the window that opened…", the server's window appeared, was closed the same way,
  and the page came back without a warning. Choosing a real file is left to the owner.
- Pitfall: stopping the isolated instance with `CommandLine -match '8597'` also killed
  one of the owner's Edge processes, whose arguments happened to contain those digits
  (one tab or extension will have crashed; a reload brings it back). Test servers are
  now stopped by process name (python/uv/streamlit) and the exact `--server.port 8597`.

### 2026-09-30 — A visible Stop button

The owner asked for it after the proposal of 2026-09-29 ("uradi sada dugme stop").
"⏹ Stop" under Start, enabled only while this run owns the job, with a caption; the
decision and both rerun paths are in §3 UI.

Before writing it, Streamlit 1.56 was read for how a click during a run is handled:
`ScriptControlException` is a `BaseException` (so the app's `except Exception` around
the job does not swallow it), every forward message and **every session-state access**
is a stop point, and a rerun without fast reruns loops in the same thread. That last
point would have restarted the job, hence `stop_requested` in `_settle_job`. The live
check then showed that the default `runner.fastReruns` takes the other path (a new
thread, the existing watchdog).

Verified:
- 299 tests. The main one presses Stop from inside a fake pipeline the way a browser's
  click arrives — a rerun request with the page's widget states and the Stop trigger
  (Streamlit internals, since AppTest cannot click during a run) — and checks that the
  job ran once. Others cover the button being the only enabled control, a click after
  the run changing nothing, the fast-reruns path, a Stop during the title (parts kept,
  the next Start saves one row), and a Stop between the row and the discard.
- The mutation check ran in a **copy** of the repo (`scratchpad/mutcopy`, the main
  venv's Python), not in the owner's checkout (lesson of 2026-09-29): 9 of 9 bugs
  caught. The first run missed one — the simulated click re-sent the Start button's
  trigger from the same run, which a browser never does, so the job was re-requested
  and the restart bug hidden; the helper now drops earlier triggers.
- Live, on an isolated instance with a fake OpenAI server that takes 4 s per part and a
  16-minute synthetic call (4 parts). Stop clicked during part 3 → part 3 finished
  and was saved, nothing more was sent, "3 part(s) are saved"; Start then sent only
  part 4 and the Result counted all four requests. Stop clicked during part 1 of a new
  run → "Stopping — waiting for the request in progress to finish…" within 1.5 s →
  "The last run was stopped before it finished. 1 part(s) are saved", again nothing
  sent after the part in flight. No errors in the log. The real history is unchanged (115 old rows by hash,
  plus the owner's own row of 2026-09-29 17:06).

### 2026-09-29 (part 3) — Titles keep to the transcript's script; recordings from a folder

The owner asked three things after their first long run (53:12, $0.17): how the
`(~M:SS)` times come about with gpt-4o-transcribe (answered: chunk start + the
paragraph's share of the chunk's characters, §2), why the AI title was Chinese, and
what the red upload error was. Both causes were measured before anything was changed:
the transcript held 30,375 letters, all Latin (a count by script, read-only), so nano
had simply switched language; the upload died in Tornado's multipart parser with
`MemoryError` while the machine had 1.1 GB of commit free (§3 UI). The owner said
"kreni" for both fixes; a visible Stop button was proposed too and waits (§5 P2).

- **Title:** the script is counted, named after the transcript, and a foreign-script
  title goes back once (§3 AI title). `make_title` now returns a list of usage records;
  `TitleError.usage_records` replaces `usage_record`. History got 🏷️ Make a title /
  New title, whose cost joins the row (`usage.add_to_cost`, `db.set_title`).
- **Folder source:** `recordings.py`, a source radio and a folder field kept in
  `preferences`, a list keyed by path with labels that do not change as a file grows,
  `prepare_audio(name, is_audio, obtain)` so both sources share one path, clean-up
  clears the pick, `ALLOW_LOCAL_FILES` to turn it off (§3 UI).

**Review.** An adversarial reviewer (a subagent, isolated folders, no network) read
the diff before any commit and confirmed nine findings by probes; all but the two-tab
race on one row's cost were fixed the same day (§3 AI title, §3 UI):
1. A new title closed the History entry and its message was never seen — Streamlit
   1.56 identifies an expander by its label too. The AppTest passed only because it
   opened the entry through `session_state`; the tests now send the expander state the
   way a browser does.
2. A folder recording still being written got a new identity on every click, wiping
   the finished transcript — now identified by its path, a change is reported with
   "↻ Load it again", and the checkpoint digests are taken when the run starts.
3. One hallucinated Chinese line in a transcript let a Chinese title through — the
   title's main script now needs a tenth of the transcript's letters.
4. `º`, `ª`, `µ`, `ＡＩ`, `ʼ` counted as scripts of their own — NFKC, and no Lm.
5. The title request ran in a callback: no progress, a page frozen for up to minutes
   (the 5-minute timeout), a double click paid twice — now a spinner in the list, a
   60 s timeout, a 2 s guard.
6. A chosen file that vanished was forgotten silently (a Start then did nothing).
7. A relative path or a bare `C:` listed the app's own folder; a path in another letter
   case lost the pick; an unreadable folder read as empty.
8. The folder source stayed on with `--server.address=0.0.0.0` — now loopback only.
9. Minor: the Result kept the old title after a new one for the same row (fixed); two
   tabs titling one row at once can lose one cost update (left as it is).

Verified: 293 tests (50 new today); each of 51 bugs put back made a test fail — the
first build's 30 (no retry, reminder missing, foreign script accepted, first answer's
cost lost, cost not added or invented for old rows, button while off or without a key,
newest file chosen at once, a folder file copied to `uploads/`, pick kept after
clean-up, pasted quotes kept, size in the label, mtime ordering, …) and the review
fixes' 21 (entry closed after a title, double click paid twice, changed recording wipes
the result, stray line accepted, NFKC dropped, relative path or app folder accepted,
digest taken after the run, network address still lists folders, …). Isolated
instances (port 8597, scratch `DATA_DIR`/`TEMP_DIR`/`UPLOAD_DIR`) in the browser:
- OpenAI at a dead address: a quoted file path chose the file and its folder; the list
  held the MKV and the M4A but not a `.txt`, newest first; the MKV's audio was
  extracted with `uploads/` staying empty; a Local run finished (an empty transcript →
  "No AI title: the transcript is empty", no request); 🏷️ New title on a seeded row
  with the Chinese title warned "Cannot reach OpenAI" and left title and cost as they
  were; clean-up emptied `temp/` and left the recordings.
- OpenAI replaced by a local fake chat server (first answer Chinese, the corrected one
  English, request headers never read): "Writing a title…" showed in the entry's place,
  the entry stayed open under "… - Deployment automation and TIS subscriptions" with its
  message, the row's title part grew by 2 requests, and a JS double click sent one pair
  of requests, not two.

The real history was hashed before and after: the 115 rows are unchanged; a 116th
appeared at 17:06 — the owner's own OpenAI run (video, 75 s, $0.076) in their app on
port 8501.

Pitfalls:
- **The owner's app runs from this working tree** (started 13:41, still running), and
  Streamlit reloads changed modules on its next rerun — so the owner's 17:06 run used
  this uncommitted code, and the mutation checks, which rewrite `app.py` / `titles.py`
  for seconds at a time, could have served a planted bug to them. Mutation checks
  belong in a `git worktree`, not in the owner's checkout.
- A keyed `st.selectbox` in Streamlit 1.56 stores its value as the formatted label
  (`value_type="string_value"`), so a label that changes loses the choice (precisely:
  on the second click after the change — measured 2026-10-03, see §3); a keyed
  `st.expander` is identified by its label as well.
- `st.rerun(scope="fragment")` raises when the fragment runs as part of a full run, so
  work that changes a fragment's labels goes before they are drawn.
- A Bash heredoc mangled backslashes in Python edit scripts three times today (a `\n`
  escape became a real newline, a Windows path became a `\U` escape) — edits with
  backslashes go through the Edit or Write tool (the machine-wide lesson).

### 2026-09-29 (part 2) — Model dropdowns explain their options

The owner asked which model the Local sizes are, and for a note on the two OpenAI
models (which is newer, which costs more). The recorded costs of their 4 runs so far
(read-only aggregate) put gpt-4o-transcribe at $0.0027–0.0033 per minute — cheaper
than whisper-1's $0.006 — so the dropdown now reads "2025, more accurate · ~$0.004/min"
against "2023, exact timestamps & .srt · $0.006/min"; the local list is labelled
"Local Whisper model" with a help line. Hints were measured to fit the dropdown at
the owner's window width (§3 UI). 243 tests pass.

### 2026-09-29 — AI title for transcripts

The owner asked for a setting, chosen once, that has AI name each transcript from its
content, replacing the file name or added after it, with a choice of model, the same
key, and an Off option; after the proposal, "kreni", adding that Local runs may be
titled too. Built as `titles.py` (request, cleaning, display), a `title` column and a
`preferences` table in the history database, a sidebar section, the title's tokens in
the run cost, and History labels and download names per mode (decisions in §3 "AI
title"). 32 new tests (242 in total); 18 of 18 bugs put back were caught. Checked by
eye on an isolated instance (seeded rows, OpenAI pointed at a dead address): the
sidebar setting, the append and replace labels and the original name in the entry's
caption; the mode change was stored. The real database was not touched (hash
unchanged). The owner then ran it live and approved it (§5 "Waiting for the owner":
an English meeting got a fitting title for 1,146 → 9 tokens); at the owner's request
the title under the result is labelled "AI title:".

### 2026-09-27 (part 2) — S1, S2, S4 fixed; S3 studied and left as it is

**S1:** a frame whose JPEG the scan shares is now taken from the video again at its own
time just before it is described (`frames.pictures`); on real ffmpeg the extracted
JPEG was byte-identical to what the scan had written (80/80 frames, two synthetic
videos), 0.2–0.3 s each. **S2:** the scan selection no longer widens the interval to
fit the cap; the cap spreads the frames and the caption says so with both numbers.
**S4:** `extract_keyframes` deleted; a scan records its threshold and grid. Each fix
has a regression test that fails when its bug is put back (9 of 9). 192 tests pass.

**Second review, then fixes.** Four lenses (scan, UI, cost, docs) with a skeptic
per finding, each agent on its own copy of the code with no `.env` and a dead
`OPENAI_BASE_URL`: 26 findings confirmed, 1 refuted. The four medium ones: MPEG-TS/PS
seeks land late (C0), a Start clicked during a scan ran without screenshots (C4),
reruns during a scan started parallel scans (C5), and a Stop during a scan locked the
page (C6). All 26 are fixed or corrected in the docs; 17 regression tests, each shown
to fail with its bug put back. 210 tests pass.

**S3:** a study with 9 agents (synthetic corpus with exact ground truth, three rule
families, two judges, cross-validation) produced two rules that were near-perfect on
synthetic video. Measured on the owner's five real recordings (counts only, in an
isolated folder, derivatives deleted afterwards; real folders verified unchanged),
then — with the owner's explicit OK — 13 disputed frame pairs viewed: 12 were Teams
UI changes, not content. Not shipped; reasons in §3, follow-up in §5 P3. The laptop
slept for six hours in the middle of the study (Modern Standby), which is why it took
most of the day.

### 2026-09-27 — Scan review: two regressions found, commit held

The adversarial review of the uncommitted scan work ran partly (the session ended
mid-run; journal `wf_6030706f-ea1`). The scan lens found, and a skeptic per finding
reproduced on synthetic videos, one high and one medium regression plus two smaller
items — listed with the corrected fixes in §5 (Done 2026-09-27). The main one: sharing one JPEG
across near-identical frames made the describer see an earlier screen than the time on
the note (same-template slides collide at dhash distance 0–1). The UI lens did not
finish. Nothing is committed yet; the owner's approval to commit and push stands.

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

**Not verified live, because the key has no credit:** the OpenAI API answered
`429 credit_balance_exhausted` after the owner's purchase (06:18 and again 07:10). So the
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

**Evening — costs, billing, script, screenshots.** Asked by the owner:
- *Record real tokens and cost per run* — built (§3), reviewed by 2 reviewers + a
  skeptic per finding (4 confirmed: a kept screenshot cache counted twice after a
  saved run, a partial run's spend invisible, the audio length doubled on a split,
  pre-change checkpoints counted as free / History dropping `≈`; all fixed with tests).
- *Why "no credit" with $5 on the dashboard* — the key in `.env` belonged to a different
  OpenAI login; read-only look at the dashboard (API keys, projects, organizations,
  billing history). The owner swapped the key.
- *Test with a small video* — a 30 s synthetic clip through the real API: $0.0023, all
  paths working (§5). Three more one-frame requests measured image tokens ($0.003).
- *Keep the original script, restore the old records* — `SERBIAN_LATIN` now defaults to
  off; the 10 converted rows were restored from the pre-conversion copy after another
  verified backup (only rows untouched since, 10 of 10; the other 92 identical).
- *Is the screenshot estimate right, and are frames cut right?* — cutting: every kept
  frame matches the slide on screen at its time; but the burst rule could stamp a new
  slide a whole interval late (fixed, §3). The estimate counts only interval samples:
  on the owner's longest real video (115 min, mostly static after minute 10) it said
  29 at 245 s and 116 at 60 s while 10 were kept either way, because identical frames
  are dropped; on a busy screen, scene changes add frames the estimate never counted.

**Exact screenshot count** (approved by the owner after the measurements above). The
video is scanned once when on-screen context is ticked; the caption's count comes
from the scan and the run describes exactly those frames (checked in a browser with a
fake client: 4 shown, 4 described). On the real recordings the scan-based counts match
the old extraction within one frame (115 min: 11 vs 10; 16 min: 28 vs 28 at 245 s,
45 vs 46 at 30 s), and the scan takes about what the extraction took.

**Later the same day:** the owner approved the Serbian backfill — 10 records converted
after a verified backup, the other 92 byte-identical — and the commit. The owner's own
test on a 1:55 video hit the empty balance as well; a scan of every transcript of this
session (main and all agents) found no successful paid request, so the missing credit
is a billing question, not spending by the tests. From now on agents never use the
owner's key for tests (fake clients only). The cause, found afterwards in the
dashboard: the key in `.env` belongs to a different OpenAI account than the one the
owner topped up (§5 → "Waiting for the owner").

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
the deletion of rows 48–49, and topped up an OpenAI account ($5, bought by the owner — to a different account than the key's, as it later turned out;
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
