# README screenshots and demo GIF

Retakes `docs/screenshots/home.png`, `transcribe.png`, `history.png` and `demo.gif` in
under three minutes, on made-up recordings, without touching the real `data/`, `temp/`
or `uploads/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dev\readme-screenshots\run.ps1
```

Then look at the images and the GIF, and `git diff --stat docs/screenshots`.

## What it does

1. **Speech** — Windows text-to-speech (voices *Microsoft Zira* and *David*) reads a
   sprint-planning dialogue of about 1:36.
2. **Recordings** (`demo.py media`) — `Sprint planning.mkv`: the dialogue over a slide,
   then room noise to 9:25 under a "Meeting ended" slide (a recorder nobody stopped);
   four more files of silence give the folder list believable lengths. Their dates are
   set as the files' creation time, which is what the list shows.
3. **App copy** (`demo.py copy-app`) — the repo's tracked files as they are in the
   working tree, so `.env` (the API key), `data/`, `temp/` and `uploads/` stay behind.
4. **Instance** — that copy on `127.0.0.1:8597` with its own data, temp and upload
   folders under `-Out`, no key, `OPENAI_BASE_URL` pointing nowhere, models read from
   the repo's `models/` (offline). It starts on the folder source (`demo.py prefs`).
5. **Screenshots** (`shoot.mjs`, puppeteer + Chrome, 1320 px, light theme):
   - `home` — the recording list opened;
   - `transcribe` — the meeting chosen, the slider's right end moved to 1:40 with the
     keyboard, a **Local `base`** run with timestamps (no OpenAI), the whole page;
   - `history` — after `demo.py history` adds four made-up older rows (OpenAI and local,
     with titles and costs) and turns AI titles on, one entry opened.
6. **Crop** (`demo.py crop`) — the empty bottom cut off, saved into `docs/screenshots`.
7. **Video** (`shoot.mjs video`, 1200×760, sidebar folded) — the transcribe flow again
   with the real mouse (a pointer and a click ring are drawn, since headless Chrome shows
   none): open the list, pick the meeting, drag the slider's right end to 1:40, Local
   with timestamps, Start, the progress, the transcript. Frames are screenshots taken
   about every 0.1 s with their own times (Chrome's screencast stopped sending frames once
   a run started), listed in `-Out\video\demo.json` with the stretches to time apart.
8. **GIF** (`demo.py gif`) — the frames joined at the speed they were taken, 1.6× faster,
   except the waits (preparing the recording, redrawing, the run itself), squeezed to at
   most 0.4–2.5 s, and the pauses to read, at real speed; 960 px, 12 fps, a 128-colour
   palette: about 27 s and 1 MB. `-Out\video\demo.mp4` is the same at 25 fps — GitHub
   shows a video only when it is uploaded through the web editor, so it is not used.

The instance is stopped at the end (by process name and port), also on failure.

## Notes

- The *Recordings folder* field shows `C:\Users\you\Videos`: a label is drawn over the
  field's text (the demo folder is a long temp path); nothing is typed into it.
- Requirements: Windows (text-to-speech, PowerShell), the repo's `.venv` with the local
  engine and the `base` model already downloaded, `ffmpeg` on PATH (or `C:\ffmpeg\bin`),
  Node with `puppeteer-core` (`PUPPETEER_FROM`, default `C:/Work/tools/`) and Chrome
  (`CHROME`).
- Options: `-Out <folder>` (default `%TEMP%\video-to-text-screenshots`, deleted and made
  again on each run — only if this script made it), `-Port`, `-TrimEnd <seconds>`,
  `-NoVideo` (screenshots only), `-Keep` (leave the instance running; the next run stops
  it).
- To redo one step: run with `-Keep`, then `node shoot.mjs <url> <mode> <out>` against
  the running instance. If a step fails, `-Out\instance.err` has the app's log.
