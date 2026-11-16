---
name: video-to-text
description: Transcribe a recording (video or audio) with this repository's video-to-text app, from its command line - find the newest recording in a folder, say what it will cost, run the transcription, check that no speech was lost, read the transcript and report what was said. Use when the user asks to transcribe a recording, wants the text or subtitles of a meeting, a call or a voice message, or asks what was said in a recording that has no transcript yet.
---

# video-to-text from Claude Code

This skill drives the app through its command line; the page (Streamlit) does not have to
be running. It knows the app only. All options are in the repository's `README.md`,
section "Command line": read it when a request needs an option that is not below.

## The command

| From | Call |
|---|---|
| Bash | `"${CLAUDE_SKILL_DIR}/../../../scripts/v2t.sh" <command> "<path>"` |
| PowerShell | `& "${CLAUDE_SKILL_DIR}\..\..\..\scripts\v2t.bat" <command> "<path>"` |

The path goes from this folder (`.claude/skills/video-to-text`) to the root of the clone.
If the skill was copied somewhere else, put the clone's path in its place. The app must be
installed as the README says (its own `.venv`, ffmpeg); the OpenAI engine reads the key
from the clone's `.env`. Never print that key.

The answer is one JSON object on standard output; progress goes to standard error.

| Command | What it does |
|---|---|
| `latest <folder> [--count N]` | The folder's newest recordings, each with `duration_seconds` and `transcript` (null when it has none). |
| `pending <folder>` | The recordings with no `transcript_….txt` beside them. |
| `estimate <recording>` | Length and cost of the run asked for. Starts nothing, needs no key. |
| `transcribe <recording>` | What Start does on the page; leaves `transcript_<recording name>.txt` beside the recording. |

| `status` | Exit code | What to do |
|---|---|---|
| `done` | 0 | Read the file named in `transcript`; pass every line of `notices` on to the user. |
| `failed` | 1 | Nothing was saved. Pass `message` on. |
| `partial` | 3 | Stopped part-way (no credit, no connection). The same command again sends only the rest. |
| `exists` | 4 | The recording already has a transcript: read that one, do not run again (`--overwrite` only when the user asks for a new one). |
| `needs_confirmation` | 4 | The estimate is above the limit ($1 by default), or the recording does not say its length. Tell the user the amount, and repeat with `--yes` only after their yes. |

## Flow

1. **Find the recording.** A path or a name from the user is that recording. For "the
   newest", `latest <folder> --count 5` and take the newest one whose `transcript` is null;
   say which one you took (name, length, when it was recorded). If several new ones have no
   transcript and it is not clear which is meant, list them and ask. For "everything that
   waits", show the `pending` list and ask which: do not run them one after another.
2. **Estimate.** `estimate "<recording>"`, then one sentence to the user: length and cost.
   Below the limit the run goes ahead without a question, because the user asked for it.
   Never pass `--yes` or `--max-cost` on your own.
3. **Transcribe.** `transcribe "<recording>" 2><a file for standard error>`, with the default
   model and no other options unless the user asked for them. Fifteen minutes of recording
   take about 40 seconds, an hour about two and a half minutes; start anything longer than
   half an hour in the background and report when it ends. Options, when they are asked for:
   `--from` and `--to` for a part of the recording, `--screens` when what was on the screen
   matters (it costs more, `estimate` says how much), `--srt` for subtitles (`whisper-1` or
   the local engine), `--engine local` for a run with no key and no internet.
4. **Check that no speech was lost.** OpenAI sometimes returns a part with most of its
   speech missing and no error. The app measures words per minute of speech for every part
   (silence is measured and left out, so a quiet stretch is not flagged), sends a thin part
   again in two halves, and writes what it could not repair into `notices`. Look at the
   `Part 00N … words/min of speech; sound: … s speech, … s silence …` lines in the standard
   error file and at `notices`; a `Part 00N is thin … its answer: "…"` warning quotes what
   OpenAI first returned for a part it sent again (kept in full as `first_answer` in the
   run's saved parts). Two cases are left to you:
   - `notices` says "Speech may be missing…" (a half came back thin the second time too);
   - the recording is shorter than five minutes, so it is one part with nothing to compare
     it with, and the words are few for its length.

   Then say so, and offer to repeat with `--verify medium --overwrite`: a local Whisper
   pass over the same recording adds what the OpenAI answer lacks (paragraphs marked `🧩`)
   and gives the paragraphs their real times. It costs no money, only time (`medium` on a CPU
   took about eleven minutes for a fifteen-minute recording), so it goes in the background,
   and it needs the offline backend installed (README, "Offline mode"). The repeated run adds
   another row to the app's History; the old row stays. The same option is worth offering
   up front when the user says the meeting matters.
5. **Read the transcript and report.** What the conversation was about in three to five
   sentences, what was agreed, and the next steps (who, what, by when, if it was said). Do
   not fill in what the transcript does not say. End with where the transcript is and what
   the run cost (`cost_usd`).

## Rules

- A transcription starts only when the user asks for one. No scheduled or automatic pass
  over a folder: a folder of recordings holds many that nobody wants transcribed, and each
  run is paid.
- Recordings are usually confidential conversations. Their content goes nowhere except
  what the app itself sends to OpenAI. Do not paste a transcript into another repository,
  a ticket or a message unless the user asks for exactly that.
- Recordings and transcripts are not deleted, renamed or moved unless the user asks.
- In real use the command writes to the app's real History, log and saved parts, and that
  is intended. A trial is not real use: synthetic sound, and `DATA_DIR`, `TEMP_DIR` and
  `UPLOAD_DIR` set to a scratch folder (the repository's `CLAUDE.md`).
  `scripts/dev/fake_openai.py` stands in for OpenAI when the point is not to check the key.
- A fault in the command is reported to the user with the JSON answer and the standard
  error file. Do not work around it by editing the result.

## Your own folders belong in your own skill

Where your recordings appear, which folder each one goes to afterwards and how you name
them is not something this repository can know. Keep that in a personal skill
(`~/.claude/skills/<name>/SKILL.md`) that sends Claude here for the commands and adds your
folders and your steps before and after. A personal skill with the same name as this one
takes its place in your own sessions.
