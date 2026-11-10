---
name: transcribe-recording
description: >-
  Run the meeting-summarizer transcription pipeline on an audio or video recording, from any
  repo or session. Use this whenever the user provides a path to a media file (.mp4, .mov, .mkv,
  .webm, .m4v, .avi, .wmv, .flv, .mp3, .wav, .m4a, .flac, .aac, .ogg), pastes a Google Drive
  link to a meeting recording, or asks to "transcribe", "summarize", or "run the process" on a
  recording, meeting, call, or video — even if they only paste a filename or link with no other
  instruction, and even when the current working directory is another repo. It converts the media to
  audio, transcribes it with Whisper (local) or the Gemini API, and produces a cleaned
  English translation plus a structured summary via the `claude` CLI, saved as a Markdown
  file next to the input. Use the Gemini engine (`--engine gemini`) whenever the user asks
  to transcribe with Gemini, use Gemini instead of Whisper, or mentions the Gemini API.
---

# Transcribe a recording

The meeting-summarizer repo (`/Users/ido.michealovich/Documents/GitHub/meeting-summarizer`)
transcribes and summarizes audio/video meetings. The single entrypoint is
`transcribe_audio.py`. Your job when a user hands over a media file is to run it correctly,
in the background, and report the result.

This skill is global (symlinked into `~/.claude/skills/` from that repo), so the session may be
in another repo. Always `cd` into the meeting-summarizer repo in the same command that runs the
script; recordings and their outputs live in that repo's root.

## Input from a Google Drive link

1. Read the file id from the URL (`/file/d/<id>/view`) and fetch its metadata with
   `mcp__claude_ai_Google_Drive__get_file_metadata` (`excludeContentSnippets: true`) for the
   title, MIME type and size.
2. Download it with `mcp__google-workspace__get_drive_file_download_url` (`file_id`). It saves to
   `~/.workspace-mcp/attachments/` under the title with a UUID suffix and no extension.
3. `mv` it into the repo root as `<title>.<ext>`, replacing `/` in the title with `_`, and with
   the extension taken from the MIME type (`video/mp4` → `.mp4`, `audio/mpeg` → `.mp3`).
   The output Markdown lands next to it.

## The command

Always run through the project's conda interpreter — a bare `python` will not have `whisper`,
`torch`, etc. installed:

```bash
cd /Users/ido.michealovich/Documents/GitHub/meeting-summarizer && \
  /opt/anaconda3/envs/meeting-summarizer/bin/python transcribe_audio.py "<path-to-file>"
```

Quote the path — recording filenames routinely contain spaces and parentheses
(e.g. `qfh-uvez-wpb (2026-07-12 16_03 GMT+3).mp4`).

Run it in the **background** and redirect output to a log. Transcription is the long pole
(see timing below), so a foreground call will look hung.

### Useful flags (add only when the user asks)

- `--engine gemini` — transcribe with the Gemini API instead of the local Whisper model.
  Use this only when the user asks for Gemini / "use Gemini instead of Whisper": it sends the
  audio of an internal meeting to an external API, while Whisper keeps it local. It uploads the
  audio to Gemini and returns a timestamped transcript in the original language; the `claude`
  clean/translate + summarize steps still run on top (Gemini replaces only stage 3). Requires
  `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment — if it's unset the run fails fast
  with a clear error, so check before a long job. Optional `--gemini-model <name>` overrides the
  model (default `gemini-2.5-pro`; `gemini-2.5-flash` is faster/cheaper, slightly less accurate).
  Gemini handles the whole file natively, so `--multiprocessing` and silence-trimming don't
  apply, and it has no per-segment confidence scores. Timestamps are ~second-accurate (Whisper's
  are exact).
- `--description "<context>"` — meeting context; improves the summary. Offer this if the user
  describes what the meeting was about.
- `--language he` / `--language en` — skip auto-detection when the user already knows the language.
- `--max-duration <minutes>` — transcribe only the first N minutes (good for a quick test).
- `--no-gpt` — Whisper transcript only, skip the `claude` clean/translate + summarize steps.
- `--multiprocessing` — parallel chunked transcription. Only helps on a machine with plenty of
  free RAM; each `large-v3-turbo` worker needs ~4 GB, so on a typical laptop the memory-safe
  cap collapses to 1 worker and it's no faster than the default single-process path. Don't
  bother unless RAM is clearly ample.

## Preflight

Confirm the toolchain before a long run so you fail fast instead of 30 minutes in:

```bash
ls -la /opt/anaconda3/envs/meeting-summarizer/bin/python   # interpreter exists
which ffmpeg ffprobe claude                                 # all three on PATH
```

`ffmpeg`/`ffprobe` do the media conversion; `claude` does the clean/translate + summary.
If `claude` is missing or not logged in, the transcription still works but the two refinement
steps fail — warn the user.

## Pipeline stages & timing

| Stage | What happens | Rough time |
|-------|--------------|-----------|
| 1. Convert | `ffmpeg` extracts audio from video (skipped for audio inputs) | ~1 min |
| 2. Detect language | Whisper `base` samples the first segment | ~1 min |
| 3. Transcribe | Whisper `large-v3-turbo` over the whole recording (CPU) | **~0.7× the audio length** (a 40-min recording ≈ 30 min) |
| 4. Clean + translate | `claude` (Sonnet 5) → fluent English, timestamps preserved | a few min |
| 5. Summarize | `claude` (Sonnet 5) → structured meeting summary | a few min |

Give the user a duration estimate up front using stage 3 as the anchor. You can read the
audio length first with `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "<file>"`.

With `--engine gemini`, stages 2–3 are replaced by a single Gemini API call (upload + generate,
usually a few minutes and far less sensitive to recording length, up to Gemini's ~9.5 h limit);
stages 4–5 are unchanged. No local Whisper model is loaded.

## Output

A Markdown file next to the input: `<basename>_transcription_<YYYYMMDD_HHMMSS>.md`, containing,
in order: **Conversation Summary** → **Cleaned & Translated Transcription** (English, timestamped)
→ **Original Whisper Transcription** (source language, with confidence scores). A metadata header
records the model, detected language, and which steps ran.

Report the output path and a short preview of the summary when done.

## Gotchas we've actually hit

- **Refinement derailment.** The `claude` clean/translate + summarize steps run as an isolated,
  tool-less one-shot (see `run_claude_cli_prompt` in `transcribe_audio.py`). If a refinement
  section ever comes back as agent narration instead of the expected text — e.g. *"I'll check
  the project's output conventions..."*, a shell command, or a PII/redaction preamble — the
  isolation has regressed and the `claude` call is picking up the ambient `CLAUDE.md` / skills /
  org context. The Whisper transcript is still fine; re-run just the refinement (below) rather
  than the whole thing.
- **Re-run refinement cheaply.** Whisper is the expensive part. If transcription succeeded but the
  summary/translation is bad, don't re-transcribe — re-run only the `claude` steps against the raw
  Whisper section of the existing file:
  ```bash
  /opt/anaconda3/envs/meeting-summarizer/bin/python transcribe_audio.py dummy \
    --re-refine "<existing_transcription>.md"
  ```
  This overwrites that file in place (~a few min). `--resummarize <file>` regenerates only the
  summary from the already-cleaned text.
- **`pytest` isn't installed** in this env; the tests are `unittest`-based. Run them with
  `/opt/anaconda3/envs/meeting-summarizer/bin/python -m unittest tests.test_transcribe_audio`.
- **README is stale** — it mentions GPT-5 / `OPENAI_API_KEY`, but the code uses the `claude` CLI
  (Sonnet 5). Trust the code, not the README.
