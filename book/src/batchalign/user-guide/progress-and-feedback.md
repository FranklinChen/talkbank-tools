# Progress and Feedback

**Status:** Current
**Last updated:** 2026-05-11 10:31 EDT

Batchalign reports real-time progress during processing. This page explains what
to expect for each command, what the progress indicators mean, and when to worry
versus when to wait.

## How Progress Works

Every processing job tracks progress at the **file level**. In server mode, the
server reports stage transitions and optional sub-file counters (e.g.,
"Aligning 3/7 groups") to all connected clients — the CLI, TUI, and React
dashboard all consume the same stream. In direct local mode, the CLI now
projects the same file-status snapshots from the in-memory direct host, so
local runs still show live terminal progress without requiring a dashboard.

For direct local runs, the CLI also prints a stable debug handle at startup:

- the direct job ID
- the local artifact directory for that job

On failures, the CLI prints any persisted bug-report IDs and direct debug
artifact paths so you can inspect the failed run later without keeping the
process alive.

There are two progress tiers:

- **Stage labels** — every file shows a stage name ("Reading", "Aligning",
  "Writing") that changes as processing advances.
- **Sub-file counters** — some stages include a current/total counter for
  fine-grained progress within a single file.

## Per-Command Expectations

### align (forced alignment)

Align processes files individually and concurrently. Each file goes through:

1. **Reading** — loading the CHAT file from disk
2. **Resolving audio** — finding and preparing the media file
3. **Recovering utterance timing** (if needed) — re-transcribing to recover
   word timing for untimed utterances. Shows sub-progress for partial-window
   UTR (e.g., "2/5" windows). This step takes roughly as long as the
   recording itself.
4. **Aligning** — forced alignment on utterance groups. Shows sub-progress
   (e.g., "3/7" groups).
5. **Writing** — saving the aligned output

**Timing:** Most of the time is spent in steps 3-4. A 10-minute recording
typically takes 5-15 minutes depending on the engine and number of utterances.

### transcribe

Transcribe processes files individually. Each file goes through:

1. **Resolving audio** → **Transcribing** → **Post-processing** →
   **Building CHAT** → optional **Segmenting** / **Morphosyntax** →
   **Finalizing** → **Writing**

Shows a pipeline stage counter (e.g., "2/5") as each stage completes.

**Timing:** Rev.AI runs roughly in real-time. Whisper may take 2-5x the
audio length.

### morphotag, utseg, translate, coref (batched commands)

These commands batch **all files together** into a single inference call for
GPU efficiency. Progress stages:

1. **Reading** — files are loaded one at a time; each transitions from
   the initial stage to "Reading" during I/O.
2. **Analyzing/Segmenting/Translating** (0/N) — the batch total is published
   before inference starts. During inference, the progress bar shows the batch
   size but individual files don't advance.
3. **Writing** (1/N, 2/N, ...) — as each file's result is written to disk,
   the counter ticks up.

For large in-place reruns driven by `--file-list`, the batched text commands
may still stage the rewritten CHAT files until the current invocation
finishes. In other words: you can see healthy progress without seeing the
input `.cha` files mutate on disk yet. If you want visible on-disk updates
during a long repair pass, split the rerun into smaller invocations.

**What "frozen" means:** During step 2, the progress bar won't advance because
all files are processed as a single batch. This is normal — the model is
working on your entire corpus at once. The elapsed timer keeps ticking to
confirm the app is alive.

**Timing:** Depends on corpus size. 50 files typically takes 1-5 minutes for
morphotag, faster for translate and utseg.

## When to Worry vs. When to Wait

**Normal:** Progress frozen during batch processing, or during UTR/transcription
(these are genuinely long-running). The elapsed timer should always be ticking.

**Investigate if:**
- The elapsed timer stops advancing (app may have frozen — try refreshing)
- A file stays in "Reading" for more than 30 seconds (possible I/O issue)
- "Resolving audio" persists for minutes (media file may be missing)

## How to Cancel

- **Desktop app:** Click the red "Cancel" button in the progress view
- **CLI:** Press `Ctrl+C` (graceful shutdown)
- **API:** `POST /jobs/{id}/cancel`

Cancellation is cooperative — the current file finishes its in-progress work
before the job stops.

## Pipeline Phase Indicator

For processing files, the dashboard and desktop app show a compact 5-segment
phase bar that maps the 23 internal progress stages into visual phases:

| Segment | Pipeline Phase | Stages Included |
|---------|---------------|-----------------|
| 1 | **Read** | Reading, Resolving audio, Checking cache, Parsing |
| 2 | **Transcribe** | Transcribing, Recovering utterance timing, Recovering timing (fallback) |
| 3 | **Align** | Aligning, Applying results |
| 4 | **Analyze** | Morphosyntax, Segmentation, Translation, Coreference, Comparing, Benchmarking |
| 5 | **Finalize** | Post-processing, Building CHAT, Finalizing, Writing |

The 23 `FileProgressStage` variants map to 5 visual phases via `phase_index()` in `crates/batchalign/src/cli/tui/ui.rs`; two variants (`Processing` generic fallback and `RetryScheduled`) deliberately do not map to a phase.

The active phase pulses; completed phases are filled; future phases are gray.
Not every command uses all phases — `morphotag` skips Transcribe and Align;
`align` skips Transcribe and Analyze.

## Progress Displays

| Client | What you see |
|--------|-------------|
| **Web dashboard** | Two-column layout: job list with pipeline phase bars, system panels (workers, memory, vitals). See [Web Dashboard](dashboard.md) for details. |
| **Desktop app** | Processing progress view with SSE-driven file list and stage labels |
| **CLI** | indicatif progress bar with file count, elapsed time, and per-file terminal logs for both server-backed and direct local runs |
| **TUI** | Per-file spinners with pipeline phase dots, elapsed timers, status breakdown, ETA, worker status, memory gauge, scroll indicators (server-backed jobs only) |

### TUI Details (`--tui`)

The TUI shows the same information as the web dashboard in a terminal-friendly
format:

- **Header** with status breakdown (`3✓ 2⠋ 1✗ 44·`), elapsed time, and ETA.
  Shows "Done!" or "Done — N failed" on completion.
- **Pipeline phase dots** next to each processing file: `●●●○○` maps the 5
  phases (Read/Transcribe/Align/Analyze/Finalize) using the same grouping as
  the web dashboard. Green = completed, cyan = active, gray = future.
- **Per-file elapsed timer** (`M:SS`) on processing files, computed from
  `started_at`. Helps spot stuck files during long align/transcribe jobs.
- **Worker status line** between the header and file list: shows active worker
  keys and warmup status.
- **Memory gauge** below the worker line: 20-char bar with used/total GB and
  gate proximity indicator (safe/warn/danger). Warns explicitly when memory
  is near or below the gate threshold.
- **Scroll indicators** (`▲ N more above` / `▼ N more below`) when groups
  have more files than fit on screen.
- **Auto-collapse** for completed groups: non-focused groups where all files
  are done or errored show a condensed title.
- **Error codes** in the error panel, extracted from the server's structured
  error codes (e.g. `[E362] Bullet timestamps must be monotonic`).

**Keybinds:** `q` quit · `c` cancel · `↑↓` scroll · `tab` group · `e` errors · `m` metrics

Press `m` to toggle the worker/memory rows on small terminals.
