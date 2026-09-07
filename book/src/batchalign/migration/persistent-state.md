# Persistent State and Behavioral Changes

**Status:** Current
**Last updated:** 2026-09-06 23:59 EDT

Batchalign3 introduces several stateful behaviors that did not exist in
Batchalign2. This page documents every form of persistent state, where it
lives, and how it differs from BA2's stateless model.

## Overview: BA2 vs BA3 execution model

**BA2:** Every invocation was a fresh Python process. Models loaded from
scratch, results computed from scratch, nothing persisted between runs. This
was simple but slow, re-processing the same file paid the full cost every
time.

**BA3:** The Rust CLI manages persistent state across runs: a local daemon
keeps models warm, a SQLite cache stores analysis results, and ML models are
cached on disk. This makes repeated operations dramatically faster but
introduces state that users need to understand.

## Persistent state locations

| State | Location | What it stores |
|-------|----------|---------------|
| Analysis cache | platform-dependent OS cache dir (see below) | SQLite database of NLP results keyed by content hash |
| Model cache | platform-dependent (see below) | Downloaded ML model weights (~2 GB) |
| Config file | `~/.batchalign.ini` | ASR engine selection, Rev.AI API key |
| Run logs | platform-dependent OS cache dir | Per-run structured logs |
| Daemon PID | platform-dependent | Background process state |

### Analysis cache and run-log locations

Both live under the OS-conventional cache directory for the
`batchalign3` application:

- **macOS:** `~/Library/Caches/batchalign3/` (cache.db + logs/)
- **Linux:** `~/.cache/batchalign3/` (cache.db + logs/)
- **Windows:** `%LocalAppData%\batchalign3\` (cache.db + logs/)

### Model cache locations

Models are stored by the ML libraries (Stanza, Whisper, etc.) in their
default cache directories:

- **macOS:** `~/Library/Caches/` (Stanza), `~/.cache/whisper/` (Whisper)
- **Linux:** `~/.cache/stanza/`, `~/.cache/whisper/`
- **Windows:** `%LOCALAPPDATA%\stanza\`, `%USERPROFILE%\.cache\whisper\`

## Analysis cache

The analysis cache is the largest behavioral difference from BA2. BA3
caches **audio inference results only**: UTR ASR (`utr_asr`) and
forced alignment (`forced_alignment`). Text-NLP commands (`morphotag`,
`translate`, `utseg`, `coref`) deliberately do not cache; each run
recomputes from scratch so that model or pipeline changes always take
effect immediately.

Audio cache keys combine media identity with the parameters owned by each
cache layer. Media identity currently uses canonical path, modification time and size;
it is not a content digest. The UTR normalized-response keys include the
selected ASR provider, language and, for segments, window bounds. Provider-free
legacy UTR entries are not automatically replayed because their producing
backend is unknown.

- Repeated alignment can reuse matching retained UTR and FA entries, but still
  performs orchestration and may load workers. Reuse does not imply instant output.
- Transcript edits can reuse raw ASR while changing downstream FA groups.
- UTR matching tuning changes projection of retained ASR; it does not change
  the raw ASR request. Selecting another UTR provider changes its key.
- Symlinks and alternate spellings of the same path share media identity.
  Moving identical audio can miss the metadata-based identity. Changing only
  FA can also miss UTR's current outer worker-version partition.
- Text-NLP commands re-run rather than returning cached analysis.

The [forced-alignment cache reference](../reference/forced-alignment.md#utr-asr-cache)
describes the two key spaces and their current limitations.

**Managing the cache:**

```bash
batchalign3 cache stats        # Show cache size and entry count
batchalign3 cache clear        # Delete cached results (with confirmation)
batchalign3 cache clear --all  # Also remove permanent UTR cache entries
```

**BA2 had no cache.** Every invocation computed results from scratch.

## Local daemon

When you run a processing command, the CLI may start a background daemon
process that keeps ML models loaded in memory. This eliminates model loading
time on subsequent runs (5-20x speedup).

**When this matters:**

- The daemon uses memory even after your command finishes
- It persists across Python process exits (important for compat shim users)
- Multiple concurrent commands share the same daemon

**Managing the daemon:**

```bash
batchalign3 serve status       # Check if daemon is running
batchalign3 serve stop         # Stop the daemon (frees memory)
batchalign3 serve start        # Start the daemon explicitly
```

**BA2 had no daemon.** Every invocation loaded models from scratch.

## ML model downloads

The first time you run a processing command, ML models are downloaded
automatically. This is a one-time cost of ~2 GB. Subsequent runs use cached
models from disk.

**When this matters:**

- First run of `morphotag` downloads Stanza models (~500 MB)
- First run of `align` downloads Whisper/Wave2Vec models (~1-2 GB)
- No network connection needed after first download
- The download itself surfaces through `progress_v2` events on every
  UI channel; there is no separate pre-warm CLI command in BA3

**BA2 also downloaded models on first use**, but the behavior is the same.

## Config file

`~/.batchalign.ini` stores the default ASR engine selection and API keys.
Created by `batchalign3 setup`. This is the same format as BA2.

## Implications for subprocess integrations

There is no public Python API in BA3; the supported integration path
from Python is `subprocess`-into-`batchalign3`. The Python `compat`
shim (`batchalign.compat.BatchalignPipeline`, etc.) has been removed
along with the rest of the BA2 Python API, see
[Developer Architecture Migration](developer-migration.md#7-python-api-migration).

For scripts that drive `batchalign3` via subprocess, the persistent-state
points still matter:

1. **The first call may be slow**: models download and daemon starts.
2. **The daemon persists**: after your driver process exits, the daemon
   continues running. Stop it explicitly with `batchalign3 serve stop`
   if you don't want it.
3. **Audio-task results are cached**: re-aligning identical media is
   near-instant; text-NLP results recompute every run.
4. **Memory usage**: the daemon holds ML models in memory (~2-4 GB)
   until you stop it.

To disable the audio cache for a specific run (BA2-like always-recompute
for the cached tasks):

```bash
batchalign3 align ~/corpus/ -o ~/output/ --override-media-cache
```

The `--override-media-cache-tasks <list>` per-command flag offers
finer-grained control. Text-NLP commands never cache, so there is no
analogous flag for them.

## Clearing all state

To reset to a clean state:

```bash
batchalign3 serve stop            # Stop daemon
batchalign3 cache clear           # Clear analysis cache
batchalign3 logs --clear          # Clear run logs
# Model caches are managed by ML libraries; delete manually if needed
```
