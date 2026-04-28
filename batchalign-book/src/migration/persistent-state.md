# Persistent State and Behavioral Changes

**Status:** Current
**Last updated:** 2026-03-16

Batchalign3 introduces several stateful behaviors that did not exist in
Batchalign2. This page documents every form of persistent state, where it
lives, and how it differs from BA2's stateless model.

## Overview: BA2 vs BA3 execution model

**BA2:** Every invocation was a fresh Python process. Models loaded from
scratch, results computed from scratch, nothing persisted between runs. This
was simple but slow — re-processing the same file paid the full cost every
time.

**BA3:** The Rust CLI manages persistent state across runs: a local daemon
keeps models warm, a SQLite cache stores analysis results, and ML models are
cached on disk. This makes repeated operations dramatically faster but
introduces state that users need to understand.

## Persistent state locations

| State | Location | What it stores |
|-------|----------|---------------|
| Analysis cache | `~/.cache/batchalign3/cache.db` | SQLite database of NLP results keyed by content hash |
| Model cache | Platform-dependent (see below) | Downloaded ML model weights (~2 GB) |
| Config file | `~/.batchalign.ini` | ASR engine selection, Rev.AI API key |
| Run logs | `~/.cache/batchalign3/logs/` | Per-run structured logs |
| Daemon PID | Platform-dependent | Background process state |

### Model cache locations

Models are stored by the ML libraries (Stanza, Whisper, etc.) in their
default cache directories:

- **macOS:** `~/Library/Caches/` (Stanza), `~/.cache/whisper/` (Whisper)
- **Linux:** `~/.cache/stanza/`, `~/.cache/whisper/`
- **Windows:** `%LOCALAPPDATA%\stanza\`, `%USERPROFILE%\.cache\whisper\`

## Analysis cache

The analysis cache is the largest behavioral difference from BA2. When you
process a file, Batchalign computes a BLAKE3 hash of the input content plus
command parameters. If the cache contains a result for that hash, it returns
the cached result without invoking any ML models.

**When this matters:**

- Re-running `morphotag` on the same corpus returns instantly
- Editing a file invalidates its cache entry (different content hash)
- Changing `--lang` or `--retokenize` invalidates the cache (different params)
- The cache is per-command: `morphotag` and `align` cache separately

**Managing the cache:**

```bash
batchalign3 cache stats        # Show cache size and entry count
batchalign3 cache clear        # Delete all cached results
batchalign3 cache clear --older-than 30d  # Delete entries older than 30 days
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
- Models can be pre-downloaded with `batchalign3 cache warm`

**BA2 also downloaded models on first use**, but the behavior is the same.

## Config file

`~/.batchalign.ini` stores the default ASR engine selection and API keys.
Created by `batchalign3 setup`. This is the same format as BA2.

## Implications for compat shim users

If you are using `batchalign.compat.BatchalignPipeline`, be aware that:

1. **The first call may be slow** — models download and daemon starts.
2. **The daemon persists** — after your Python process exits, the daemon
   continues running. Stop it explicitly if you don't want it.
3. **Results are cached** — re-processing identical input is near-instant,
   which is different from BA2's always-compute behavior.
4. **Memory usage** — the daemon holds ML models in memory (~2-4 GB).
   This persists until you stop the daemon.

To disable caching and daemon behavior entirely (BA2-like cold execution):

```bash
# Run without cache
batchalign3 --no-cache morphotag ~/corpus/ -o ~/output/
```

## Clearing all state

To reset to a clean state:

```bash
batchalign3 serve stop            # Stop daemon
batchalign3 cache clear           # Clear analysis cache
batchalign3 logs --clear          # Clear run logs
# Model caches are managed by ML libraries; delete manually if needed
```
