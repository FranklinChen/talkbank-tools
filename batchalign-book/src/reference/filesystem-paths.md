# Filesystem Paths Used by batchalign3

**Status:** Current
**Last updated:** 2026-03-15

All current filesystem paths used by the public `batchalign3` runtime.

Unless otherwise noted, paths are rooted under `batchalign_app::config::ba_state_dir()`,
which defaults to `~/.batchalign3` and can be overridden with
`BATCHALIGN_STATE_DIR`.

## Configuration

| Path | Purpose | Defined in |
|------|---------|------------|
| `~/.batchalign.ini` | User config shared with older tooling (for example Rev.AI credentials and default ASR selection) | `batchalign/config.py`, `crates/batchalign-cli/src/setup_cmd.rs` |
| `~/.batchalign3/server.yaml` | Server/daemon configuration | `crates/batchalign-app/src/types/config.rs` |

## Runtime data

| Path | Purpose | Defined in |
|------|---------|------------|
| `~/.batchalign3/logs/` | Structured CLI run logs (`run-*.jsonl`) and exported log zips | `crates/batchalign-cli/src/logs_cmd.rs` |
| `~/.batchalign3/server.pid` | PID file for manual `batchalign3 serve start` | `crates/batchalign-cli/src/serve_cmd.rs` |
| `~/.batchalign3/server.log` | stderr log for manual `batchalign3 serve start` | `crates/batchalign-cli/src/serve_cmd.rs` |
| `~/.batchalign3/daemon.json` | main auto-daemon state | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/daemon.lock` | main auto-daemon startup lock | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/daemon.log` | main auto-daemon stderr log | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.json` | sidecar-daemon state for transcribe-heavy workloads | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.lock` | sidecar-daemon startup lock | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.log` | sidecar-daemon stderr log | `crates/batchalign-cli/src/daemon.rs` |
| `~/.batchalign3/jobs/` | per-job staging directories | `crates/batchalign-app/src/lib.rs`, `crates/batchalign-app/src/routes/jobs/mod.rs` |
| `~/.batchalign3/jobs.db` | SQLite job persistence database | `crates/batchalign-app/src/db/mod.rs` |

## Caches

### Analysis cache (SQLite)

Utterance-level cache entries for morphosyntax, utterance segmentation,
translation, and forced alignment.

The default path comes from `dirs::cache_dir()` in Rust and intentionally
matches the Python-side `platformdirs` location.

| Platform example | Path |
|------------------|------|
| macOS | `~/Library/Caches/batchalign3/cache.db` |
| Linux | `~/.cache/batchalign3/cache.db` |

Override with `BATCHALIGN_ANALYSIS_CACHE_DIR`, which relocates the database to
`$BATCHALIGN_ANALYSIS_CACHE_DIR/cache.db`.

Defined in:

- `crates/batchalign-app/src/cache/sqlite.rs`
- `crates/batchalign-cli/src/cache_cmd.rs`

### Media cache

Directory used for cached media artifacts and exposed through
`batchalign3 cache stats|clear`.

The default path comes from `dirs::data_dir()`.

| Platform example | Path |
|------------------|------|
| macOS | `~/Library/Application Support/batchalign3/media_cache/` |
| Linux | `~/.local/share/batchalign3/media_cache/` |

Override with `BATCHALIGN_MEDIA_CACHE_DIR`.

Defined in:

- `crates/batchalign-app/src/ensure_wav.rs`
- `crates/batchalign-cli/src/cache_cmd.rs`

## Legacy compatibility note

BA2-era tooling used:

- `~/.batchalign.ini`
- `~/.batchalign/`
- `~/.cache/batchalign/`
- `~/Library/Application Support/batchalign/media_cache/`

The current release intentionally still shares `~/.batchalign.ini`, but the
state directory, jobs DB, logs, daemon state, and caches otherwise use the
`batchalign3` prefix.
